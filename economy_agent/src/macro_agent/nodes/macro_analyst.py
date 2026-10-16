"""
Macro Analyst Node — 거시경제 정량 지표 분석 전담

실행 흐름:
    Step 1. state.ts_features  → ECOS / FRED 섹션 포맷 (DB 재조회 없음)
    Step 2. state.rag_docs     → 뉴스·리포트 컨텍스트 포맷
    Step 3. LLM (gpt-4o)      → build_macro_only_chain() 호출
    Step 4. Parse              → JSON 파싱 + 폴백
    결과를 state.macro_analysis_raw에 저장

설계 원칙:
    - DB 조회는 data_retrieval_node가 전담 → 이 노드는 순수 LLM 분석
    - geo_analyst_node와 병렬 실행 가능 (state 필드 충돌 없음)
    - warnings: 이 노드의 신규 경고만 반환 (Annotated 리듀서가 누적)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timezone
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from macro_agent.prompts.macro_prompt import build_chain_input, build_macro_only_chain
from macro_agent.state import AgentState

logger = logging.getLogger(__name__)

_N_NEWS    = 20
_N_REPORTS = 10


# ── LLM 싱글톤 ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=4096)


# ── Step 1: ts_features → ECOS / FRED 섹션 포맷 ──────────────────────

def _format_ts_section(ts_features: dict[str, dict], prefix: str) -> str:
    """
    state.ts_features에서 prefix에 해당하는 지표만 걸러 LLM 주입용 JSON으로 변환합니다.

    포함 필드:
        latest_value, unit, as_of, region           (기본)
        mom_change_pct, yoy_change_pct               (변동률)
        ma_30, ma_90                                 (이동평균)
        trend_label, trend_3m                        (추세)
        historical_percentile, volatility_3m         (통계)
    """
    filtered = {
        k: v for k, v in ts_features.items()
        if k.startswith(prefix) or (prefix == "fred" and k.startswith("market."))
    }
    if not filtered:
        return json.dumps({"status": "TimescaleDB 데이터 없음"}, ensure_ascii=False)

    output: dict[str, Any] = {"status": "TimescaleDB", "metrics": {}}
    for name, row in filtered.items():
        entry: dict[str, Any] = {
            "latest_value": row.get("current_value") if row.get("current_value") is not None else row.get("value"),
            "unit":         row.get("unit", ""),
            "as_of":        str(row.get("as_of") or row.get("time", ""))[:10],
            "region":       row.get("region", ""),
        }
        # 변동률
        for fld, label in (
            ("mom_change",     "mom_change_pct"),
            ("mom_change_pct", "mom_change_pct"),
            ("yoy_change",     "yoy_change_pct"),
            ("yoy_change_pct", "yoy_change_pct"),
        ):
            if row.get(fld) is not None:
                entry[label] = row[fld]

        # 이동평균
        if row.get("ma_30") is not None:
            entry["ma_30"] = row["ma_30"]
        if row.get("ma_90") is not None:
            entry["ma_90"] = row["ma_90"]

        # 추세
        if row.get("trend_label"):
            entry["trend_label"] = row["trend_label"]
        elif row.get("trend_3m"):
            entry["trend_3m"] = row["trend_3m"]

        # 통계
        if row.get("historical_pct") is not None:
            entry["historical_percentile"] = row["historical_pct"]
        elif row.get("historical_percentile") is not None:
            entry["historical_percentile"] = row["historical_percentile"]
        if row.get("volatility_3m") is not None:
            entry["volatility_3m"] = row["volatility_3m"]
        for field in ("sample_count_3m", "total_samples"):
            if row.get(field) is not None:
                entry[field] = row[field]

        output["metrics"][name] = entry

    return json.dumps(output, ensure_ascii=False, indent=2)


# ── Step 2: rag_docs → 뉴스·리포트 컨텍스트 ─────────────────────────

def _format_docs_section(docs: list[dict], label: str) -> str:
    """뉴스 또는 리포트 목록을 레이블을 붙여 포맷합니다."""
    if not docs:
        return f"[{label}] 관련 문서 없음"

    parts: list[str] = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.get("metadata", {})
        content = doc.get("content", "").strip()
        score   = doc.get("relevance_score", 0.0)
        header  = (
            f"[{label} {i}] 출처: {meta.get('source_name', '알 수 없음')} | "
            f"날짜: {meta.get('published_date', '날짜 미상')} | "
            f"관련성: {score:.2f}"
        )
        parts.append(f"{header}\n{content}")
    return "\n\n" + ("-" * 60 + "\n\n").join(parts)


# ── Step 3: LLM 호출 ─────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke_llm(chain_input: dict[str, str]) -> Any:
    return build_macro_only_chain(_get_llm()).invoke(chain_input)


# ── Step 4: JSON 파싱 ─────────────────────────────────────────────────

def _safe_parse_json(raw: Any, question: str = "") -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw

    raw_str = str(raw) if not isinstance(raw, str) else raw
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        pass

    for pattern in [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*(\{[\s\S]*?\})\s*```",
        r"(\{[\s\S]*\})",
    ]:
        m = re.search(pattern, raw_str)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                continue

    logger.error("거시 분석 JSON 파싱 실패. 미리보기:\n%s", raw_str[:400])
    return {
        "schema_version":        "1.0",
        "is_my_domain":          False,
        "domain_rejection_reason": "LLM이 유효한 JSON을 출력하지 않았습니다.",
        "query":                 question,
        "analysis_timestamp":    datetime.now(timezone.utc).isoformat(),
        "reasoning_flow":        None,
        "macro_analysis":        None,
        "overall_risk_level":    None,
        "key_signals":           [],
        "summary":               None,
        "confidence_score":      0.0,
        "data_sources":          [],
        "_parse_error":          True,
        "_raw_output_preview":   raw_str[:500],
    }


def check_metric_evidence(result: dict, metrics: dict, scopes: dict) -> dict:
    """Check copied metric/field/value pairs; free-text meaning still needs review."""
    issues = []
    analysis = result.get("macro_analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}
    points = analysis.get("key_data_points")
    if not isinstance(points, dict):
        points = {}
        issues.append("missing_metric_evidence")
    if set(points) != set(metrics):
        issues.append("metric_coverage_mismatch")
    for metric_id, fields in points.items():
        expected = metrics.get(metric_id)
        if expected is None:
            issues.append(f"{metric_id}: unknown_metric")
            continue
        if not isinstance(fields, dict):
            issues.append(f"{metric_id}: invalid_fields")
            continue
        if not {"latest_value", "unit", "as_of"} <= set(fields):
            issues.append(f"{metric_id}: missing_value_unit_date")
        for field, value in fields.items():
            if (field not in expected or isinstance(value, bool) or value != expected[field]):
                issues.append(f"{metric_id}.{field}: value_mismatch")
    if any(scope["state"] in {"partial", "omitted"} for scope in scopes.values()):
        issues.append("metric_input_truncated")
    sector_ids = {"ecos.electronics_shipment_index", "ecos.electronics_inventory_index", "market.sox_close"}
    observed_sector = [key for key in sector_ids if metrics.get(key, {}).get("latest_value") is not None]
    if analysis.get("phase") != "UNKNOWN" and len(observed_sector) < 2:
        issues.append("phase_without_sector_observations")
    if analysis.get("phase") not in {"UNKNOWN", "BOOM", "SLOWDOWN", "RECESSION", "RECOVERY"}:
        issues.append("invalid_phase")
    return {"status": "needs_review" if issues else "passed", "issues": issues,
            "checked_metric_ids": list(metrics), "semantic_accuracy": "not_verified"}


# ── 메인 노드 함수 ────────────────────────────────────────────────────

def macro_analyst_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph 노드: 거시경제 정량 분석.

    data_retrieval_node가 채운 ts_features와 rag_docs를 읽어
    LLM으로 거시경제 분석을 수행합니다. DB 직접 조회 없음.

    반환 필드:
        macro_analysis_raw : 거시경제 LLM 분석 결과 (원본)
        warnings           : 이 노드의 신규 경고 (리듀서가 누적)
    """
    question:    str             = state.get("question", "").strip()
    ts_features: dict[str, dict] = state.get("ts_features", {})
    rag_docs:    list[dict]      = state.get("rag_docs", [])
    new_warnings: list[str]      = []

    logger.info(
        "=== macro_analyst_node 시작 (ts_features=%d개, rag_docs=%d건) ===",
        len(ts_features), len(rag_docs),
    )

    if not question:
        return {"macro_analysis_raw": {}, "warnings": ["macro_analyst: question 필드 비어있음"]}

    # ── Step 1: ts_features → ECOS / FRED 섹션 포맷 ──────────────────
    logger.info("[Step 1/3] TimescaleDB 컨텍스트 포맷 중...")
    ecos_json_str = _format_ts_section(ts_features, prefix="ecos")
    fred_json_str = _format_ts_section(ts_features, prefix="fred")
    try:
        ecos_data = json.loads(ecos_json_str)
        fred_data = json.loads(fred_json_str)
    except json.JSONDecodeError:
        ecos_data, fred_data = {}, {}

    # ── Step 2: rag_docs → 뉴스·리포트 컨텍스트 ─────────────────────
    logger.info("[Step 2/3] RAG 컨텍스트 포맷 중 (docs=%d건)...", len(rag_docs))
    news_docs   = [d for d in rag_docs if d.get("metadata", {}).get("source_type") == "news"]
    report_docs = [d for d in rag_docs if d.get("metadata", {}).get("source_type") == "report"]
    news_context   = _format_docs_section(news_docs,   label="뉴스")
    report_context = _format_docs_section(report_docs, label="리포트")

    chain_input = build_chain_input(
        query          = question,
        ecos_result    = ecos_data,
        fred_result    = fred_data,
        news_context   = news_context,
        report_context = report_context,
        current_date   = date.today().isoformat(),
    )

    # ── Step 3: LLM 호출 ─────────────────────────────────────────────
    logger.info("[Step 3/3] LLM(gpt-4o) 거시 분석 호출 중...")
    macro_raw: dict[str, Any]
    try:
        raw = _invoke_llm(chain_input)
        macro_raw = _safe_parse_json(raw, question)
    except Exception as exc:
        msg = f"거시 분석 LLM 호출 실패 (3회 소진): {exc}"
        logger.error(msg, exc_info=True)
        macro_raw = _safe_parse_json("", question)
        macro_raw["domain_rejection_reason"] = msg
        new_warnings.append(msg)

    scopes = {key: json.loads(chain_input[key])["source_scope"] for key in ("ecos_data", "fred_data")}
    macro_raw["_grounding_check"] = check_metric_evidence(
        macro_raw, {**ecos_data.get("metrics", {}), **fred_data.get("metrics", {})}, scopes)
    if macro_raw["_grounding_check"]["status"] == "needs_review":
        new_warnings.append("macro_analyst: 지표-값 대조 불일치; 모델 해석 검토 필요")

    logger.info(
        "=== macro_analyst_node 완료 — risk=%s, confidence=%.2f ===",
        macro_raw.get("overall_risk_level", "N/A"),
        macro_raw.get("confidence_score", 0.0),
    )

    return {
        "macro_analysis_raw": macro_raw,
        "macro_input_scope": scopes,
        "warnings":           new_warnings,   # 리듀서가 geo_analyst 경고와 합산
    }
