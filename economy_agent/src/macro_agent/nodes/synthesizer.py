"""
Synthesizer Node — 거시경제·지정학 결과 병합 및 최종 출력 완성

실행 흐름:
    Step 1. state.macro_analysis_raw + state.geo_analysis_raw 읽기
    Step 2. 도메인 外 판정이면 macro_analysis_raw를 그대로 analysis_result로 사용
    Step 3. 두 분석 결과를 최종 출력 스키마(JSON)로 병합
    결과를 state.analysis_result에 저장, run_at 기록

병합 규칙:
    - overall_risk_level : max(거시 위험, 지정학 위험)
    - key_signals        : 거시 key_signals + 지정학 watchlist 상위 3개 ([지정학] 접두어)
    - geo_analysis       : geo_prompt 출력 스키마 → 메인 스키마 geo_analysis 포맷 변환
    - confidence_score   : 두 분석 평균 (한 쪽만 있으면 해당 값)
    - data_sources       : 합집합
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from macro_agent.state import AgentState

logger = logging.getLogger(__name__)

_RISK_ORDER: dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


# ── 병합 헬퍼 ─────────────────────────────────────────────────────────

def _max_risk(level_a: str | None, level_b: str | None) -> str | None:
    """두 위험 레벨 중 더 높은 쪽을 반환합니다."""
    known = [level for level in (level_a, level_b) if level in _RISK_ORDER]
    return max(known, key=_RISK_ORDER.get) if known else None


def _geo_events_to_schema(geo_raw: dict) -> dict | None:
    """
    geo_prompt 출력 스키마 → 메인 스키마의 geo_analysis 블록으로 변환합니다.
    risk_events가 없고 geo_risk_summary도 없으면 None 반환.
    """
    events = geo_raw.get("risk_events", [])
    if not events and not geo_raw.get("geo_risk_summary"):
        return None

    return {
        "geo_risk_summary": geo_raw.get("geo_risk_summary", ""),
        "overall_geo_risk": geo_raw.get("overall_geo_risk"),
        "risk_items": [
            {
                "region":                   e.get("region", ""),
                "event":                    e.get("event_description", ""),
                "category":                 e.get("category", ""),
                "regulation_reference":     e.get("regulation_reference"),
                "risk_level":               e.get("risk_level", "MEDIUM"),
                "affected_segments":        e.get("affected_semiconductor_segments", []),
                "impact_horizon":           e.get("impact_horizon", "MID"),
                "korea_exposure_level":     e.get("korea_exposure_level", "MEDIUM"),
                "reasoning_flow":           e.get("reasoning_flow"),
            }
            for e in events
        ],
        "watchlist": geo_raw.get("watchlist", []),
    }


def _merge_results(
    macro: dict[str, Any],
    geo:   dict[str, Any],
    question: str,
) -> dict[str, Any]:
    """
    거시 분석 결과(macro_analysis_raw)와 지정학 분석 결과(geo_analysis_raw)를
    최종 출력 스키마로 병합합니다.
    """
    # overall_risk_level: 두 결과 중 더 높은 레벨
    macro_risk = macro.get("overall_risk_level")
    # No retrieved event is not evidence of low geopolitical risk.
    geo_risk   = geo.get("overall_geo_risk") if geo.get("risk_events") else None
    final_risk = _max_risk(macro_risk, geo_risk)

    # key_signals: 거시 신호 + 지정학 watchlist 상위 3개
    macro_signals: list[str] = macro.get("key_signals", [])
    geo_watchlist: list[str] = [
        f"[지정학] {s}" for s in geo.get("watchlist", [])[:3]
    ]

    # data_sources: 합집합
    sources = list({
        *macro.get("data_sources", []),
        *geo.get("data_sources",  []),
    })

    # geo_analysis 블록 변환
    geo_analysis = _geo_events_to_schema(geo) or macro.get("geo_analysis")

    # confidence: 두 분석 평균 (한 쪽만 있으면 해당 값 사용)
    macro_conf = float(macro.get("confidence_score") or 0.0)
    geo_conf   = float(geo.get("confidence_score")   or 0.0)
    if macro_conf > 0 and geo_conf > 0:
        final_conf = round((macro_conf + geo_conf) / 2, 2)
    else:
        final_conf = macro_conf or geo_conf

    result = {
        "schema_version":          "1.0",
        "is_my_domain":            macro.get("is_my_domain", True),
        "domain_rejection_reason": macro.get("domain_rejection_reason"),
        "query":                   question,
        "analysis_timestamp":      datetime.now(timezone.utc).isoformat(),
        "reasoning_flow":          macro.get("reasoning_flow"),
        "macro_analysis":          macro.get("macro_analysis"),
        "geo_analysis":            geo_analysis,
        "overall_risk_level":      final_risk,
        "key_signals":             macro_signals + geo_watchlist,
        "summary":                 macro.get("summary"),
        "confidence_score":        final_conf if final_risk is not None else 0.0,
        "data_sources":            sources,
        # 내부 진단
        "_parse_error":            macro.get("_parse_error", False),
        "_grounding_check":        macro.get("_grounding_check"),
    }
    if (macro.get("_grounding_check") or {}).get("status") == "needs_review":
        # Preserve the model output in state.macro_analysis_raw; do not endorse it.
        result.update(overall_risk_level=None, confidence_score=0.0, key_signals=[],
                      reasoning_flow=None, macro_analysis=None,
                      summary="판단 유보: 경제 지표와 모델 인용 값이 일치하지 않아 해석 검토가 필요합니다.")
    return result


# ── 메인 노드 함수 ────────────────────────────────────────────────────

def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph 노드: 거시경제·지정학 분석 결과 병합.

    geo_analyst_node와 macro_analyst_node의 병렬 실행이 완료된 후
    두 결과를 최종 출력 스키마로 합산합니다.

    반환 필드:
        analysis_result : 최종 분석 결과 dict
        run_at          : 그래프 완료 시각 (ISO8601 UTC)
        warnings        : 이 노드의 신규 경고 (리듀서가 누적)
    """
    question:         str            = state.get("question", "").strip()
    macro_raw:        dict[str, Any] = state.get("macro_analysis_raw", {})
    geo_raw:          dict[str, Any] = state.get("geo_analysis_raw",   {})
    new_warnings:     list[str]      = []

    logger.info(
        "=== synthesizer_node 시작 (macro_risk=%s, geo_risk=%s) ===",
        macro_raw.get("overall_risk_level", "N/A"),
        geo_raw.get("overall_geo_risk",     "N/A"),
    )

    # ── 도메인 外 판정이면 병합 스킵 ─────────────────────────────────
    if not macro_raw.get("is_my_domain", True):
        logger.info("거시 분석이 도메인 外 판정 → 지정학 병합 스킵")
        result = {
            **macro_raw,
            "query":              question,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return {
            "analysis_result": result,
            "run_at":          datetime.now(timezone.utc).isoformat(),
            "warnings":        new_warnings,
        }

    # ── geo_analysis_raw 비어있으면 경고 기록 후 빈 dict로 진행 ──────
    if not geo_raw:
        msg = "geo_analysis_raw 비어있음 — 지정학 분석 없이 거시 결과만 최종 출력"
        logger.warning(msg)
        new_warnings.append(msg)

    # ── 결과 병합 ─────────────────────────────────────────────────────
    analysis_result = _merge_results(macro_raw, geo_raw, question)
    run_at = datetime.now(timezone.utc).isoformat()

    logger.info(
        "=== synthesizer_node 완료 — final_risk=%s, confidence=%.2f ===",
        analysis_result.get("overall_risk_level", "N/A"),
        analysis_result.get("confidence_score", 0.0),
    )

    return {
        "analysis_result": analysis_result,
        "run_at":          run_at,
        "warnings":        new_warnings,
    }
