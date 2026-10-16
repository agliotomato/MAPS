"""
Geo Analyst Node — 지정학 리스크 분석 전담

실행 흐름:
    Step 1. state.rag_docs → 지정학 체인 입력 구성 (DB 재조회 없음)
    Step 2. LLM (gpt-4o)  → build_geo_chain() 호출
    Step 3. Parse          → JSON 파싱 + 폴백
    결과를 state.geo_analysis_raw에 저장

설계 원칙:
    - 최종 결과 병합은 synthesizer_node가 전담 → 이 노드는 순수 지정학 분석
    - macro_analyst_node와 병렬 실행 가능 (state 필드 충돌 없음)
    - warnings: 이 노드의 신규 경고만 반환 (Annotated 리듀서가 누적)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from macro_agent.prompts.geo_prompt import build_geo_chain, build_geo_chain_input
from macro_agent.state import AgentState

logger = logging.getLogger(__name__)


# ── LLM 싱글톤 ────────────────────────────────────────────────────────

def _get_llm():
    from functools import lru_cache

    from langchain_openai import ChatOpenAI

    @lru_cache(maxsize=1)
    def _cached():
        return ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=3000)
    return _cached()


# ── RAG 문서 → 지정학 컨텍스트 포맷 ─────────────────────────────────

def _format_docs_for_geo(docs: list[dict], label: str) -> str:
    """뉴스 또는 리포트 목록을 레이블을 붙여 지정학 프롬프트용 텍스트로 변환합니다."""
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


# ── LLM 호출 ─────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _invoke_geo_llm(chain_input: dict[str, str]) -> Any:
    return build_geo_chain(_get_llm()).invoke(chain_input)


def _safe_parse(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    raw_str = str(raw) if not isinstance(raw, str) else raw
    try:
        return json.loads(raw_str)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\{[\s\S]*\})", raw_str)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    logger.warning("지정학 분석 JSON 파싱 실패. 미리보기:\n%s", raw_str[:300])
    return {}


# ── 메인 노드 함수 ────────────────────────────────────────────────────

def geo_analyst_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph 노드: 지정학 리스크 분석.

    data_retrieval_node가 채운 rag_docs를 읽어 지정학 분석을 수행합니다.
    최종 결과 병합은 synthesizer_node가 담당합니다.

    반환 필드:
        geo_analysis_raw : 지정학 LLM 분석 결과 (원본)
        warnings         : 이 노드의 신규 경고 (리듀서가 누적)
    """
    rag_docs:    list[dict]  = state.get("rag_docs", [])
    new_warnings: list[str] = []

    logger.info("=== geo_analyst_node 시작 (rag_docs=%d건) ===", len(rag_docs))

    # ── Step 1: rag_docs → 뉴스/리포트 분리 후 체인 입력 구성 ─────────
    news_docs   = [d for d in rag_docs if d.get("metadata", {}).get("source_type") == "news"]
    report_docs = [d for d in rag_docs if d.get("metadata", {}).get("source_type") == "report"]
    logger.info("geo_analyst: 뉴스=%d건, KCIF 리포트=%d건", len(news_docs), len(report_docs))

    news_context   = _format_docs_for_geo(news_docs,   label="뉴스")
    report_context = _format_docs_for_geo(report_docs, label="KCIF 리포트")

    chain_input = build_geo_chain_input(
        news_context   = news_context,
        report_context = report_context,
        focus_regions  = "미-중, 한-미, 대만해협, 러시아, 중동",
    )

    # ── Step 2: LLM 지정학 분석 ──────────────────────────────────────
    logger.info("[Step 1/1] LLM(gpt-4o) 지정학 분석 호출 중...")
    geo_raw: dict[str, Any] = {}
    try:
        raw    = _invoke_geo_llm(chain_input)
        geo_raw = _safe_parse(raw)
        logger.info(
            "지정학 분석 완료 — overall_geo_risk=%s, events=%d건",
            geo_raw.get("overall_geo_risk", "N/A"),
            len(geo_raw.get("risk_events", [])),
        )
    except Exception as exc:
        msg = f"지정학 분석 LLM 실패 (3회 소진): {exc}"
        logger.warning(msg, exc_info=True)
        new_warnings.append(msg)

    logger.info("=== geo_analyst_node 완료 ===")

    return {
        "geo_analysis_raw": geo_raw,
        "warnings":         new_warnings,   # 리듀서가 macro_analyst 경고와 합산
    }
