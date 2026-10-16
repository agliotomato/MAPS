"""
Data Retrieval Node — RAG + TimescaleDB 데이터 수집 전담

실행 흐름:
    Step 1. Vector DB (ChromaDB)   → 뉴스 + 리포트 RAG 문서 조회
    Step 2. TimescaleDB            → get_macro_ts_features()
                                     결과를 지표명 단위로 저장
    State 저장:
        rag_docs      : 뉴스 + 리포트 RAG 문서 리스트
        ts_features   : 지표별 시계열 피처 dict (MA·MoM·YoY·변동성·백분위)
        rag_context   : Vector DB 조회 요약 문자열
        ts_context    : TimescaleDB 조회 요약 문자열

후속 노드(geo_analyst, macro_analyst)는 이 노드가 채운 상태를 읽기만 합니다.
DB 조회가 완전히 분리되어 두 분석 노드의 LLM 병렬 실행이 가능합니다.
"""

from __future__ import annotations

import logging
from typing import Any

from macro_agent.db.timescale.repository import get_macro_ts_features
from macro_agent.db.vector.repository import query_vector_db
from macro_agent.state import AgentState

logger = logging.getLogger(__name__)

# Vector DB 조회 건수 상수
_N_NEWS    = 20  # 뉴스 청크 최대 건수
_N_REPORTS = 10  # 리포트 청크 최대 건수

# 뉴스 기본 날짜 필터: 최근 90일 이내 발행된 기사 우선
# None이면 날짜 필터 없음 (발행일 미기록 문서도 포함됨)
_DEFAULT_NEWS_DAYS = 90


def _news_cutoff_date(days: int) -> str:
    """오늘로부터 days일 전 날짜를 "YYYY-MM-DD" 형식으로 반환합니다."""
    from datetime import date, timedelta
    return (date.today() - timedelta(days=days)).isoformat()


# ── Step 1: Vector DB RAG 조회 ────────────────────────────────────────

def _fetch_rag_docs(
    question: str,
    news_published_after: str | None = None,
) -> tuple[list[dict], list[str]]:
    """
    뉴스와 리포트를 각각 조회해 합산 반환합니다.

    Args:
        question:             검색 질의
        news_published_after: 뉴스 날짜 하한 ("YYYY-MM-DD"). None이면 필터 없음.

    Returns:
        (rag_docs, new_warnings)
    """
    docs: list[dict] = []
    new_warnings: list[str] = []

    # 뉴스 조회 (날짜 필터 적용)
    try:
        news = query_vector_db(
            query=question,
            n_results=_N_NEWS,
            filter_source_type="news",
            published_after=news_published_after,
        )
        docs.extend(news)
        logger.info(
            "Vector DB 뉴스 조회 완료: %d건 (after=%s)",
            len(news), news_published_after or "none",
        )
    except Exception as exc:
        msg = f"Vector DB 뉴스 조회 실패: {exc}"
        logger.warning(msg)
        new_warnings.append(msg)

    # 리포트 조회 (현재 날짜 하한 없음; 반환된 발간일을 함께 전달)
    try:
        reports = query_vector_db(
            query=question,
            n_results=_N_REPORTS,
            filter_source_type="report",
        )
        docs.extend(reports)
        logger.info("Vector DB 리포트 조회 완료: %d건", len(reports))
    except Exception as exc:
        msg = f"Vector DB 리포트 조회 실패: {exc}"
        logger.warning(msg)
        new_warnings.append(msg)

    return docs, new_warnings


# ── Step 2: TimescaleDB 시계열 피처 조회 ────────────────────────────

def _fetch_ts_features() -> tuple[dict[str, dict], list[str]]:
    """Read the date-window statistics once and preserve database failures."""
    try:
        return get_macro_ts_features(months_back=36), []
    except Exception as exc:
        message = f"TimescaleDB 지표 조회 실패: {type(exc).__name__}"
        logger.warning(message)
        return {}, [message]


# ── 메인 노드 함수 ────────────────────────────────────────────────────

def data_retrieval_node(state: AgentState) -> dict[str, Any]:
    """
    LangGraph 노드: RAG + TimescaleDB 데이터 수집.

    후속 병렬 노드(geo_analyst, macro_analyst)가 DB를 직접 조회하지 않도록
    여기서 모든 데이터를 미리 수집해 state에 저장합니다.

    반환 필드:
        rag_docs      : Vector DB 조회 문서 (뉴스 + 리포트 혼합)
        ts_features   : TimescaleDB 시계열 피처 dict
        rag_context   : 조회 결과 요약 (agent._agent_meta에 노출)
        ts_context    : 조회 결과 요약 (agent._agent_meta에 노출)
        warnings      : 이 노드에서 발생한 경고 (리듀서가 누적)
    """
    question: str = state.get("question", "").strip()
    logger.info("=== data_retrieval_node 시작 ===")

    # ── Step 1: Vector DB ─────────────────────────────────────────────
    # 뉴스는 최근 90일 이내 발행된 기사만 우선 검색
    # (발행일 미기록 문서는 날짜 필터와 무관하게 포함됨)
    news_cutoff = _news_cutoff_date(_DEFAULT_NEWS_DAYS)
    logger.info(
        "[Step 1/2] Vector DB RAG 조회 중 (뉴스 %d건 after=%s + 리포트 %d건)...",
        _N_NEWS, news_cutoff, _N_REPORTS,
    )
    rag_docs, rag_warnings = _fetch_rag_docs(question, news_published_after=news_cutoff)

    # ── Step 2: TimescaleDB ───────────────────────────────────────────
    logger.info("[Step 2/2] TimescaleDB 시계열 피처 조회 중...")
    ts_features, ts_warnings = _fetch_ts_features()

    # ── 요약 문자열 ───────────────────────────────────────────────────
    news_cnt   = sum(1 for d in rag_docs if d.get("metadata", {}).get("source_type") == "news")
    report_cnt = len(rag_docs) - news_cnt
    rag_context = f"뉴스 청크 {news_cnt}건 + 리포트 청크 {report_cnt}건"
    ts_context  = f"TimescaleDB {len(ts_features)}개 지표 (최대 36개월 조회; 지표별 관측일·표본 수 확인)"

    new_warnings = rag_warnings + ts_warnings
    if new_warnings:
        data_status = "partial" if rag_docs or ts_features else "error"
    elif not rag_docs and not ts_features:
        data_status = "empty"
    else:
        data_status = "available" if rag_docs and ts_features else "partial"

    logger.info(
        "=== data_retrieval_node 완료 — RAG: %s, TS: %s ===",
        rag_context, ts_context,
    )

    return {
        "rag_docs":    rag_docs,
        "ts_features": ts_features,
        "data_status": data_status,
        "rag_context": rag_context,
        "ts_context":  ts_context,
        "warnings":    new_warnings,   # 리듀서가 기존 warnings에 append
    }
