"""
LangGraph 공유 상태 스키마 (AgentState)

멀티 에이전트 시스템에서 모든 노드가 이 상태를 읽고 씁니다.
필드는 그래프 흐름에 따라 순차·병렬로 채워집니다.

그래프 흐름:
    START
      │
    [domain_filter]       ← is_domain 판정
      │ (is_domain=True)
    [data_retrieval]      ← rag_docs, ts_features 채움
      │
      ├──→ [geo_analyst]  ← geo_analysis_raw 채움  ┐
      │                                             │ 병렬 실행
      └──→ [macro_analyst] ← macro_analysis_raw 채움┘
                                    │
                           [synthesizer]            ← analysis_result 완성
                                    │
                                   END

병렬 실행 안전성:
    warnings 필드는 Annotated[list[str], operator.add] 리듀서로 선언하여
    병렬로 실행되는 geo_analyst / macro_analyst가 각자의 경고를 append해도
    LangGraph가 자동으로 합산합니다.

    병렬 노드 작성 패턴:
        new_warnings: list[str] = []          # 자기 노드의 경고만
        ...
        return {..., "warnings": new_warnings}  # 리듀서가 누적
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):

    # ── 입력 (오케스트레이터 또는 사용자가 주입) ──────────────────────
    question: str       # 분석 요청 자연어 질문 (필수)
    sender: str         # 질문 발신자 식별자 (예: "orchestrator", "user")

    # ── domain_filter 산출물 ──────────────────────────────────────────
    is_domain: bool     # True → data_retrieval 진입 / False → END

    # ── data_retrieval 산출물 ─────────────────────────────────────────
    rag_docs: list[dict]            # Vector DB RAG 문서
    data_status: str               # available / partial / empty / error (수집 결과)
    #   구조: [{"content": str, "metadata": dict, "relevance_score": float}, ...]

    ts_features: dict[str, dict]    # TimescaleDB 시계열 피처 (enriched snapshot 포함)
    #   구조: {"ecos.usd_krw": {"current_value": ..., "ma_30": ..., "yoy_change": ...}, ...}
    #   출처: get_macro_ts_features() 날짜 구간별 집계

    # ── 컨텍스트 요약 (agent.py _agent_meta용) ───────────────────────
    rag_context: str    # Vector DB 조회 결과 요약 문자열
    ts_context: str     # TimescaleDB 조회 결과 요약 문자열

    # ── 분석 노드 산출물 (병렬 실행) ──────────────────────────────────
    geo_analysis_raw: dict[str, Any]    # geo_analyst_node 원본 LLM 출력
    #   구조: geo_prompt 스키마 (overall_geo_risk, risk_events, watchlist ...)

    macro_analysis_raw: dict[str, Any]  # macro_analyst_node 원본 LLM 출력
    macro_input_scope: dict[str, Any]   # ECOS/FRED 직렬화 입력의 실제 전달/생략 문자 수
    #   구조: macro_prompt 스키마 (overall_risk_level, macro_analysis, key_signals ...)

    # ── synthesizer 산출물 (최종 출력) ────────────────────────────────
    analysis_result: dict[str, Any]     # 두 분석 병합 + 최종 스키마
    run_at: str                         # 그래프 완료 시각 (ISO8601 UTC)

    # ── 오류 추적 ─────────────────────────────────────────────────────
    error: str | None       # 치명적 오류 메시지 (None이면 정상)

    # 병렬 노드가 각자의 경고만 반환하면 리듀서가 자동 합산합니다.
    warnings: Annotated[list[str], operator.add]
