"""
LangGraph StateGraph 정의 — 멀티 에이전트 팬아웃/팬인 구조

그래프 흐름:
    START
      │
    [domain_filter]          ── 도메인 外 ──→ END  (거부 스키마 즉시 반환)
      │ 도메인 內
      │
    [data_retrieval]         ← RAG (ChromaDB) + TimescaleDB 시계열 피처
      │                        rag_docs, ts_features → state
      │
      ├──→ [geo_analyst]     ← rag_docs만 활용, 지정학 LLM 분석
      │                        geo_analysis_raw → state
      │
      └──→ [macro_analyst]   ← ts_features + rag_docs 활용, 거시경제 LLM 분석
                               macro_analysis_raw → state
               │
               │  (두 노드 완료 후 자동 합류 — LangGraph 내부 barrier)
               │
           [synthesizer]     ← geo_analysis_raw + macro_analysis_raw 병합
                               analysis_result → state
               │
              END

병렬 실행 보장:
    data_retrieval → geo_analyst / macro_analyst 팬아웃 후,
    두 노드가 모두 완료되면 synthesizer로 자동 합류합니다.

    AgentState.warnings 필드는 Annotated[list[str], operator.add] 리듀서로
    선언되어 있어 병렬 노드의 경고가 유실 없이 누적됩니다.
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from macro_agent.nodes.data_retrieval import data_retrieval_node
from macro_agent.nodes.domain_filter import domain_filter_node
from macro_agent.nodes.geo_analyst import geo_analyst_node
from macro_agent.nodes.macro_analyst import macro_analyst_node
from macro_agent.nodes.synthesizer import synthesizer_node
from macro_agent.state import AgentState


# ── 라우팅 함수 ───────────────────────────────────────────────────────

def _route_after_filter(state: AgentState) -> str:
    """
    도메인 필터 결과에 따라 데이터 수집 노드 또는 END로 라우팅합니다.

    is_domain=True  → "data_retrieval"
    is_domain=False → END  (domain_filter_node가 analysis_result에 거부 스키마 세팅)
    """
    return "data_retrieval" if state.get("is_domain", True) else END


# ── 그래프 빌더 ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def build_graph() -> StateGraph:
    """
    컴파일된 LangGraph StateGraph를 반환합니다.
    lru_cache로 프로세스 내 1회만 빌드됩니다.

    노드 실행 순서:
        1. domain_filter   (직렬)
        2. data_retrieval  (직렬 — DB 조회)
        3. geo_analyst     (병렬 ┐
           macro_analyst   (병렬 ┘ — LLM 호출)
        4. synthesizer     (직렬 — 결과 병합)
    """
    builder = StateGraph(AgentState)

    # ── 노드 등록 ────────────────────────────────────────────────────
    builder.add_node("domain_filter",  domain_filter_node)
    builder.add_node("data_retrieval", data_retrieval_node)
    builder.add_node("geo_analyst",    geo_analyst_node)
    builder.add_node("macro_analyst",  macro_analyst_node)
    builder.add_node("synthesizer",    synthesizer_node)

    # ── 엣지 정의 ────────────────────────────────────────────────────

    # START → domain_filter (직렬)
    builder.add_edge(START, "domain_filter")

    # domain_filter → data_retrieval (도메인 內) | END (도메인 外)
    builder.add_conditional_edges("domain_filter", _route_after_filter)

    # data_retrieval → 팬아웃 (geo_analyst ‖ macro_analyst, 병렬)
    builder.add_edge("data_retrieval", "geo_analyst")
    builder.add_edge("data_retrieval", "macro_analyst")

    # 팬인 → synthesizer (두 노드 완료 후 자동 합류)
    builder.add_edge("geo_analyst",   "synthesizer")
    builder.add_edge("macro_analyst", "synthesizer")

    # synthesizer → END
    builder.add_edge("synthesizer", END)

    return builder.compile()
