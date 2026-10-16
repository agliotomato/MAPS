"""
MacroAnalysisAgent — 멀티 에이전트 시스템용 퍼블릭 인터페이스

오케스트레이터나 타 에이전트는 이 클래스만 알면 됩니다.
내부 DB, LLM, 프롬프트 구조는 완전히 캡슐화되어 있습니다.

Usage:
    agent = MacroAnalysisAgent()

    # 동기 호출
    result: dict = agent.invoke("미국 금리 인상이 반도체 수출에 미치는 영향은?")

    # 비동기 호출 (async 환경)
    result: dict = await agent.ainvoke("미중 수출통제 리스크를 분석해 줘.")

    # 도메인 외 질문 감지
    if not result.get("is_my_domain"):
        print(result["domain_rejection_reason"])
"""

from __future__ import annotations

import logging
from typing import Any

from macro_agent.db.timescale.client import initialize_schema
from macro_agent.db.vector.repository import ChromaReportStore
from macro_agent.graph import build_graph
from macro_agent.state import AgentState
from macro_agent.tools.news_tool import set_report_store

logger = logging.getLogger(__name__)


class MacroAnalysisAgent:
    """
    반도체 도메인 거시경제·지정학 리스크 분석 에이전트.

    초기화 옵션:
        auto_init_db: True이면 애플리케이션 시작 시 TimescaleDB 스키마를 자동 생성합니다.
                      False이면 별도로 initialize_schema()를 호출해야 합니다.

    반환 보장:
        invoke() / ainvoke()는 항상 dict를 반환합니다.
        LLM 오류나 JSON 파싱 실패 시에도 표준 에러 스키마 dict를 반환합니다.
        예외를 직접 전파하지 않습니다.
    """

    def __init__(self, auto_init_db: bool = False) -> None:
        self._graph = build_graph()

        # Vector DB를 ChromaDB로 교체 (InMemoryReportStore 대신)
        set_report_store(ChromaReportStore())
        logger.info("MacroAnalysisAgent: ChromaReportStore 주입 완료")

        if auto_init_db:
            try:
                initialize_schema()
            except Exception as exc:
                logger.warning(
                    "TimescaleDB 스키마 초기화 실패 (에이전트는 계속 동작): %s", exc
                )

    def invoke(
        self,
        question: str,
        sender: str = "user",
    ) -> dict[str, Any]:
        """
        질문을 동기적으로 분석하여 JSON dict를 반환합니다.

        Args:
            question: 분석 요청 자연어 질문
            sender:   질문 발신자 식별자 (로깅·추적용)

        Returns:
            분석 결과 dict (schema_version, is_my_domain, reasoning_flow 등 포함)
            파싱 실패 시: {"_parse_error": True, ...} 포함 에러 스키마
        """
        initial_state: AgentState = {
            "question": question,
            "sender": sender,
            "warnings": [],
        }

        try:
            final_state: AgentState = self._graph.invoke(initial_state)
            return self._extract_result(final_state)
        except Exception as exc:
            logger.error("MacroAnalysisAgent.invoke 예외: %s", exc, exc_info=True)
            return _fatal_error_schema(question, str(exc))

    async def ainvoke(
        self,
        question: str,
        sender: str = "user",
    ) -> dict[str, Any]:
        """
        질문을 비동기적으로 분석하여 JSON dict를 반환합니다.
        LangGraph ainvoke()를 사용하므로 async 환경(FastAPI, async 오케스트레이터)에 적합합니다.
        """
        initial_state: AgentState = {
            "question": question,
            "sender": sender,
            "warnings": [],
        }

        try:
            final_state: AgentState = await self._graph.ainvoke(initial_state)
            return self._extract_result(final_state)
        except Exception as exc:
            logger.error("MacroAnalysisAgent.ainvoke 예외: %s", exc, exc_info=True)
            return _fatal_error_schema(question, str(exc))

    def _extract_result(self, final_state: AgentState) -> dict[str, Any]:
        """최종 상태에서 분석 결과와 메타데이터를 추출합니다."""
        result = dict(final_state.get("analysis_result") or {})

        # 에이전트 실행 메타데이터를 결과에 병합
        result["_agent_meta"] = {
            "is_domain": final_state.get("is_domain"),
            "data_status": final_state.get("data_status", "not_requested"
                                            if final_state.get("is_domain") is False else "unknown"),
            "document_count": len(final_state.get("rag_docs") or []),
            "metric_count": len(final_state.get("ts_features") or {}),
            "run_at": final_state.get("run_at"),
            "rag_context": final_state.get("rag_context"),
            "ts_context": final_state.get("ts_context"),
            "sender": final_state.get("sender"),
            "warnings": final_state.get("warnings", []),
            "fatal_error": final_state.get("error"),
            "analysis_inputs": final_state.get("macro_input_scope", {}),
        }
        result["evidence_metadata"] = {
            "documents": [dict(doc.get("metadata") or {}) | {"content": doc.get("content")}
                          for doc in final_state.get("rag_docs") or []],
            "metrics": {name: {"source": row.get("source"),
                                "as_of": str(row.get("as_of") or row.get("time") or "") or None,
                                "values": dict(row)}
                        for name, row in (final_state.get("ts_features") or {}).items()},
        }
        if final_state.get("data_status") in {"empty", "error"}:
            result.update(overall_risk_level=None, confidence_score=0.0, key_signals=[],
                          summary="판단 유보: 수집된 경제 자료가 없습니다. 조회 상태와 경고를 확인하세요.",
                          reasoning_flow=None)
        return result


# ── 에러 스키마 팩토리 ─────────────────────────────────────────────────

def _fatal_error_schema(question: str, error_message: str) -> dict[str, Any]:
    """치명적 예외 발생 시 반환할 표준 에러 스키마"""
    return {
        "schema_version": "1.0",
        "is_my_domain": False,
        "domain_rejection_reason": f"에이전트 내부 오류: {error_message}",
        "query": question,
        "reasoning_flow": None,
        "macro_analysis": None,
        "geo_analysis": None,
        "overall_risk_level": None,
        "key_signals": [],
        "summary": None,
        "confidence_score": 0.0,
        "data_sources": [],
        "_parse_error": True,
        "_agent_meta": {"fatal_error": error_message},
    }
