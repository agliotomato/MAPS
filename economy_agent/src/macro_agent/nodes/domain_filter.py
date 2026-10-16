"""
Domain Filter Node — 반도체 거시경제·지정학 도메인 사전 필터

실행 흐름:
    1차) 정규식 키워드 매칭 (LLM 호출 없음, ~0ms)
    2차) gpt-4o-mini 빠른 판단 (키워드 미매칭 시에만, ~300ms)
    폴백) 판단 실패 시 허용 (false negative 방지)

도메인 外 질문이면 analysis_result에 거부 스키마를 세팅하고
is_domain=False를 반환 → graph.py에서 END로 라우팅됩니다.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from macro_agent.state import AgentState

logger = logging.getLogger(__name__)

# ── 1차 키워드 정규식 ─────────────────────────────────────────────────

_DOMAIN_PATTERN = re.compile(
    # 반도체 제품·공정
    r"반도체|semiconductor|chip(?:s)?|칩|HBM|DRAM|NAND|LPDDR|파운드리|foundry"
    r"|웨이퍼|wafer|팹|fab"
    # 시장 지수·사이클
    r"|SOX|필라델피아 반도체|반도체지수|재고조정|업황|다운사이클|업사이클"
    # 거시경제
    r"|환율|금리|거시경제|macro|PMI|CPI|GDP|인플레이션|inflation"
    r"|연준|Fed(?:eral)?|FOMC|기준금리|통화정책"
    r"|달러|원달러|USD|KRW|엔화|위안|JPY|CNY|FX|환헤지"
    # 지정학·무역
    r"|지정학|geopolit|수출통제|export.?control|BIS|EAR|Entity.?List"
    r"|OFAC|제재|sanction|바세나르|Wassenaar"
    r"|무역전쟁|trade.?war|관세|tariff|미중|US.{0,3}China|대만해협"
    r"|공급망|supply.?chain|friend.?shor|칩4|Chip4"
    # 기업 (거시 맥락에서 언급될 경우)
    r"|TSMC|삼성전자|SK하이닉스|인텔|엔비디아|ASML|퀄컴|마이크론|Micron"
    # 지표·통계
    r"|수출|출하|재고|inventory|생산지수|ECOS|FRED|yfinance",
    re.IGNORECASE,
)

# ── 2차 gpt-4o-mini 판단 ──────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_mini_llm():
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=5)


def _llm_domain_check(question: str) -> bool:
    """gpt-4o-mini로 도메인 여부를 YES/NO로 판단합니다."""
    mini = _get_mini_llm()
    resp = mini.invoke([
        (
            "system",
            "You are a domain classifier for a semiconductor macroeconomic & geopolitical risk agent. "
            "Answer only 'YES' if the question relates to: semiconductor industry cycles, "
            "exchange rates, interest rates, trade policy, export controls, geopolitical risks "
            "affecting semiconductors, or macroeconomic indicators. Answer 'NO' otherwise.",
        ),
        ("human", question),
    ])
    return "YES" in resp.content.upper()


# ── 거부 스키마 팩토리 ────────────────────────────────────────────────

def _rejection_schema(question: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "is_my_domain": False,
        "domain_rejection_reason": (
            "이 에이전트는 반도체 도메인 거시경제·지정학 리스크 분석 전문입니다. "
            "담당 영역을 벗어난 질문입니다. 적절한 에이전트로 라우팅하세요."
        ),
        "query": question,
        "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
        "reasoning_flow": None,
        "macro_analysis": None,
        "geo_analysis": None,
        "overall_risk_level": None,
        "key_signals": [],
        "summary": None,
        "confidence_score": 0.0,
        "data_sources": [],
    }


# ── 메인 노드 함수 ────────────────────────────────────────────────────

def domain_filter_node(state: AgentState) -> AgentState:
    """
    LangGraph 노드: 도메인 필터.

    1차 키워드 매칭 → 통과 시 is_domain=True 즉시 반환 (LLM 호출 없음).
    미매칭 시 gpt-4o-mini로 2차 판단.
    판단 실패 시 허용(True)으로 폴백.
    """
    question = state.get("question", "").strip()
    warnings: list[str] = list(state.get("warnings", []))

    # ── 1차: 정규식 키워드 (비용 없음) ──────────────────────────────
    if _DOMAIN_PATTERN.search(question):
        logger.info("도메인 필터: 키워드 매칭 통과")
        return {**state, "is_domain": True, "warnings": warnings}

    # ── 2차: gpt-4o-mini ─────────────────────────────────────────────
    logger.info("도메인 필터: 키워드 미매칭 → gpt-4o-mini 판단 시작")
    try:
        is_domain = _llm_domain_check(question)
        logger.info("도메인 필터: gpt-4o-mini 결과 = %s", "통과" if is_domain else "거부")
    except Exception as exc:
        logger.warning("도메인 필터 LLM 호출 실패, 허용으로 폴백: %s", exc)
        warnings.append(f"도메인 필터 판단 실패 (허용 처리): {exc}")
        is_domain = True

    if not is_domain:
        logger.info("도메인 필터: 거부 — %s", question[:60])
        return {
            **state,
            "is_domain": False,
            "analysis_result": _rejection_schema(question),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "warnings": warnings,
        }

    return {**state, "is_domain": True, "warnings": warnings}
