"""
반도체 전문가 에이전트 — 메인 파이프라인

사용법:
    from app.agent.pipeline import run
    from app.agent.models import TechEvent

    event = TechEvent(
        title="HBM4 Cu-Cu Hybrid Bonding with Sub-1µm Pitch",
        content="...논문 초록...",
        source_type="paper",
        source="arxiv",
        published_at="2025-05-01",
    )
    report = run(event)
    print(report.full_report)
"""
from __future__ import annotations

import logging
import time

from app.agent.models import IntelligenceReport, TechEvent
from app.agent.steps import (
    calc_expected_value,
    competitive_map,
    context_collect,
    market_check,
    tech_evaluate,
)
from app.agent.reporter import build_report

_log = logging.getLogger(__name__)


def run(event: TechEvent, verbose: bool = True) -> IntelligenceReport:
    """
    5단계 파이프라인 실행.
    verbose=True 이면 각 단계 진행 상황 출력.
    """
    t0 = time.time()

    def _log_step(n: int, name: str):
        if verbose:
            print(f"[{n}/5] {name}...")

    # ① 기술 컨텍스트 수집
    _log_step(1, "기술 컨텍스트 수집 (RAG)")
    ctx = context_collect(event)

    # ② 기술 이해 & 평가
    _log_step(2, "기술 평가 (LLM)")
    evaluation = tech_evaluate(ctx)

    # ③ 경쟁 구도 매핑
    _log_step(3, "경쟁 구도 매핑 (LLM)")
    competitive = competitive_map(ctx, evaluation)

    # ④ 기댓값 계산
    _log_step(4, "기댓값 계산 (LLM)")
    ev = calc_expected_value(evaluation, competitive)

    # ⑤ 시장 반영 여부
    _log_step(5, "시장 반영 여부 확인 (Postgres)")
    market = market_check(event, ev, competitive)

    # 리포트 생성
    report = build_report(event, ctx, evaluation, competitive, ev, market)

    elapsed = time.time() - t0
    if verbose:
        print(f"\n✓ 완료 ({elapsed:.1f}s)\n")
        print("=" * 60)
        print(f"헤드라인: {report.headline}")
        print(f"기댓값: {ev.ev_score.upper()} | 시그널: {market.signal}")
        print("=" * 60)

    return report


def run_from_rag(query: str, verbose: bool = True) -> IntelligenceReport:
    """
    RAG 검색으로 관련성 최상위 자료를 선택해서 파이프라인 실행. 최신 자료임을 보장하지 않음.
    가장 관련성 높은 청크를 이벤트로 변환.
    """
    from app.rag.retriever import search

    results = search(query, top_k=1)
    if not results:
        raise ValueError(f"RAG 검색 결과 없음: {query}")

    top = results[0]
    published_at = top.get("published_at") or ""
    if hasattr(published_at, "isoformat"):
        published_at = published_at.isoformat()
    event = TechEvent(
        title=top.get("title", query),
        content=top.get("text", ""),
        source_type=top.get("source_type", "paper"),
        source=top.get("source", ""),
        published_at=str(published_at),
        url=top.get("url") or "",
        source_id=top.get("doc_uid") or top.get("id") or "",
        year=top.get("year") or None,
    )
    return run(event, verbose=verbose)
