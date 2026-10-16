"""
Final report renderer for the semiconductor intelligence agent.
"""

from __future__ import annotations

from app.agent.llm import call_llm
from app.agent.models import (
    CompetitiveMap,
    ExpectedValue,
    IntelligenceReport,
    MarketCheck,
    TechContext,
    TechEvaluation,
    TechEvent,
)

_STARS = {
    1: "★",
    2: "★★",
    3: "★★★",
    4: "★★★★",
    5: "★★★★★",
}
_TRL_LABEL = {
    1: "기초 개념",
    2: "개념 정립",
    3: "개념 검증",
    4: "실험실 검증",
    5: "유효성 검증",
    6: "파일럿 검증",
    7: "시제품 완성",
    8: "양산 준비",
    9: "양산 완료",
}
_HORIZON_KR = {
    "short": "단기(~3개월)",
    "mid": "중기(~1년)",
    "long": "장기(2년+)",
}
_SIGNAL_KR = {
    "unavailable": "판단 유보: 시장 자료 부재 또는 조회 실패",
    "buy_signal": "차트/수급상 매수 시그널",
    "caution": "이미 상당 부분 반영되어 추격 주의",
    "neutral": "중립. 추가 데이터 확인 필요",
}


def build_report(
    event: TechEvent,
    ctx: TechContext,
    evaluation: TechEvaluation,
    competitive: CompetitiveMap,
    ev: ExpectedValue,
    market: MarketCheck,
) -> IntelligenceReport:
    headline = _make_headline(event, evaluation, ev)
    full_report = _render_markdown(event, evaluation, competitive, ev, market)
    as_of = market.as_of or ev.as_of or ctx.as_of or event.published_at

    return IntelligenceReport(
        event=event,
        context=ctx,
        evaluation=evaluation,
        competitive=competitive,
        ev=ev,
        market=market,
        as_of=as_of,
        headline=headline,
        full_report=full_report,
    )


def _make_headline(event: TechEvent, evaluation: TechEvaluation, ev: ExpectedValue) -> str:
    system = "반도체 기술 분석가입니다. 짧은 한국어 문장으로만 답하세요."
    user = f"""다음 이벤트를 한 줄 헤드라인으로 요약해줘.

제목: {event.title}
혁신성: {evaluation.innovation_score}/5
기댓값: {ev.ev_score}
근거: {ev.rationale[:220]}"""
    return call_llm(system, user).strip()


def _company_lines(items: list[dict]) -> str:
    if not items:
        return "  - 없음"
    return "\n".join(
        f"  - **{item.get('company', '')}** ({item.get('code', '')})"
        f" [{item.get('role', '')}] {float(item['change_pct']):+.1f}%"
        f" ({item.get('observed_from') or '미상'} ~ {item.get('data_as_of') or '미상'})"
        for item in items
    )


def _render_markdown(
    event: TechEvent,
    evaluation: TechEvaluation,
    competitive: CompetitiveMap,
    ev: ExpectedValue,
    market: MarketCheck,
) -> str:
    beneficiaries = "\n".join(
        f"  - **{b['company']}** ({b.get('code','')}) - {b['reason']}"
        for b in competitive.beneficiaries
    ) or "  - 없음"

    threats = "\n".join(
        f"  - **{t['company']}** ({t.get('code','')}) - {t['reason']}"
        for t in competitive.threats
    ) or "  - 없음"

    ripple = "\n".join(f"  - {r}" for r in competitive.ripple_effects) or "  - 없음"
    key_claims = "\n".join(f"  - {c}" for c in evaluation.key_claims) or "  - 없음"
    company_moves = _company_lines(market.company_moves)
    price_change = "미상" if market.price_change_pct is None else f"{market.price_change_pct:+.1f}%"
    priced_in = "미상" if market.already_priced_in is None else "예" if market.already_priced_in else "아니오"
    limitations = "\n".join(f"- {item}" for item in market.limitations)

    return f"""# 기술 인텔리전스 리포트
**{event.title}**
> {event.source_type.upper()} | {event.source} | 게시일: {event.published_at or '미상'}
> 분석 생성 시각 | {market.as_of or ev.as_of or '미상'} (자료 관측일이 아님)

---

## 1) 기술 요약

| 항목 | 값 |
|---|---|
| 기술 혁신성 | {_STARS.get(evaluation.innovation_score, '★')} ({evaluation.innovation_score}/5) |
| TRL | {evaluation.trl}/9 - {_TRL_LABEL.get(evaluation.trl, '')} |
| 재현 가능성 | {evaluation.reproducibility} |
| 선행 기술 존재 | {'있음' if evaluation.prior_art_exists else '없음/희소'} |

**핵심 주장**
{key_claims}

**요약**
{evaluation.summary}

---

## 2) 경쟁 구도

**수혜 기업**
{beneficiaries}

**위협 기업**
{threats}

**파급 효과**
{ripple}

{competitive.competitive_summary}

---

## 3) 기댓값

| 항목 | 값 |
|---|---|
| P(기술 실현) | {ev.p_realization:.0%} |
| P(수혜 기업 포착) | {ev.p_benefit:.0%} |
| 임팩트 크기 | {ev.impact_magnitude} |
| 반영 시계 | {_HORIZON_KR.get(ev.time_horizon, ev.time_horizon)} |
| 종합 기댓값 | {ev.ev_score.upper()} |

{ev.rationale}

---

## 4) 시장 반영 / 주가 영향

**판단**
{market.thesis}

**시장 시그널**
{_SIGNAL_KR.get(market.signal, market.signal)}

**수혜주 최근 변화**
{market.note}

**개별 종목 변화**
{company_moves}

**상태**
- 조회 상태: {market.analysis_status}
- 이미 반영 여부: {priced_in}
- 관측된 종목의 구간 평균 변화: {price_change}
- 반영 시계: {_HORIZON_KR.get(market.time_horizon, market.time_horizon)}
{limitations}
"""
