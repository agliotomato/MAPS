from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from app.agent.semiconductor_tools import (
    extract_tech_event_from_docs,
    finalize_assessment,
    get_backtest_profile,
    get_company_official_docs,
    get_competitor_docs,
    get_evidence_gap_check,
    get_event_candidates,
    get_similar_cases,
    get_standard_docs,
    get_tool_inventory,
    rag_search,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReactToolCall:
    step: int
    tool: str
    reason: str
    args: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemiconductorReactState:
    question: str
    company: str = ""
    domain: str = ""
    horizon_days: int = 30
    as_of: str = field(default_factory=_now_iso)
    available_tools: list[dict[str, Any]] = field(default_factory=list)
    partial_tools: list[dict[str, Any]] = field(default_factory=list)
    planned_tools: list[dict[str, Any]] = field(default_factory=list)
    initial_answer: dict[str, Any] = field(default_factory=dict)
    observations: dict[str, Any] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    tool_calls: list[ReactToolCall] = field(default_factory=list)
    final_answer: dict[str, Any] = field(default_factory=dict)


def _inventory_groups() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    available: list[dict[str, Any]] = []
    partial: list[dict[str, Any]] = []
    planned: list[dict[str, Any]] = []
    for item in get_tool_inventory():
        payload = asdict(item)
        if item.status == "available":
            available.append(payload)
        elif item.status == "partial":
            partial.append(payload)
        else:
            planned.append(payload)
    return available, partial, planned


def _summarize_result(tool: str, result: dict[str, Any]) -> dict[str, Any]:
    if tool in {"rag_search", "get_standard_docs"}:
        coverage = result.get("coverage", {}) or {}
        return {
            "count": int(coverage.get("count", 0) or 0),
            "source_count": len(coverage.get("sources", []) or []),
            "recent_90d_count": int(coverage.get("recent_90d_count", 0) or 0),
        }
    if tool in {"get_company_official_docs", "get_similar_cases", "get_event_candidates"}:
        return {"count": int(result.get("count", 0) or 0)}
    if tool == "extract_tech_event_from_docs":
        event = result.get("event", {}) or {}
        return {
            "status": result.get("status", ""),
            "event_type": event.get("event_type", ""),
            "technology": event.get("technology", ""),
            "company": event.get("related_company", ""),
        }
    if tool == "get_backtest_profile":
        return {"profile_count": len(result.get("profiles", []) or [])}
    if tool == "get_competitor_docs":
        results = result.get("results", []) or []
        return {
            "company_count": len(results),
            "non_empty_companies": sum(1 for item in results if (item.get("coverage", {}) or {}).get("count", 0) > 0),
        }
    if tool == "finalize_assessment":
        assessment = result.get("assessment", {}) or {}
        return {
            "recommendation": assessment.get("recommendation", ""),
            "confidence": assessment.get("confidence", 0.0),
            "evidence_grade": (assessment.get("evidence_quality", {}) or {}).get("grade", ""),
        }
    if tool == "get_evidence_gap_check":
        return {
            "gaps": list(result.get("gaps", []) or []),
            "next_tools": list(result.get("next_tools", []) or []),
        }
    return {}


def _call_tool(
    state: SemiconductorReactState,
    *,
    step: int,
    tool_name: str,
    reason: str,
    args: dict[str, Any],
    fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    result = fn(**args)
    state.tool_calls.append(
        ReactToolCall(
            step=step,
            tool=tool_name,
            reason=reason,
            args=args,
            result_summary=_summarize_result(tool_name, result),
        )
    )
    state.observations[tool_name] = result
    return result


def _choose_competitors(company: str) -> list[str]:
    company = str(company or "").strip().upper()
    if company == "005930":
        return ["000660", "MU"]
    if company == "000660":
        return ["005930", "MU"]
    if company == "042700":
        return ["005930", "000660", "ASML"]
    return ["005930", "000660", "MU"]


def _gap_inputs_from_state(state: SemiconductorReactState) -> dict[str, Any]:
    initial = state.initial_answer or {}
    rag = initial.get("rag_search") or state.observations.get("rag_search") or {}
    official = initial.get("company_official_docs") or state.observations.get("get_company_official_docs") or {}
    standards = initial.get("standard_docs") or state.observations.get("get_standard_docs") or {}
    similar = initial.get("similar_cases") or state.observations.get("get_similar_cases") or {}
    events = initial.get("event_candidates") or state.observations.get("get_event_candidates") or {}
    return {
        "question": state.question,
        "company": state.company,
        "domain": state.domain,
        "horizon_days": state.horizon_days,
        "coverage": rag.get("coverage", {}) or {},
        "official_count": int(official.get("count", 0) or 0),
        "standard_count": int((standards.get("coverage", {}) or {}).get("count", 0) or 0),
        "similar_case_count": int(similar.get("count", 0) or 0),
        "existing_event_count": int(events.get("count", 0) or 0),
    }


def _current_event_type(state: SemiconductorReactState) -> str:
    normalized = (
        state.observations.get("extract_tech_event_from_docs")
        or state.initial_answer.get("normalized_event")
        or {}
    )
    event = normalized.get("event", {}) if isinstance(normalized, dict) else {}
    return str(event.get("event_type") or "").strip() or "unknown"


def run_bounded_semiconductor_react(
    question: str,
    *,
    company: str = "",
    domain: str = "",
    horizon_days: int = 30,
    max_steps: int = 3,
    top_k: int = 8,
    finalize: bool = True,
) -> SemiconductorReactState:
    available, partial, planned = _inventory_groups()
    state = SemiconductorReactState(
        question=question,
        company=company,
        domain=domain,
        horizon_days=horizon_days,
        available_tools=available,
        partial_tools=partial,
        planned_tools=planned,
    )

    # Step 0: data-first observation skeleton from our own stores, without an LLM pass yet.
    state.initial_answer = {
        "rag_search": rag_search(question, company=company, domain=domain, top_k=top_k),
        "event_candidates": get_event_candidates(query=question, company=company, domain=domain, limit=4),
        "normalized_event": extract_tech_event_from_docs(question, company=company, domain=domain, top_k=min(top_k, 6)),
    }
    state.observations["initial_answer"] = state.initial_answer

    gap = _call_tool(
        state,
        step=0,
        tool_name="get_evidence_gap_check",
        reason="Check whether the data-first answer is missing official, standards, or similar-case evidence.",
        args=_gap_inputs_from_state(state),
        fn=get_evidence_gap_check,
    )
    state.gaps = list(gap.get("gaps", []) or [])

    tool_order = {
        "get_company_official_docs": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_company_official_docs",
            reason="Fill missing company-official evidence for the main company.",
            args={"company": company, "topic": question, "limit": 4},
            fn=get_company_official_docs,
        ),
        "rag_search": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="rag_search",
            reason="Expand general evidence coverage when source diversity or recency is thin.",
            args={"query": question, "company": company, "domain": domain, "top_k": top_k},
            fn=rag_search,
        ),
        "get_standard_docs": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_standard_docs",
            reason="Add standards or roadmap context for technology maturity and bottleneck interpretation.",
            args={"topic": question, "top_k": 4},
            fn=get_standard_docs,
        ),
        "extract_tech_event_from_docs": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="extract_tech_event_from_docs",
            reason="Normalize retrieved documents into one price-relevant semiconductor technology event.",
            args={"query": question, "company": company, "domain": domain, "top_k": min(top_k, 6)},
            fn=extract_tech_event_from_docs,
        ),
        "get_similar_cases": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_similar_cases",
            reason="Bring in experience-memory examples to check how similar events behaved historically.",
            args={
                "event_type": _current_event_type(state),
                "company": company,
                "domain": domain,
                "horizon_days": horizon_days,
                "limit": 4,
            },
            fn=get_similar_cases,
        ),
        "get_competitor_docs": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_competitor_docs",
            reason="Add competitor-side evidence so company impact is not judged from one company alone.",
            args={"companies": _choose_competitors(company), "topic": question, "top_k_per_company": 3},
            fn=get_competitor_docs,
        ),
        "get_backtest_profile": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_backtest_profile",
            reason="Check historical hit-rate and recurring failure modes before finalizing confidence.",
            args={
                "event_type": _current_event_type(state),
                "company": company,
                "domain": domain,
                "horizon_days": horizon_days,
            },
            fn=get_backtest_profile,
        ),
        "get_event_candidates": lambda: _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_event_candidates",
            reason="Check whether the event has already been normalized in our structured event store.",
            args={"query": question, "company": company, "domain": domain, "limit": 4},
            fn=get_event_candidates,
        ),
    }

    used_tools: set[str] = {"get_evidence_gap_check"}
    next_tools = list(gap.get("next_tools", []) or [])

    for _ in range(max_steps):
        if not next_tools:
            break
        tool_name = next_tools.pop(0)
        if tool_name in used_tools:
            continue
        if tool_name not in tool_order:
            continue
        tool_order[tool_name]()
        used_tools.add(tool_name)

        gap = _call_tool(
            state,
            step=len(state.tool_calls),
            tool_name="get_evidence_gap_check",
            reason="Re-check remaining evidence gaps after the latest tool call.",
            args=_gap_inputs_from_state(state),
            fn=get_evidence_gap_check,
        )
        state.gaps = list(gap.get("gaps", []) or [])
        next_tools = [name for name in (gap.get("next_tools", []) or []) if name not in used_tools]

    if finalize:
        state.final_answer = finalize_assessment(question, company=company, domain=domain, top_k=top_k)
    else:
        state.final_answer = {
            "tool": "finalize_assessment",
            "skipped": True,
            "reason": "dry_run",
        }
    state.observations["final_answer"] = state.final_answer
    return state


def render_bounded_react_summary(state: SemiconductorReactState) -> str:
    lines = [
        f"# Semiconductor Bounded ReAct",
        "",
        f"- Question: `{state.question}`",
        f"- Company: `{state.company}`",
        f"- Domain: `{state.domain}`",
        f"- Horizon: `{state.horizon_days}d`",
        f"- As of: `{state.as_of}`",
        "",
        "## Tool Inventory",
        f"- Available: {', '.join(item['name'] for item in state.available_tools) or '-'}",
        f"- Partial: {', '.join(item['name'] for item in state.partial_tools) or '-'}",
        f"- Planned: {', '.join(item['name'] for item in state.planned_tools) or '-'}",
        "",
        "## Tool Calls",
    ]
    if not state.tool_calls:
        lines.append("- None")
    else:
        for call in state.tool_calls:
            lines.append(
                f"- Step {call.step}: `{call.tool}` | {call.reason} | summary={call.result_summary}"
            )
    lines.extend(["", "## Remaining Gaps"])
    if state.gaps:
        lines.extend(f"- `{gap}`" for gap in state.gaps)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Final Assessment Snapshot",
            f"- Recommendation: `{(state.final_answer.get('assessment', {}) or {}).get('recommendation', '')}`",
            f"- Confidence: `{(state.final_answer.get('assessment', {}) or {}).get('confidence', 0.0)}`",
            f"- Evidence Grade: `{((state.final_answer.get('assessment', {}) or {}).get('evidence_quality', {}) or {}).get('grade', '')}`",
        ]
    )
    return "\n".join(lines)
