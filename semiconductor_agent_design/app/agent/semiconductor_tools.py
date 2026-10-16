from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
import copy
from typing import Any

from app.agent.company_profiles import get_company_profile as _load_company_profile
from app.agent.semiconductor_event_utils import normalize_event_from_evidence
from app.agent.tech_potential import assess_technology_potential
from app.db.postgres import get_pg_conn
from app.experience import find_similar_experience_cases, get_case_profile, get_experience_profile
from app.rag.evidence_builder import build_evidence_pack


OFFICIAL_SOURCE_BY_COMPANY: dict[str, list[str]] = {
    "005930": ["samsung_global_newsroom"],
    "000660": ["skhynix_newsroom", "skhynix_press_center"],
    "042700": [],
    "NVDA": ["nvidia_ir_rss", "nvidia_newsroom", "nvidia_blog_feed", "nvidia_developer_blog", "nvidia_newsroom_home"],
    "TSM": ["tsmc_monthly_revenue"],
    "ASML": ["asml_press_releases"],
    "AMAT": ["applied_materials_ir", "applied_materials_newsroom"],
    "MU": ["micron_ir", "micron_insight_feed", "micron_investor_home"],
}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    status: str
    category: str
    implemented: bool
    description: str
    notes: str = ""


def _iso(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value or "")


@lru_cache(maxsize=256)
def _cached_pack(query: str, company: str, domain: str, top_k: int):
    return build_evidence_pack(query, company=company or None, domain=domain or None, top_k=top_k)


def rag_search(query: str, company: str = "", domain: str = "", top_k: int = 8) -> dict[str, Any]:
    pack = _cached_pack(query, company, domain, top_k)
    return {
        "tool": "rag_search",
        "query": query,
        "company": company,
        "domain": domain,
        "coverage": pack.coverage,
        "items": [asdict(item) for item in pack.items],
        "context_text": pack.context_text(max_chars=5000),
    }


def get_company_official_docs(company: str, since: str = "", limit: int = 10, topic: str = "") -> dict[str, Any]:
    company = str(company or "").strip().upper()
    since_dt: datetime | None = None
    if since:
        try:
            since_dt = datetime.fromisoformat(str(since).replace("Z", "+00:00"))
        except Exception:
            since_dt = None

    if topic.strip():
        pack = _cached_pack(topic, company, "", max(limit * 2, 8))
        items = []
        for item in pack.items:
            if item.source_type != "company_official":
                continue
            items.append(
                {
                    "doc_uid": item.payload.get("doc_uid", ""),
                    "source": item.source,
                    "source_type": item.source_type,
                    "title": item.title,
                    "summary": item.text[:1200],
                    "url": item.payload.get("url", ""),
                    "published_at": item.published_at,
                    "collected_at": item.payload.get("collected_at", ""),
                    "tags": item.payload.get("tags", []),
                    "confidence": float(item.raw_score or 0.0),
                    "evidence_score": float(item.evidence_score or 0.0),
                    "extra": item.payload,
                }
            )
            if len(items) >= max(1, int(limit)):
                break
        return {
            "tool": "get_company_official_docs",
            "company": company,
            "since": since,
            "topic": topic,
            "count": len(items),
            "items": items,
        }

    sources = OFFICIAL_SOURCE_BY_COMPANY.get(company, [])
    clauses = ["source_type = 'company_official'"]
    params: list[Any] = []

    if company:
        clauses.append("(COALESCE(extra->>'company_code', extra->>'company', '') = %s OR source = ANY(%s))")
        params.extend([company, sources or [""]])
    if since_dt is not None:
        clauses.append("COALESCE(published_at, collected_at) >= %s")
        params.append(since_dt)

    sql = f"""
        SELECT doc_uid, source, source_type, title, summary, url, published_at, collected_at,
               tags, confidence, extra
        FROM tech_documents
        WHERE {" AND ".join(clauses)}
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT %s
    """
    params.append(max(1, int(limit)))

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    items = []
    for row in rows or []:
        doc_uid, source, source_type, title, summary, url, published_at, collected_at, tags, confidence, extra = row
        items.append(
            {
                "doc_uid": doc_uid,
                "source": source,
                "source_type": source_type,
                "title": title,
                "summary": summary,
                "url": url,
                "published_at": _iso(published_at),
                "collected_at": _iso(collected_at),
                "tags": list(tags or []),
                "confidence": float(confidence or 0.0),
                "extra": dict(extra or {}),
            }
        )
    return {
        "tool": "get_company_official_docs",
        "company": company,
        "since": since,
        "topic": topic,
        "count": len(items),
        "items": items,
    }


def get_company_profile(company: str) -> dict[str, Any]:
    profile = _load_company_profile(company)
    return {
        "tool": "get_company_profile",
        "company": str(company or "").strip().upper(),
        "status": "available" if profile else "missing",
        "profile": profile,
    }


def get_competitor_docs(companies: list[str], topic: str, top_k_per_company: int = 4) -> dict[str, Any]:
    results = []
    for code in [str(c).strip().upper() for c in companies if str(c).strip()]:
        pack = _cached_pack(topic, code, "", top_k_per_company)
        results.append(
            {
                "company": code,
                "coverage": pack.coverage,
                "items": [asdict(item) for item in pack.items],
            }
        )
    return {
        "tool": "get_competitor_docs",
        "topic": topic,
        "companies": [r["company"] for r in results],
        "results": results,
    }


def get_similar_cases(
    event_type: str,
    company: str,
    domain: str = "",
    horizon_days: int = 30,
    limit: int = 5,
) -> dict[str, Any]:
    """
    Partial implementation.

    We do not yet persist an exact `event_type` field in experience memory, so this
    tool uses company/domain/horizon filters and returns the requested event_type as
    a search hint.
    """
    cases = find_similar_experience_cases(
        company_code=company,
        horizon_days=horizon_days,
        related_domain=domain or None,
        event_type=event_type or None,
        limit=limit,
    )
    return {
        "tool": "get_similar_cases",
        "status": "partial",
        "event_type": event_type,
        "company": company,
        "domain": domain,
        "horizon_days": horizon_days,
        "count": len(cases),
        "items": cases,
    }


def get_standard_docs(topic: str, top_k: int = 6) -> dict[str, Any]:
    pack = _cached_pack(topic, "", "standards", top_k)
    return {
        "tool": "get_standard_docs",
        "topic": topic,
        "coverage": pack.coverage,
        "items": [asdict(item) for item in pack.items],
        "context_text": pack.context_text(max_chars=4000),
    }


def get_backtest_profile(
    event_type: str,
    company: str,
    domain: str = "",
    horizon_days: int = 30,
) -> dict[str, Any]:
    profile = get_case_profile(
        company_code=company or None,
        horizon_days=horizon_days,
        related_domain=domain or None,
        event_type=event_type or None,
    )
    profiles = get_experience_profile(
        company_code=company or None,
        horizon_days=horizon_days,
        related_domain=domain or None,
        stat_group="company_horizon",
    ) if not event_type else []
    return {
        "tool": "get_backtest_profile",
        "status": "available" if event_type else "partial",
        "event_type": event_type,
        "company": company,
        "domain": domain,
        "horizon_days": horizon_days,
        "profile": profile,
        "profiles": profiles,
    }


def get_event_candidates(query: str = "", company: str = "", domain: str = "", limit: int = 10) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if company:
        clauses.append("related_company = %s")
        params.append(company)
    if domain:
        clauses.append("related_domain = %s")
        params.append(domain)
    if query:
        clauses.append("(title ILIKE %s OR summary ILIKE %s)")
        like = f"%{query}%"
        params.extend([like, like])

    sql = f"""
        SELECT event_id, event_date, event_type, source, source_tier, title, summary,
               related_company, related_domain, evidence_doc_uid, confidence, status, created_at, extra
        FROM event_candidates
        WHERE {" AND ".join(clauses)}
        ORDER BY event_date DESC
        LIMIT %s
    """
    params.append(max(1, int(limit)))

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    items = []
    for row in rows or []:
        (
            event_id,
            event_date,
            event_type,
            source,
            source_tier,
            title,
            summary,
            related_company,
            related_domain,
            evidence_doc_uid,
            confidence,
            status,
            created_at,
            extra,
        ) = row
        items.append(
            {
                "event_id": event_id,
                "event_date": _iso(event_date),
                "event_type": event_type,
                "source": source,
                "source_tier": int(source_tier),
                "title": title,
                "summary": summary,
                "related_company": related_company,
                "related_domain": related_domain,
                "evidence_doc_uid": evidence_doc_uid,
                "confidence": float(confidence or 0.0),
                "status": status,
                "created_at": _iso(created_at),
                "extra": dict(extra or {}),
            }
        )
    return {
        "tool": "get_event_candidates",
        "query": query,
        "company": company,
        "domain": domain,
        "count": len(items),
        "items": items,
    }


def extract_tech_event_from_docs(query: str, company: str = "", domain: str = "", top_k: int = 8) -> dict[str, Any]:
    pack = _cached_pack(query, company, domain, top_k)
    items = [asdict(item) for item in pack.items]
    existing = get_event_candidates(query=query, company=company, domain=domain, limit=3)
    if existing.get("count", 0):
        top = (existing.get("items") or [])[0]
        return {
            "tool": "extract_tech_event_from_docs",
            "status": "normalized_from_existing_event",
            "query": query,
            "event": normalize_event_from_evidence(
                query=query,
                items=items,
                company=company,
                domain=domain,
                existing_event=top,
                recent_90d_count=int((pack.coverage or {}).get("recent_90d_count", 0) or 0),
            ),
            "evidence": items,
        }

    if not items:
        return {
            "tool": "extract_tech_event_from_docs",
            "status": "no_evidence",
            "query": query,
            "event": {},
            "evidence": [],
        }

    return {
        "tool": "extract_tech_event_from_docs",
        "status": "normalized_from_documents",
        "query": query,
        "event": normalize_event_from_evidence(
            query=query,
            items=items,
            company=company,
            domain=domain,
            recent_90d_count=int((pack.coverage or {}).get("recent_90d_count", 0) or 0),
        ),
        "evidence": items,
    }


def get_evidence_gap_check(
    question: str,
    company: str = "",
    domain: str = "",
    horizon_days: int = 30,
    coverage: dict[str, Any] | None = None,
    official_count: int | None = None,
    standard_count: int | None = None,
    similar_case_count: int | None = None,
    existing_event_count: int | None = None,
) -> dict[str, Any]:
    if coverage is None:
        pack = _cached_pack(question, company, domain, 8)
        coverage = copy.deepcopy(pack.coverage or {})
    official_count = int(official_count or 0)
    standard_count = int(standard_count or 0)
    similar_case_count = int(similar_case_count or 0)
    existing_event_count = int(existing_event_count or 0)

    gaps: list[str] = []
    next_tools: list[str] = []

    if official_count == 0:
        gaps.append("official_company_evidence_missing")
        next_tools.append("get_company_official_docs")
    if int(len(coverage.get("sources", []) or [])) < 2:
        gaps.append("source_diversity_low")
        next_tools.append("rag_search")
    if int(coverage.get("recent_90d_count", 0) or 0) == 0:
        gaps.append("recent_evidence_thin")
        next_tools.append("rag_search")
    if standard_count == 0:
        gaps.append("standard_or_roadmap_context_missing")
        next_tools.append("get_standard_docs")
    if similar_case_count == 0:
        gaps.append("similar_case_memory_missing")
        next_tools.append("get_similar_cases")
    if existing_event_count == 0:
        gaps.append("tech_event_context_missing")
        next_tools.append("extract_tech_event_from_docs")

    return {
        "tool": "get_evidence_gap_check",
        "question": question,
        "company": company,
        "domain": domain,
        "horizon_days": horizon_days,
        "gaps": list(dict.fromkeys(gaps)),
        "next_tools": list(dict.fromkeys(next_tools)),
        "coverage": coverage,
        "official_count": official_count,
        "standard_count": standard_count,
        "similar_case_count": similar_case_count,
        "existing_event_count": existing_event_count,
    }


def finalize_assessment(question: str, company: str = "", domain: str = "", top_k: int = 8) -> dict[str, Any]:
    assessment = assess_technology_potential(
        question,
        company_hint=company,
        domain_hint=domain,
        top_k=top_k,
    )
    return {
        "tool": "finalize_assessment",
        "assessment": asdict(assessment),
    }


def get_tool_inventory() -> list[ToolSpec]:
    return [
        ToolSpec(
            name="rag_search",
            status="available",
            category="retrieval",
            implemented=True,
            description="Search our RAG evidence store by query/company/domain.",
        ),
        ToolSpec(
            name="get_company_profile",
            status="available",
            category="company_context",
            implemented=True,
            description="Return a semiconductor value-chain profile for a company.",
            notes="Useful for anchoring company impact in fixed business context before interpreting events.",
        ),
        ToolSpec(
            name="get_company_official_docs",
            status="available",
            category="official_docs",
            implemented=True,
            description="Fetch company-official documents after an optional cutoff.",
        ),
        ToolSpec(
            name="get_competitor_docs",
            status="available",
            category="comparison",
            implemented=True,
            description="Pull evidence packs for multiple competitor companies on the same topic.",
        ),
        ToolSpec(
            name="get_similar_cases",
            status="available",
            category="experience_memory",
            implemented=True,
            description="Find past similar backtest cases.",
            notes="Supports event_type-aware lookup using heuristic event typing stored in experience memory.",
        ),
        ToolSpec(
            name="get_standard_docs",
            status="available",
            category="standards",
            implemented=True,
            description="Fetch standards/roadmap-oriented evidence for a topic.",
        ),
        ToolSpec(
            name="get_backtest_profile",
            status="available",
            category="experience_memory",
            implemented=True,
            description="Return company/domain/horizon backtest profile stats.",
            notes="Supports event_type-aware case profiling; legacy aggregate profiles remain available when event_type is omitted.",
        ),
        ToolSpec(
            name="get_event_candidates",
            status="available",
            category="events",
            implemented=True,
            description="Query structured technology events by query/company/domain.",
        ),
        ToolSpec(
            name="finalize_assessment",
            status="available",
            category="assessment",
            implemented=True,
            description="Generate the final semiconductor technology assessment.",
        ),
        ToolSpec(
            name="extract_tech_event_from_docs",
            status="available",
            category="events",
            implemented=True,
            description="Turn retrieved documents into one normalized technology event object.",
            notes="Heuristic v1 built from retrieved evidence and existing event_candidates.",
        ),
        ToolSpec(
            name="get_evidence_gap_check",
            status="available",
            category="react_control",
            implemented=True,
            description="Tell the agent what evidence is missing before another ReAct step.",
            notes="Basic heuristic version; can be made more event-aware later.",
        ),
    ]
