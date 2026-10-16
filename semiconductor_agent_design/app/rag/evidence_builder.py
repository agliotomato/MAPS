"""
Evidence pack builder for RAG.

The retriever returns matching chunks. This module turns them into a ranked
evidence pack that is useful for investment reasoning: high-trust sources,
fresh documents, company/domain alignment, and diversity are favored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from typing import Any

from app.rag.lexical import search_multi_query_lexical
from app.rag.retriever import search_multi_query
from app.taxonomy import taxonomy_analysis

logger = logging.getLogger(__name__)

SOURCE_TIER = {
    "samsung_global_newsroom": 1,
    "skhynix_newsroom": 1,
    "tsmc_monthly_revenue": 1,
    "asml_press_releases": 1,
    "nvidia_ir_rss": 1,
    "micron_ir": 1,
    "applied_materials_ir": 1,
    "dart": 1,
    "jedec": 1,
    "irds": 1,
    "arxiv": 2,
    "openalex": 2,
    "semantic_scholar": 2,
    "hotchips": 2,
    "semiconductor_engineering": 3,
    "eetimes": 3,
    "ieee_spectrum": 3,
    "rss_news": 3,
    "tech_blog": 3,
}

DOMAIN_KEYWORDS = {
    "hbm": ["hbm", "hbm3", "hbm3e", "hbm4", "high bandwidth memory"],
    "packaging": ["advanced packaging", "cowos", "hybrid bonding", "tc bonding", "tsv", "chiplet"],
    "litho": ["euv", "high-na", "lithography"],
    "nand": ["nand", "3d nand", "flash"],
    "dram": ["dram", "ddr5", "lpddr", "memory"],
    "logic": ["gaa", "finfet", "nanosheet", "foundry"],
    "financials": ["earnings", "revenue", "guidance", "sales", "results", "capex"],
}

_TAXONOMY_ANALYSIS = taxonomy_analysis()
if _TAXONOMY_ANALYSIS:
    for domain, hints in (_TAXONOMY_ANALYSIS.get("topic_clusters") or {}).items():
        DOMAIN_KEYWORDS.setdefault(domain, [])
        DOMAIN_KEYWORDS[domain] = list(dict.fromkeys(DOMAIN_KEYWORDS[domain] + [str(h) for h in (hints or [])]))

COMPANY_HINTS = {
    "005930": ["samsung", "삼성"],
    "000660": ["sk hynix", "hynix", "하이닉스"],
    "042700": ["hanmi", "한미반도체"],
    "NVDA": ["nvidia", "blackwell", "gpu"],
    "TSM": ["tsmc", "cowos"],
    "ASML": ["asml", "euv"],
    "AMAT": ["applied materials", "amat"],
    "LRCX": ["lam research", "lam"],
    "KLAC": ["kla", "k-l-a", "kla corporation"],
    "MU": ["micron"],
}


@dataclass
class EvidenceItem:
    id: str
    title: str
    text: str
    source: str
    source_type: str
    domain: str
    company: str
    year: int
    raw_score: float
    evidence_score: float
    published_at: str = ""
    reasons: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidencePack:
    query: str
    items: list[EvidenceItem]
    coverage: dict[str, Any]

    def context_text(self, max_chars: int | None = 6000) -> str:
        parts = []
        total = 0
        for idx, item in enumerate(self.items, start=1):
            block = (
                f"[E{idx}] source={item.source} tier={_tier(item.source)} "
                f"domain={item.domain} company={item.company} year={item.year} "
                f"published_at={item.published_at or 'unknown'} "
                f"score={item.evidence_score:.3f}\n"
                f"{item.title}\n{item.text}\n"
            )
            if max_chars is not None and total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n---\n".join(parts)


def _tier(source: str) -> int:
    return SOURCE_TIER.get(source, 3)


def _freshness(year: int) -> float:
    if not year:
        return 0.45
    now_year = datetime.now(timezone.utc).year
    age = max(0, now_year - int(year))
    if age <= 1:
        return 1.0
    if age <= 3:
        return 0.8
    if age <= 5:
        return 0.6
    return 0.35


def _parse_published_at(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        cleaned = text.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    try:
        dt = datetime.strptime(text[:10], "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _recency_boost(published_at: str | None, year: int) -> float:
    dt = _parse_published_at(published_at)
    if dt is None:
        return _freshness(year)
    now = datetime.now(timezone.utc)
    age_days = max(0, (now - dt).days)
    if age_days <= 30:
        return 1.0
    if age_days <= 90:
        return 0.92
    if age_days <= 180:
        return 0.84
    if age_days <= 365:
        return 0.74
    if age_days <= 730:
        return 0.62
    if age_days <= 1095:
        return 0.52
    return 0.38


def _days_since_published(published_at: str | None) -> int | None:
    dt = _parse_published_at(published_at)
    if dt is None:
        return None
    return max(0, (datetime.now(timezone.utc) - dt).days)


def _domain_match(text: str, wanted_domain: str | None) -> tuple[float, str | None]:
    lowered = text.lower()
    if wanted_domain:
        hits = DOMAIN_KEYWORDS.get(wanted_domain, [])
        if any(k in lowered for k in hits):
            return 1.0, wanted_domain
        return 0.35, None
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in lowered for k in kws):
            return 0.75, domain
    return 0.45, None


def _company_match(text: str, wanted_company: str | None) -> float:
    lowered = text.lower()
    if wanted_company:
        wanted_company = str(wanted_company)
        hints = COMPANY_HINTS.get(wanted_company, [wanted_company.lower()])
        return 1.0 if any(h in lowered for h in hints) else 0.4
    return 0.7 if any(any(h in lowered for h in hints) for hints in COMPANY_HINTS.values()) else 0.5


def _query_variants(query: str, company: str | None, domain: str | None) -> list[str]:
    queries = [
        query,
        f"{query} official company investor relations",
        f"{query} technical evidence paper standard roadmap",
        f"{query} news blog press release",
    ]
    if company:
        queries.append(f"{query} {company}")

    domain_boosters = {
        "hbm": "HBM3E HBM4 memory stack AI",
        "packaging": "chiplet hybrid bonding 2.5D 3D interposer",
        "litho": "EUV High-NA photoresist pellicle scanner",
        "nand": "3D NAND V-NAND flash",
        "dram": "DRAM DDR5 LPDDR HBM memory",
        "logic": "GAA FinFET nanosheet backside power",
        "process": "yield metrology inspection failure analysis",
        "materials": "substrate glass substrate photoresist slurry",
        "power": "SiC GaN power semiconductor reliability",
        "design": "EDA chiplet UCIe die-to-die",
        "reliability": "yield metrology inspection electromigration",
        "standards": "JEDEC IRDS standard roadmap",
        "business": "fabless foundry OSAT IDM ecosystem",
        "advanced_memory": "CXL PIM processing in memory",
        "device_structure": "GAA FinFET backside power",
        "lithography_process": "EUV High-NA PR pellicle ALD ALE",
        "advanced_packaging_tech": "TSV hybrid bonding CoWoS chiplet",
        "specialty_devices": "CIS PMIC RFIC FPGA",
        "inspection_failure_analysis": "TEM FIB reliability electromigration",
    }
    if domain:
        booster = domain_boosters.get(domain, "")
        if booster:
            queries.append(f"{query} {booster}")
    return list(dict.fromkeys(queries))


def _score_item(raw: dict[str, Any], company: str | None, domain: str | None) -> tuple[float, list[str]]:
    source = str(raw.get("source", ""))
    title = str(raw.get("title", ""))
    text = str(raw.get("text", ""))
    combined = f"{title} {text}"
    published_at = str(raw.get("published_at", "") or "")
    year = int(raw.get("year") or 0)

    tier_score = {1: 1.0, 2: 0.82, 3: 0.62, 4: 0.35}.get(_tier(source), 0.55)
    fresh = _freshness(year)
    recency = _recency_boost(published_at, year)
    d_score, matched_domain = _domain_match(combined, domain)
    c_score = _company_match(combined, company)
    retrieval_score = min(1.0, float(raw.get("score", 0.0)) * 40.0)

    score = (
        0.28 * tier_score
        + 0.18 * retrieval_score
        + 0.18 * d_score
        + 0.14 * c_score
        + 0.12 * fresh
        + 0.10 * recency
    )
    reasons = [
        f"tier={_tier(source)}",
        f"freshness={fresh:.2f}",
        f"recency={recency:.2f}",
        f"domain={matched_domain or raw.get('domain', '') or 'weak'}",
        f"company_match={c_score:.2f}",
        f"published_at={published_at or 'unknown'}",
    ]
    return score, reasons


def build_evidence_pack(
    query: str,
    *,
    company: str | None = None,
    domain: str | None = None,
    top_k: int = 8,
    prefetch_k: int = 24,
) -> EvidencePack:
    queries = _query_variants(query, company=company, domain=domain)
    search_mode = os.getenv("RAG_SEARCH_MODE", "hybrid").strip().lower()
    results = []
    if search_mode == "lexical_payload":
        results = search_multi_query_lexical(queries, top_k=prefetch_k)
    else:
        try:
            results = search_multi_query(queries, top_k=prefetch_k)
        except Exception as e:
            logger.warning("hybrid search failed, falling back to lexical | %s", e)
            results = []
        if not results:
            results = search_multi_query_lexical(queries, top_k=prefetch_k)

    seen_docs: set[str] = set()
    items: list[EvidenceItem] = []
    for raw in results:
        doc_key = str(raw.get("doc_uid") or raw.get("id"))
        if doc_key in seen_docs:
            continue
        seen_docs.add(doc_key)
        score, reasons = _score_item(raw, company, domain)
        items.append(
            EvidenceItem(
                id=str(raw.get("id", "")),
                title=str(raw.get("title", "")),
                text=str(raw.get("text", "")),
                source=str(raw.get("source", "")),
                source_type=str(raw.get("source_type", "")),
                domain=str(raw.get("domain", "")),
                company=str(raw.get("company", "")),
                year=int(raw.get("year") or 0),
                published_at=str(raw.get("published_at", "") or ""),
                raw_score=float(raw.get("score", 0.0)),
                evidence_score=score,
                reasons=reasons,
                payload=raw,
            )
        )

    items.sort(key=lambda x: x.evidence_score, reverse=True)
    selected: list[EvidenceItem] = []
    source_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    max_per_source = max(1, int(os.getenv("RAG_MAX_EVIDENCE_PER_SOURCE", "3")))
    max_per_domain = max(1, int(os.getenv("RAG_MAX_EVIDENCE_PER_DOMAIN", "2")))
    for item in items:
        if source_counts.get(item.source, 0) >= max_per_source:
            continue
        if item.domain and domain_counts.get(item.domain, 0) >= max_per_domain:
            continue
        selected.append(item)
        source_counts[item.source] = source_counts.get(item.source, 0) + 1
        if item.domain:
            domain_counts[item.domain] = domain_counts.get(item.domain, 0) + 1
        if len(selected) >= top_k:
            break
    if len(selected) < top_k:
        selected_ids = {item.id for item in selected}
        for item in items:
            if item.id in selected_ids:
                continue
            selected.append(item)
            if len(selected) >= top_k:
                break

    coverage = {
        "count": len(selected),
        "tier1_count": sum(1 for i in selected if _tier(i.source) == 1),
        "tier12_count": sum(1 for i in selected if _tier(i.source) <= 2),
        "domains": sorted(set(i.domain for i in selected if i.domain)),
        "sources": sorted(set(i.source for i in selected if i.source)),
        "avg_evidence_score": sum(i.evidence_score for i in selected) / len(selected) if selected else 0.0,
        "recent_365d_count": sum(1 for i in selected if (_days_since_published(i.published_at) is not None and _days_since_published(i.published_at) <= 365)),
        "recent_90d_count": sum(1 for i in selected if (_days_since_published(i.published_at) is not None and _days_since_published(i.published_at) <= 90)),
    }
    return EvidencePack(query=query, items=selected, coverage=coverage)
