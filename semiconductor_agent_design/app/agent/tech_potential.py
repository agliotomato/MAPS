"""
Technology potential assessment.

This module answers questions like:
- "When will a semiconductor technology matter to price action?"
- "Is this technology a bottleneck?"
- "Which company gets helped or hurt?"

It turns RAG evidence into a semiconductor-first assessment:
1. Catalyst Imminence
2. Bottleneck Importance
3. Company Impact
4. Evidence Quality
5. Novelty / Surprise
6. Revenue Linkage
7. Market Transmission Speed
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.agent.llm import call_llm_json
from app.agent.models import TechnologyPotentialAssessment
from app.agent.semiconductor_event_utils import normalize_event_from_evidence
from app.agent.semiconductor_prompt import make_semiconductor_system_prompt
from app.rag.evidence_builder import build_evidence_pack
from app.taxonomy import taxonomy_analysis

_log = logging.getLogger(__name__)


COMPANY_CANDIDATES = [
    ("005930", "Samsung Electronics"),
    ("000660", "SK hynix"),
    ("042700", "Hanmi Semiconductor"),
    ("NVDA", "NVIDIA"),
    ("TSM", "TSMC"),
    ("ASML", "ASML"),
    ("AMAT", "Applied Materials"),
    ("LRCX", "Lam Research"),
    ("KLAC", "KLA"),
    ("MU", "Micron"),
    ("INTC", "Intel"),
]

DOMAIN_HINTS = {
    "hbm": ["hbm", "high bandwidth memory", "memory stacking", "co-packaged"],
    "litho": ["euv", "high-na", "lithography", "photoresist"],
    "packaging": ["cowos", "hybrid bonding", "chiplet", "2.5d", "3d integration", "tsv"],
    "nand": ["nand", "3d nand", "flash"],
    "dram": ["dram", "ddr5", "lpddr"],
    "logic": ["gaa", "nanosheet", "finfet", "foundry"],
    "equipment": ["etch", "deposition", "inspection", "metrology", "scanner", "equipment"],
}

_PLAYBOOK_PATH = Path(__file__).resolve().parents[2] / "data" / "tech_potential_playbook.yaml"
_PLAYBOOK: dict[str, Any] = {}
_TAXONOMY_ANALYSIS: dict[str, Any] = taxonomy_analysis()
_DEFAULT_AXIS_SPEC: dict[str, str] = {
    "catalyst_imminence": "How quickly this semiconductor event can affect price action, focusing on 0-7d, 8-30d, 31-90d, and 3m+ windows.",
    "bottleneck_importance": "Whether the technology is a true industry bottleneck, how hard it is to substitute, and how supply constrained it is.",
    "company_impact": "Which companies benefit, face threats, or remain neutral, with strict evidence linkage.",
    "evidence_quality": "How strong the evidence is across official materials, papers, standards, and earnings sources.",
    "novelty": "Whether the information is new to the market or largely already known.",
    "revenue_linkage": "How directly and how quickly the technology can flow into revenue, guidance, or margins.",
    "market_transmission_speed": "How fast the market typically reacts to this type of semiconductor information.",
}
_DEFAULT_GOOD_ANSWER_CRITERIA: dict[str, list[str]] = {
    "catalyst_imminence": [
        "Explain the likely price-action window explicitly.",
        "Distinguish immediate catalysts from longer structural effects.",
    ],
    "bottleneck_importance": [
        "State whether this is a real bottleneck or just an important technology.",
        "Separate substitutability from supply constraint.",
    ],
    "company_impact": [
        "Name the beneficiary or threat company and explain the connection.",
        "Keep confidence conservative if evidence is weak.",
    ],
    "evidence_quality": [
        "Prefer official materials, papers, standards, and earnings over generic news.",
        "Call out missing or weak evidence explicitly.",
    ],
}


def _load_playbook() -> dict[str, Any]:
    global _PLAYBOOK
    if _PLAYBOOK:
        return _PLAYBOOK
    if not _PLAYBOOK_PATH.exists():
        _PLAYBOOK = {}
        return _PLAYBOOK
    try:
        data = yaml.safe_load(_PLAYBOOK_PATH.read_text(encoding="utf-8")) or {}
        _PLAYBOOK = data if isinstance(data, dict) else {}
        if _TAXONOMY_ANALYSIS:
            for key in ("evidence_priority", "topic_clusters", "good_answer_criteria", "axis_spec"):
                if key not in _PLAYBOOK and key in _TAXONOMY_ANALYSIS:
                    _PLAYBOOK[key] = _TAXONOMY_ANALYSIS[key]
            if "topic_clusters" in _TAXONOMY_ANALYSIS:
                merged_clusters = dict(_PLAYBOOK.get("topic_clusters") or {})
                for domain, hints in (_TAXONOMY_ANALYSIS.get("topic_clusters") or {}).items():
                    merged_clusters.setdefault(domain, [])
                    merged_clusters[domain] = list(dict.fromkeys((merged_clusters[domain] or []) + (hints or [])))
                _PLAYBOOK["topic_clusters"] = merged_clusters
        merged_axis = dict(_PLAYBOOK.get("axis_spec") or {})
        for key, value in _DEFAULT_AXIS_SPEC.items():
            merged_axis.setdefault(key, value)
        _PLAYBOOK["axis_spec"] = merged_axis
        merged_good = dict(_PLAYBOOK.get("good_answer_criteria") or {})
        for key, value in _DEFAULT_GOOD_ANSWER_CRITERIA.items():
            merged_good.setdefault(key, value)
        _PLAYBOOK["good_answer_criteria"] = merged_good
    except Exception as e:
        _log.warning("tech_potential playbook load failed: %s", e)
        _PLAYBOOK = {}
    return _PLAYBOOK


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _infer_domain(topic: str) -> str:
    t = _norm(topic)
    playbook = _load_playbook()
    merged_hints = dict(DOMAIN_HINTS)
    for domain, hints in (playbook.get("topic_clusters") or {}).items():
        merged_hints.setdefault(domain, [])
        merged_hints[domain].extend(hints or [])
    for domain, hints in merged_hints.items():
        if any(h in t for h in hints):
            return domain
    return ""


def _legacy_longevity_from_catalyst(catalyst: dict[str, Any] | None) -> dict[str, Any]:
    window = str((catalyst or {}).get("dominant_window", "")).strip().lower()
    if window in {"0-7d", "0_7d", "0to7d", "0-7"}:
        return {"1y": "high", "3y": "medium", "5y": "low", "10y": "low", "dominant_horizon": "1y"}
    if window in {"8-30d", "8_30d", "8to30d", "8-30"}:
        return {"1y": "high", "3y": "medium", "5y": "low", "10y": "low", "dominant_horizon": "1y"}
    if window in {"31-90d", "31_90d", "31to90d", "31-90"}:
        return {"1y": "medium", "3y": "high", "5y": "medium", "10y": "low", "dominant_horizon": "3y"}
    if window in {"3m+", "3m_plus", "3mplus", "3m"}:
        return {"1y": "low", "3y": "medium", "5y": "high", "10y": "high", "dominant_horizon": "5y"}
    return {"1y": "medium", "3y": "medium", "5y": "medium", "10y": "medium", "dominant_horizon": "3y"}


def _evidence_quality(pack) -> dict[str, Any]:
    coverage = pack.coverage
    count = int(coverage.get("count", 0))
    tier1 = int(coverage.get("tier1_count", 0))
    tier12 = int(coverage.get("tier12_count", 0))
    sources = len(coverage.get("sources", []) or [])
    avg = float(coverage.get("avg_evidence_score", 0.0))
    recent_365 = int(coverage.get("recent_365d_count", 0))
    recent_90 = int(coverage.get("recent_90d_count", 0))

    if count >= 8 and tier12 >= 4 and sources >= 3 and avg >= 0.75:
        grade = "A"
    elif count >= 6 and tier12 >= 3 and sources >= 3 and avg >= 0.65:
        grade = "B"
    elif count >= 4 and tier12 >= 2 and sources >= 2 and avg >= 0.55:
        grade = "C"
    else:
        grade = "D"

    return {
        "grade": grade,
        "count": count,
        "tier1_count": tier1,
        "tier12_count": tier12,
        "source_count": sources,
        "avg_evidence_score": round(avg, 3),
        "recent_365d_count": recent_365,
        "recent_90d_count": recent_90,
        "sources": coverage.get("sources", []),
        "domains": coverage.get("domains", []),
        "reason": (
            f"count={count}, tier12={tier12}, sources={sources}, avg={avg:.3f}, "
            f"recent365={recent_365}, recent90={recent_90}"
        ),
    }


def _reasoning_confidence(
    quality: dict[str, Any],
    company_impact: list[dict[str, Any]],
    evidence_pack,
    red_flags: list[str] | None = None,
    missing_data: list[str] | None = None,
) -> dict[str, Any]:
    sources = list(quality.get("sources", []) or [])
    domains = list(quality.get("domains", []) or [])
    supported = 0
    for item in company_impact or []:
        if item.get("supported_in_evidence"):
            supported += 1
    support_ratio = supported / max(1, len(company_impact or []))
    coverage = float(quality.get("count", 0)) / 8.0
    tier_strength = float(quality.get("tier12_count", 0)) / 4.0
    source_diversity = min(1.0, len(sources) / 4.0)
    domain_diversity = min(1.0, len(domains) / 3.0)
    evidence_strength = float(quality.get("avg_evidence_score", 0.0))
    recency_strength = min(1.0, float(quality.get("recent_365d_count", 0)) / 3.0)
    recent_freshness = min(1.0, float(quality.get("recent_90d_count", 0)) / 2.0)
    red_penalty = min(0.30, 0.05 * len(red_flags or []) + 0.03 * len(missing_data or []))
    reasoning_confidence = (
        0.28 * evidence_strength
        + 0.18 * source_diversity
        + 0.14 * domain_diversity
        + 0.18 * min(1.0, coverage)
        + 0.12 * min(1.0, tier_strength)
        + 0.10 * support_ratio
        + 0.06 * recency_strength
        + 0.04 * recent_freshness
        - red_penalty
    )
    reasoning_confidence = max(0.0, min(1.0, reasoning_confidence))
    return {
        "reasoning_confidence": round(reasoning_confidence, 3),
        "breakdown": {
            "evidence_strength": round(evidence_strength, 3),
            "source_diversity": round(source_diversity, 3),
            "domain_diversity": round(domain_diversity, 3),
            "coverage": round(min(1.0, coverage), 3),
            "tier_strength": round(min(1.0, tier_strength), 3),
            "support_ratio": round(support_ratio, 3),
            "recency_strength": round(recency_strength, 3),
            "recent_freshness": round(recent_freshness, 3),
            "red_penalty": round(red_penalty, 3),
        },
    }


def _candidate_companies(topic: str, company_hint: str = "", evidence_pack=None) -> list[dict[str, str]]:
    topic_l = _norm(topic)
    hinted = _norm(company_hint)
    playbook = _load_playbook()
    candidates: list[dict[str, str]] = []

    def add(code: str, company: str, reason: str) -> None:
        if any(c["code"] == code for c in candidates):
            return
        candidates.append({"code": code, "company": company, "reason": reason})

    if any(k in topic_l for k in DOMAIN_HINTS["hbm"]):
        add("000660", "SK hynix", "HBM supply leadership and memory execution")
        add("005930", "Samsung Electronics", "HBM competition and process transition impact")
        add("MU", "Micron", "Direct competitor in HBM and DRAM market structure")
        add("NVDA", "NVIDIA", "Directly linked to AI server memory demand")
    if any(k in topic_l for k in DOMAIN_HINTS["litho"]):
        add("ASML", "ASML", "Core supplier of EUV and High-NA lithography systems")
        add("005930", "Samsung Electronics", "Direct impact on leading-edge process investment")
        add("TSM", "TSMC", "Direct tie to advanced logic manufacturing adoption")
        add("INTC", "Intel", "Execution risk and opportunity in advanced node transition")
    if any(k in topic_l for k in DOMAIN_HINTS["packaging"]):
        add("042700", "Hanmi Semiconductor", "Direct beneficiary of packaging equipment and process shifts")
        add("000660", "SK hynix", "Directly tied to HBM packaging structure and scaling")
        add("005930", "Samsung Electronics", "Packaging capex and transition cost impact")
        add("TSM", "TSMC", "Direct exposure to CoWoS and advanced packaging ecosystem")
    if "equipment" in topic_l:
        add("ASML", "ASML", "Direct leverage to lithography equipment demand")
        add("AMAT", "Applied Materials", "Direct leverage to process equipment demand")
        add("LRCX", "Lam Research", "Direct leverage to etch and deposition demand")

    if hinted:
        company_map = playbook.get("company_watchlist") or {}
        for code, company in COMPANY_CANDIDATES:
            if code.lower() in hinted or company.lower() in hinted:
                add(code, company, "User-provided company hint")
        for code, company in company_map.items():
            code_str = str(code)
            if code_str.lower() in hinted or str(company).lower() in hinted:
                add(code_str, str(company), "User-provided company hint")

    if evidence_pack is not None:
        for item in evidence_pack.items:
            if item.company:
                for code, company in COMPANY_CANDIDATES:
                    if code == item.company or company.lower() in item.company.lower():
                        add(code, company, f"Evidence pack repeatedly mentions {company}")

    return candidates[:6]
def _company_supported_in_evidence(code: str, company: str, evidence_pack) -> bool:
    code_l = _norm(code)
    company_l = _norm(company)
    for item in getattr(evidence_pack, "items", []) or []:
        hay = _norm(
            " ".join(
                [
                    str(getattr(item, "title", "")),
                    str(getattr(item, "text", "")),
                    str(getattr(item, "source", "")),
                    str(getattr(item, "company", "")),
                    str(getattr(item, "domain", "")),
                ]
            )
        )
        if code_l and code_l in hay:
            return True
        if company_l and company_l in hay:
            return True
    return False


def _refine_company_impact(
    topic: str,
    company_hint: str,
    domain_hint: str,
    raw_company_impact: list[dict[str, Any]],
    evidence_pack,
    candidates: list[dict[str, str]],
) -> list[dict[str, Any]]:
    candidate_map = {c["code"].upper(): c for c in candidates if c.get("code")}
    topic_l = _norm(topic)
    domain_l = _norm(domain_hint)
    refined: list[dict[str, Any]] = []

    def add_item(item: dict[str, Any]) -> None:
        code = str(item.get("code", "") or "").strip().upper()
        company = str(item.get("company", "") or "").strip()
        stance = str(item.get("stance", "neutral") or "neutral").strip().lower()
        reason = str(item.get("reason", "") or "").strip()
        confidence = float(item.get("confidence", 0.0) or 0.0)
        supported = _company_supported_in_evidence(code, company, evidence_pack)
        candidate = candidate_map.get(code)

        if stance not in {"benefit", "threat", "neutral"}:
            stance = "neutral"

        # Strict filter: keep only if evidence supports it or confidence is clearly high.
        if not supported and confidence < 0.70 and code not in candidate_map:
            return
        if confidence < 0.45 and not supported:
            return

        if not company and candidate:
            company = candidate["company"]

        if not company:
            company = code

        if not reason:
            reason = candidate["reason"] if candidate else "Evidence-linked impact candidate"

        if not supported:
            confidence = min(confidence, 0.65)

        refined.append(
            {
                "company": company,
                "code": code,
                "stance": stance,
                "confidence": round(confidence, 3),
                "reason": reason,
                "supported_in_evidence": supported,
            }
        )

    for item in raw_company_impact or []:
        if isinstance(item, dict):
            add_item(item)

    # Conservative fallback if the model was too sparse.
    if not refined:
        for c in candidates:
            code = c["code"]
            company = c["company"]
            supported = _company_supported_in_evidence(code, company, evidence_pack)
            if not supported and code.upper() not in {"000660", "005930", "NVDA", "TSM", "ASML", "AMAT", "LRCX", "KLAC", "MU", "INTC", "042700"}:
                continue
            refined.append(
                {
                    "company": company,
                    "code": code,
                    "stance": "benefit" if supported else "neutral",
                    "confidence": 0.70 if supported else 0.50,
                    "reason": c["reason"],
                    "supported_in_evidence": supported,
                }
            )

    # De-duplicate and keep strongest items only.
    dedup: dict[str, dict[str, Any]] = {}
    for item in refined:
        code = str(item.get("code", "")).upper()
        prev = dedup.get(code)
        if prev is None or float(item.get("confidence", 0.0)) > float(prev.get("confidence", 0.0)):
            dedup[code] = item

    items = list(dedup.values())
    items.sort(key=lambda x: (x.get("supported_in_evidence", False), float(x.get("confidence", 0.0))), reverse=True)
    return items[:6]


def _longevity_prompt(topic: str, domain_hint: str, evidence_pack, evidence_quality: dict[str, Any]) -> dict[str, Any]:
    playbook = _load_playbook()
    system = make_semiconductor_system_prompt(
        mode="technology_potential_assessment",
        use_rag=True,
        output_format="Return JSON only, with no markdown or commentary.",
    )

    user = f"""[Topic]
{topic}

[Domain hint]
{domain_hint or "auto"}

[Evidence quality]
{json.dumps(evidence_quality, ensure_ascii=False)}

[Good answer criteria]
{json.dumps(playbook.get("good_answer_criteria", {}), ensure_ascii=False)}

[Axis spec]
{json.dumps(playbook.get("axis_spec", {}), ensure_ascii=False)}

[Questions]
{json.dumps(playbook.get("questions", []), ensure_ascii=False)}

[Evidence]
{evidence_pack.context_text(max_chars=5000)}

[Output schema]
{{
  "catalyst_imminence": {{
    "0_7d": "high|medium|low",
    "8_30d": "high|medium|low",
    "31_90d": "high|medium|low",
    "3m_plus": "high|medium|low",
    "dominant_window": "0-7d|8-30d|31-90d|3m+",
    "reason": "..."
  }},
  "bottleneck": {{
    "is_bottleneck": true,
    "importance": "low|medium|high|critical",
    "substitutability": "easy|moderate|hard",
    "supply_constraint": "low|medium|high",
    "reason": "..."
  }},
  "company_impact": [
    {{
      "company": "ASML",
      "code": "ASML",
      "stance": "benefit|threat|neutral",
      "confidence": 0.0,
      "reason": "..."
    }}
  ],
  "evidence_quality": {{
    "grade": "A|B|C|D",
    "reason": "..."
  }},
  "novelty": {{
    "surprise_level": "high|medium|low",
    "market_awareness": "new|partially_known|already_known",
    "reason": "..."
  }},
  "revenue_linkage": {{
    "link_strength": "direct|moderate|weak",
    "time_to_monetize": "0-7d|8-30d|31-90d|3m+|unknown",
    "reason": "..."
  }},
  "market_transmission_speed": {{
    "speed": "fast|medium|slow",
    "reason": "..."
  }},
  "longevity": {{
    "1y": "high|medium|low",
    "3y": "high|medium|low",
    "5y": "high|medium|low",
    "10y": "high|medium|low",
    "dominant_horizon": "1y|3y|5y|10y+",
    "reason": "legacy view for compatibility"
  }},
  "overall_thesis": "...",
  "red_flags": ["..."],
  "missing_data": ["..."],
  "recommendation": "investable|watchlist|too_early"
}}"""

    data = call_llm_json(system, user)
    return data if isinstance(data, dict) else {}


def assess_technology_potential(
    topic: str,
    *,
    company_hint: str = "",
    domain_hint: str = "",
    top_k: int = 8,
) -> TechnologyPotentialAssessment:
    """
    Evaluate semiconductor technology potential around near-term catalysts,
    bottlenecks, company impact, evidence quality, novelty, and monetization.
    """
    if not domain_hint:
        domain_hint = _infer_domain(topic)

    playbook = _load_playbook()
    queries = [
        topic,
        f"{topic} official company investor relations",
        f"{topic} paper standard roadmap",
    ]
    evidence_pack = build_evidence_pack(
        topic,
        company=company_hint or None,
        domain=domain_hint or None,
        top_k=top_k,
    )
    quality = _evidence_quality(evidence_pack)
    companies = _candidate_companies(topic, company_hint=company_hint, evidence_pack=evidence_pack)

    raw = _longevity_prompt(topic, domain_hint, evidence_pack, quality)

    catalyst_imminence = raw.get("catalyst_imminence", {}) if isinstance(raw, dict) else {}
    longevity = raw.get("longevity", {}) if isinstance(raw, dict) else {}
    if not longevity:
        longevity = _legacy_longevity_from_catalyst(catalyst_imminence)
    bottleneck = raw.get("bottleneck", {}) if isinstance(raw, dict) else {}
    evidence_quality = raw.get("evidence_quality", {}) if isinstance(raw, dict) else {}
    novelty = raw.get("novelty", {}) if isinstance(raw, dict) else {}
    revenue_linkage = raw.get("revenue_linkage", {}) if isinstance(raw, dict) else {}
    market_transmission_speed = raw.get("market_transmission_speed", {}) if isinstance(raw, dict) else {}
    raw_company_impact = raw.get("company_impact", []) if isinstance(raw, dict) else []
    company_impact = _refine_company_impact(
        topic,
        company_hint,
        domain_hint,
        raw_company_impact if isinstance(raw_company_impact, list) else [],
        evidence_pack,
        companies,
    )
    red_flags = list(raw.get("red_flags", []) or [])
    missing_data = list(raw.get("missing_data", []) or [])
    reasoning = _reasoning_confidence(quality, company_impact, evidence_pack, red_flags=red_flags, missing_data=missing_data)
    supporting_evidence = [
        {
            "title": item.title,
            "source": item.source,
            "source_type": item.source_type,
            "domain": item.domain,
            "company": item.company,
            "year": item.year,
            "evidence_score": round(item.evidence_score, 4),
            "published_at": item.published_at,
            "reasons": item.reasons,
        }
        for item in evidence_pack.items[:5]
    ]
    normalized_event = normalize_event_from_evidence(
        query=topic,
        items=[asdict(item) for item in evidence_pack.items],
        company=company_hint,
        domain=domain_hint,
        recent_90d_count=int(quality.get("recent_90d_count", 0) or 0),
    )

    return TechnologyPotentialAssessment(
        topic=topic,
        as_of=_now_iso(),
        company_hint=company_hint,
        domain_hint=domain_hint,
        catalyst_imminence=catalyst_imminence,
        longevity=longevity,
        bottleneck=bottleneck,
        company_impact=company_impact,
        evidence_quality={
            **quality,
            **evidence_quality,
        },
        novelty=novelty,
        revenue_linkage=revenue_linkage,
        market_transmission_speed=market_transmission_speed,
        overall_thesis=str(raw.get("overall_thesis", "")).strip(),
        red_flags=red_flags,
        missing_data=missing_data,
        recommendation=str(raw.get("recommendation", "watchlist")).strip(),
        confidence=float(quality.get("avg_evidence_score", 0.0)),
        reasoning_confidence=float(reasoning.get("reasoning_confidence", 0.0)),
        reasoning_breakdown=reasoning.get("breakdown", {}),
        supporting_evidence=supporting_evidence,
        evidence_pack={
            "as_of": _now_iso(),
            "coverage": evidence_pack.coverage,
            "items": [asdict(item) for item in evidence_pack.items],
            "normalized_event": normalized_event,
            "queries": queries,
            "playbook": {
                "evidence_priority": playbook.get("evidence_priority", []),
                "event_examples": playbook.get("event_examples", []),
            },
        },
    )


def render_technology_potential_markdown(assessment: TechnologyPotentialAssessment) -> str:
    catalyst = assessment.catalyst_imminence or {}
    longevity = assessment.longevity or {}
    bottleneck = assessment.bottleneck or {}
    evidence = assessment.evidence_quality or {}
    novelty = assessment.novelty or {}
    revenue = assessment.revenue_linkage or {}
    transmission = assessment.market_transmission_speed or {}

    company_lines = "\n".join(
        f"- **{item.get('company', '')}** ({item.get('code', '')}) - {item.get('stance', '')}: {item.get('reason', '')}"
        for item in assessment.company_impact
    ) or "- ?놁쓬"
    red_flags = "\n".join(f"- {x}" for x in assessment.red_flags) or "- ?놁쓬"
    missing = "\n".join(f"- {x}" for x in assessment.missing_data) or "- ?놁쓬"
    support = "\n".join(
        f"- {item.get('source_type', '')}/{item.get('source', '')}: {item.get('title', '')} (score={float(item.get('evidence_score', 0.0)):.3f})"
        for item in assessment.supporting_evidence
    ) or "- ?놁쓬"
    normalized_event = ((assessment.evidence_pack or {}).get("normalized_event") or {}) if isinstance(assessment.evidence_pack, dict) else {}
    breakdown = assessment.reasoning_breakdown or {}

    return f"""# Technology Potential Assessment

**Topic**: {assessment.topic}
**As of**: {assessment.as_of or 'unknown'}
**Domain hint**: {assessment.domain_hint or 'auto'}
**Recommendation**: {assessment.recommendation}
**Confidence**: {assessment.confidence:.2f}
**Reasoning confidence**: {assessment.reasoning_confidence:.2f}

## 0. Normalized Technology Event

| Item | Value |
| --- | --- |
| Event type | {normalized_event.get('event_type', '')} |
| Technology | {normalized_event.get('technology', '')} |
| Event date | {normalized_event.get('event_date', '')} |
| Company | {normalized_event.get('related_company', '')} |
| Domain | {normalized_event.get('related_domain', '')} |
| Catalyst hint | {normalized_event.get('catalyst_imminence', '')} |
| Revenue linkage hint | {normalized_event.get('revenue_linkage', '')} |
| Transmission speed hint | {normalized_event.get('market_transmission_speed', '')} |

{normalized_event.get('headline', '')}

{normalized_event.get('summary', '')}

## 1. Catalyst Imminence

| Horizon | View |
| --- | --- |
| 0-7d | {catalyst.get('0_7d', '')} |
| 8-30d | {catalyst.get('8_30d', '')} |
| 31-90d | {catalyst.get('31_90d', '')} |
| 3m+ | {catalyst.get('3m_plus', '')} |
| Dominant | {catalyst.get('dominant_window', '')} |

{catalyst.get('reason', '')}

## 2. Bottleneck Importance

| Item | Value |
| --- | --- |
| Is bottleneck | {bottleneck.get('is_bottleneck', '')} |
| Importance | {bottleneck.get('importance', '')} |
| Substitutability | {bottleneck.get('substitutability', '')} |
| Supply constraint | {bottleneck.get('supply_constraint', '')} |

{bottleneck.get('reason', '')}

## 3. Company Impact

{company_lines}

## 4. Evidence Quality

| Item | Value |
| --- | --- |
| Grade | {evidence.get('grade', '')} |
| Count | {evidence.get('count', '')} |
| Tier1+2 | {evidence.get('tier12_count', '')} |
| Source count | {evidence.get('source_count', '')} |
| Avg score | {evidence.get('avg_evidence_score', '')} |

{evidence.get('reason', '')}

## 5. Novelty / Surprise

| Item | Value |
| --- | --- |
| Surprise level | {novelty.get('surprise_level', '')} |
| Market awareness | {novelty.get('market_awareness', '')} |

{novelty.get('reason', '')}

## 6. Revenue Linkage

| Item | Value |
| --- | --- |
| Link strength | {revenue.get('link_strength', '')} |
| Time to monetize | {revenue.get('time_to_monetize', '')} |

{revenue.get('reason', '')}

## 7. Market Transmission Speed

| Item | Value |
| --- | --- |
| Speed | {transmission.get('speed', '')} |

{transmission.get('reason', '')}

## Legacy Longevity View

| Horizon | View |
| --- | --- |
| 1y | {longevity.get('1y', '')} |
| 3y | {longevity.get('3y', '')} |
| 5y | {longevity.get('5y', '')} |
| 10y | {longevity.get('10y', '')} |
| Dominant | {longevity.get('dominant_horizon', '')} |

{longevity.get('reason', '')}

## Thesis

{assessment.overall_thesis}

## Reasoning Breakdown

| Item | Value |
| --- | --- |
| evidence_strength | {breakdown.get('evidence_strength', '')} |
| source_diversity | {breakdown.get('source_diversity', '')} |
| domain_diversity | {breakdown.get('domain_diversity', '')} |
| coverage | {breakdown.get('coverage', '')} |
| tier_strength | {breakdown.get('tier_strength', '')} |
| support_ratio | {breakdown.get('support_ratio', '')} |
| red_penalty | {breakdown.get('red_penalty', '')} |

## Supporting Evidence

{support}

## Red Flags

{red_flags}

## Missing Data

{missing}
"""


def assess_technology_potential_without_rag(
    topic: str,
    *,
    company_hint: str = "",
    domain_hint: str = "",
) -> dict[str, Any]:
    """
    Baseline-only assessment with no database context.

    This is used for benchmark comparisons against the RAG-backed version.
    """
    if not domain_hint:
        domain_hint = _infer_domain(topic)

    playbook = _load_playbook()
    system = make_semiconductor_system_prompt(
        mode="technology_potential_baseline_assessment",
        use_rag=False,
        output_format="Return JSON only, with no markdown or commentary.",
    )

    user = f"""[Topic]
{topic}

[Company hint]
{company_hint or "none"}

[Domain hint]
{domain_hint or "auto"}

[Good answer criteria]
{json.dumps(playbook.get("good_answer_criteria", {}), ensure_ascii=False)}

[Axis spec]
{json.dumps(playbook.get("axis_spec", {}), ensure_ascii=False)}

[Question set]
{json.dumps(playbook.get("questions", []), ensure_ascii=False)}

[Output schema]
{{
  "catalyst_imminence": {{
    "0_7d": "high|medium|low",
    "8_30d": "high|medium|low",
    "31_90d": "high|medium|low",
    "3m_plus": "high|medium|low",
    "dominant_window": "0-7d|8-30d|31-90d|3m+",
    "reason": "..."
  }},
  "bottleneck": {{
    "is_bottleneck": true,
    "importance": "low|medium|high|critical",
    "substitutability": "easy|moderate|hard",
    "supply_constraint": "low|medium|high",
    "reason": "..."
  }},
  "company_impact": [
    {{
      "company": "ASML",
      "code": "ASML",
      "stance": "benefit|threat|neutral",
      "confidence": 0.0,
      "reason": "..."
    }}
  ],
  "evidence_quality": {{
    "grade": "A|B|C|D",
    "reason": "..."
  }},
  "novelty": {{
    "surprise_level": "high|medium|low",
    "market_awareness": "new|partially_known|already_known",
    "reason": "..."
  }},
  "revenue_linkage": {{
    "link_strength": "direct|moderate|weak",
    "time_to_monetize": "0-7d|8-30d|31-90d|3m+|unknown",
    "reason": "..."
  }},
  "market_transmission_speed": {{
    "speed": "fast|medium|slow",
    "reason": "..."
  }},
  "longevity": {{
    "1y": "high|medium|low",
    "3y": "high|medium|low",
    "5y": "high|medium|low",
    "10y": "high|medium|low",
    "dominant_horizon": "1y|3y|5y|10y+",
    "reason": "legacy view for compatibility"
  }},
  "overall_thesis": "...",
  "red_flags": ["..."],
  "missing_data": ["..."],
  "recommendation": "investable|watchlist|too_early"
}}"""

    data = call_llm_json(system, user)
    return data if isinstance(data, dict) else {}


def compare_technology_potential_with_baseline(
    topic: str,
    *,
    company_hint: str = "",
    domain_hint: str = "",
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Run both baseline LLM and RAG+LLM assessments for comparison.
    This should stay within the LLM call budget if used on a small case set.
    """
    baseline = assess_technology_potential_without_rag(
        topic,
        company_hint=company_hint,
        domain_hint=domain_hint,
    )
    rag = assess_technology_potential(
        topic,
        company_hint=company_hint,
        domain_hint=domain_hint,
        top_k=top_k,
    )
    return {
        "topic": topic,
        "company_hint": company_hint,
        "domain_hint": domain_hint,
        "baseline": baseline,
        "rag": asdict(rag),
        "delta": {
            "evidence_grade": f"{baseline.get('evidence_quality', {}).get('grade', '?')} -> {rag.evidence_quality.get('grade', '?')}",
            "confidence": round(float(rag.confidence) - float(baseline.get("confidence", 0.0)), 4) if isinstance(baseline, dict) else float(rag.confidence),
        },
    }

