from __future__ import annotations

import re
from typing import Any


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def classify_event_type(text: str, domain: str = "") -> str:
    t = norm_text(text)
    if any(k in t for k in ["sample shipment", "samples", "샘플", "qualification sample", "sampled to customer"]):
        return "sample_shipment"
    if any(k in t for k in ["mass production", "양산", "production start", "ramp-up", "ramp up"]):
        return "mass_production_timing"
    if any(k in t for k in ["yield", "수율"]):
        return "yield_update"
    if any(k in t for k in ["capacity", "증설", "expansion", "fab", "new line", "production hub"]):
        return "capacity_expansion"
    if any(k in t for k in ["order", "수주", "equipment order", "booking", "backlog"]):
        return "equipment_order"
    if any(k in t for k in ["adoption", "customer", "design win", "qualification", "채택"]):
        return "customer_adoption"
    if any(k in t for k in ["guidance", "earnings", "revenue", "실적", "sales"]):
        return "earnings_guidance"
    if any(k in t for k in ["jedec", "irds", "standard", "roadmap", "spec"]):
        return "standard_update"
    if domain in {"hbm", "packaging", "litho", "logic", "nand", "dram"}:
        return "technology_breakthrough"
    return "technology_update"


def infer_technology_label(text: str, domain: str = "") -> str:
    t = norm_text(text)
    patterns = [
        ("HBM4E", ["hbm4e"]),
        ("HBM4", ["hbm4"]),
        ("HBM3E", ["hbm3e"]),
        ("HBM", ["high bandwidth memory", "hbm"]),
        ("High-NA EUV", ["high-na euv", "high na euv", "high-na"]),
        ("EUV", ["euv"]),
        ("Hybrid Bonding", ["hybrid bonding"]),
        ("CoWoS", ["cowos"]),
        ("TSV", ["tsv", "through-silicon via"]),
        ("GAA", ["gate-all-around", "gaa", "nanosheet"]),
        ("FinFET", ["finfet"]),
        ("V-NAND", ["v-nand", "3d nand"]),
        ("CXL", ["cxl", "compute express link"]),
        ("PIM", ["processing-in-memory", "processing in memory", "pim"]),
        ("SiC", ["sic", "silicon carbide"]),
        ("GaN", ["gan", "gallium nitride"]),
    ]
    for label, hints in patterns:
        if any(h in t for h in hints):
            return label
    fallback = {
        "hbm": "HBM",
        "packaging": "Advanced Packaging",
        "litho": "Lithography",
        "logic": "Logic Process",
        "nand": "NAND",
        "dram": "DRAM",
        "standards": "Standards/Roadmap",
    }
    return fallback.get(domain or "", "Semiconductor Technology")


def classify_technology_category(text: str, domain: str = "") -> str:
    t = norm_text(text)
    if any(k in t for k in ["hbm", "dram", "ddr", "lpddr", "nand", "v-nand", "flash", "cxl", "pim"]):
        return "memory_architecture"
    if any(k in t for k in ["euv", "high-na", "lithography", "photoresist", "pellicle"]):
        return "lithography_process"
    if any(k in t for k in ["gaa", "finfet", "nanosheet", "backside power", "bspdn", "18a", "2nm", "3nm"]):
        return "logic_process"
    if any(k in t for k in ["hybrid bonding", "cowos", "tsv", "interposer", "chiplet", "osat", "packaging"]):
        return "advanced_packaging"
    if any(k in t for k in ["yield", "inspection", "metrology", "reliability", "defect", "tem", "fib"]):
        return "yield_reliability"
    if any(k in t for k in ["equipment", "etch", "deposition", "ald", "ale", "cmp", "tool"]):
        return "equipment_process_tool"
    if any(k in t for k in ["jedec", "irds", "standard", "roadmap", "spec"]):
        return "standards_roadmap"
    fallback = {
        "hbm": "memory_architecture",
        "dram": "memory_architecture",
        "nand": "memory_architecture",
        "litho": "lithography_process",
        "logic": "logic_process",
        "packaging": "advanced_packaging",
        "standards": "standards_roadmap",
        "reliability": "yield_reliability",
        "equipment": "equipment_process_tool",
    }
    return fallback.get(domain or "", "general_semiconductor")


def event_to_short_horizon_hints(event_type: str, source_type: str, source: str) -> dict[str, Any]:
    if event_type == "sample_shipment":
        return {
            "catalyst_imminence": "0-7d",
            "revenue_linkage": "moderate",
            "market_transmission_speed": "fast",
            "detail_reason": "Sample shipment improves customer qualification visibility and signals near-term execution.",
        }
    if event_type == "mass_production_timing":
        return {
            "catalyst_imminence": "8-30d",
            "revenue_linkage": "direct",
            "market_transmission_speed": "fast",
            "detail_reason": "Mass-production timing usually changes revenue timing and investor expectations directly.",
        }
    if event_type == "yield_update":
        return {
            "catalyst_imminence": "8-30d",
            "revenue_linkage": "moderate",
            "market_transmission_speed": "medium",
            "detail_reason": "Yield changes matter when they affect ramp quality, cost, and customer readiness rather than immediate shipments.",
        }
    if event_type == "capacity_expansion":
        return {
            "catalyst_imminence": "8-30d",
            "revenue_linkage": "moderate",
            "market_transmission_speed": "medium",
            "detail_reason": "Capacity expansion matters if it changes supply visibility, but monetization is usually not immediate.",
        }
    if event_type == "customer_adoption":
        return {
            "catalyst_imminence": "0-7d",
            "revenue_linkage": "direct",
            "market_transmission_speed": "fast",
            "detail_reason": "Customer adoption is one of the cleanest paths from technology progress to revenue expectations.",
        }
    if event_type in {"equipment_order", "earnings_guidance"}:
        return {
            "catalyst_imminence": "0-7d",
            "revenue_linkage": "direct",
            "market_transmission_speed": "fast",
            "detail_reason": "Orders and guidance changes are usually quickly transmitted to estimates and price action.",
        }
    if event_type == "standard_update":
        return {
            "catalyst_imminence": "31-90d",
            "revenue_linkage": "weak",
            "market_transmission_speed": "slow",
            "detail_reason": "Standards shape medium-term positioning, but they usually do not translate into immediate revenue.",
        }
    if source_type == "company_official":
        return {
            "catalyst_imminence": "8-30d",
            "revenue_linkage": "moderate",
            "market_transmission_speed": "medium",
            "detail_reason": "Official technology updates matter, but their short-horizon price effect depends on customer, timing, and execution linkage.",
        }
    return {
        "catalyst_imminence": "31-90d",
        "revenue_linkage": "weak",
        "market_transmission_speed": "slow",
        "detail_reason": "Generic technical information is often structurally useful but slow to affect short-horizon price action.",
    }


def normalize_event_from_evidence(
    *,
    query: str,
    items: list[dict[str, Any]],
    company: str = "",
    domain: str = "",
    existing_event: dict[str, Any] | None = None,
    recent_90d_count: int = 0,
) -> dict[str, Any]:
    if existing_event:
        title = str(existing_event.get("title") or "").strip()
        summary = str(existing_event.get("summary") or "").strip()
        related_domain = str(existing_event.get("related_domain") or domain or "").strip()
        raw_event_type = str(existing_event.get("event_type") or "").strip()
        if raw_event_type in {"paper", "company_official", "rss_news", "tech_blog", "conference_metadata", "event_candidate"}:
            raw_event_type = ""
        event_type = raw_event_type or classify_event_type(f"{title} {summary}", related_domain)
        hints = event_to_short_horizon_hints(event_type, "event_candidate", str(existing_event.get("source") or ""))
        return {
            "event_id": existing_event.get("event_id", ""),
            "event_date": existing_event.get("event_date", ""),
            "event_type": event_type,
            "technology": infer_technology_label(f"{title} {summary}", related_domain),
            "technology_category": classify_technology_category(f"{title} {summary}", related_domain),
            "headline": title,
            "summary": summary,
            "related_company": existing_event.get("related_company", company),
            "related_domain": related_domain,
            "source": existing_event.get("source", ""),
            "source_tier": existing_event.get("source_tier", 0),
            "confidence": float(existing_event.get("confidence", 0.0) or 0.0),
            "novelty_hint": "high" if recent_90d_count > 0 else "medium",
            "evidence_doc_uids": [existing_event.get("evidence_doc_uid", "")] if existing_event.get("evidence_doc_uid") else [],
            **hints,
        }

    if not items:
        return {}

    top = items[0]
    payload = dict(top.get("payload") or {})
    title = str(top.get("title") or "").strip()
    summary = str(top.get("text") or "")[:1200]
    related_domain = str(domain or top.get("domain") or payload.get("domain") or "").strip()
    combined = f"{title} {summary}"
    event_type = classify_event_type(combined, related_domain)
    hints = event_to_short_horizon_hints(event_type, str(top.get("source_type") or ""), str(top.get("source") or ""))
    published_at = str(top.get("published_at") or payload.get("published_at") or payload.get("collected_at") or "")
    related_company = str(company or top.get("company") or payload.get("company_code") or payload.get("company") or "").strip()
    evidence_doc_uids: list[str] = []
    for item in items:
        uid = str((item.get("payload") or {}).get("doc_uid") or "").strip()
        if uid:
            evidence_doc_uids.append(uid)
    return {
        "event_id": "",
        "event_date": published_at,
        "event_type": event_type,
        "technology": infer_technology_label(combined, related_domain),
        "technology_category": classify_technology_category(combined, related_domain),
        "headline": title,
        "summary": summary,
        "related_company": related_company,
        "related_domain": related_domain,
        "source": top.get("source", ""),
        "source_tier": 1 if str(top.get("source_type") or "") == "company_official" else 2 if str(top.get("source_type") or "") == "paper" else 3,
        "confidence": round(float(top.get("evidence_score", 0.0) or 0.0), 3),
        "novelty_hint": "high" if recent_90d_count > 0 else "medium",
        "evidence_doc_uids": list(dict.fromkeys(evidence_doc_uids)),
        **hints,
    }
