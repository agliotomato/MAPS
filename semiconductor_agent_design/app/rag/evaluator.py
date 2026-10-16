"""
RAG quality evaluator.

This is intentionally strict: the agent should only claim a strong RAG
advantage when it retrieves high-trust, domain-relevant evidence.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from app.rag.evidence_builder import build_evidence_pack

logger = logging.getLogger(__name__)


DEFAULT_CASES = [
    {
        "id": "hbm_hybrid_bonding",
        "query": "HBM4 hybrid bonding Cu-Cu bonding SK hynix Samsung Hanmi",
        "company": "000660",
        "domain": "packaging",
        "must_have_terms": ["hbm", "bonding"],
    },
    {
        "id": "euv_high_na",
        "query": "High-NA EUV lithography ASML Samsung foundry",
        "company": "ASML",
        "domain": "litho",
        "must_have_terms": ["euv"],
    },
    {
        "id": "ai_hbm_demand",
        "query": "NVIDIA AI data center demand HBM memory SK hynix",
        "company": "NVDA",
        "domain": "hbm",
        "must_have_terms": ["ai", "hbm"],
    },
    {
        "id": "tsmc_cowos_capacity",
        "query": "TSMC CoWoS advanced packaging AI accelerator capacity",
        "company": "TSM",
        "domain": "packaging",
        "must_have_terms": ["cowos", "packaging"],
    },
]

DIFFICULTY_REQUIREMENTS = {
    "easy": {"count": 4, "tier12": 2, "avg": 0.58, "sources": 3},
    "medium": {"count": 5, "tier12": 2, "avg": 0.62, "sources": 3},
    "hard": {"count": 6, "tier12": 3, "avg": 0.66, "sources": 4},
}


def _load_cases(path: str | None = None) -> list[dict[str, Any]]:
    if not path:
        default_path = Path(__file__).resolve().parents[2] / "data" / "rag_eval_cases.yaml"
        if default_path.exists():
            path = str(default_path)
        else:
            return DEFAULT_CASES
    p = Path(path)
    if not p.exists():
        return DEFAULT_CASES
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if isinstance(data, dict):
        return data.get("cases", DEFAULT_CASES)
    if isinstance(data, list):
        return data
    return DEFAULT_CASES


def _case_score(case: dict[str, Any]) -> dict[str, Any]:
    prev_mode = os.getenv("RAG_SEARCH_MODE")
    os.environ["RAG_SEARCH_MODE"] = "lexical_payload"
    try:
        pack = build_evidence_pack(
            case["query"],
            company=case.get("company"),
            domain=case.get("domain"),
            top_k=int(case.get("top_k", 8)),
        )
    finally:
        if prev_mode is None:
            os.environ.pop("RAG_SEARCH_MODE", None)
        else:
            os.environ["RAG_SEARCH_MODE"] = prev_mode
    context = pack.context_text(max_chars=8000).lower()
    must_terms = [str(x).lower() for x in case.get("must_have_terms", [])]
    term_hits = {term: (term in context) for term in must_terms}

    coverage = pack.coverage
    avg = float(coverage.get("avg_evidence_score", 0.0))
    tier12 = int(coverage.get("tier12_count", 0))
    sources = coverage.get("sources", [])
    pass_reasons = []
    fail_reasons = []

    difficulty = str(case.get("difficulty", "medium")).lower()
    req = DIFFICULTY_REQUIREMENTS.get(difficulty, DIFFICULTY_REQUIREMENTS["medium"])

    if coverage.get("count", 0) >= req["count"]:
        pass_reasons.append("enough_evidence")
    else:
        fail_reasons.append("too_few_evidence_items")
    if tier12 >= req["tier12"]:
        pass_reasons.append("has_high_trust_sources")
    else:
        fail_reasons.append("not_enough_tier1_or_tier2_sources")
    if avg >= req["avg"]:
        pass_reasons.append("strong_avg_evidence_score")
    else:
        fail_reasons.append("weak_avg_evidence_score")
    if len(sources) >= req["sources"]:
        pass_reasons.append("source_diversity")
    else:
        fail_reasons.append("low_source_diversity")
    missing_terms = [term for term, ok in term_hits.items() if not ok]
    if missing_terms:
        fail_reasons.append(f"missing_terms={','.join(missing_terms)}")
    else:
        pass_reasons.append("required_terms_present")

    passed = not fail_reasons
    return {
        "id": case.get("id", case["query"]),
        "query": case["query"],
        "difficulty": difficulty,
        "passed": passed,
        "coverage": coverage,
        "term_hits": term_hits,
        "pass_reasons": pass_reasons,
        "fail_reasons": fail_reasons,
        "top_evidence": [
            {
                "title": item.title,
                "source": item.source,
                "domain": item.domain,
                "year": item.year,
                "evidence_score": round(item.evidence_score, 4),
                "reasons": item.reasons,
            }
            for item in pack.items[:5]
        ],
    }


def evaluate_rag(cases_path: str | None = None, limit: int | None = None) -> dict[str, Any]:
    cases = _load_cases(cases_path)
    if limit is not None:
        cases = cases[: max(1, limit)]
    results = []
    for case in cases:
        try:
            results.append(_case_score(case))
        except Exception as e:
            logger.exception("RAG eval case failed: %s", case.get("id", case.get("query")))
            results.append({
                "id": case.get("id", case.get("query")),
                "query": case.get("query"),
                "passed": False,
                "fail_reasons": [str(e)],
            })
    passed = sum(1 for r in results if r.get("passed"))
    return {
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }


def print_eval_report(cases_path: str | None = None, limit: int | None = None) -> None:
    report = evaluate_rag(cases_path, limit=limit)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    Path("rag_eval_report.json").write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
