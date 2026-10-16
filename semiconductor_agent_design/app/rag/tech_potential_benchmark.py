from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from app.agent.llm import remaining_call_budget
from app.agent.tech_potential import (
    assess_technology_potential,
    assess_technology_potential_without_rag,
)


GRADE_SCORE = {"A": 4, "B": 3, "C": 2, "D": 1}

_CASE_PATH = Path(__file__).resolve().parents[2] / "data" / "tech_potential_eval_cases.yaml"


def _load_cases() -> list[dict[str, Any]]:
    if not _CASE_PATH.exists():
        return []
    data = yaml.safe_load(_CASE_PATH.read_text(encoding="utf-8")) or {}
    cases = data.get("cases", []) if isinstance(data, dict) else []
    return cases if isinstance(cases, list) else []


def _text_blob(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False).lower()


def _normalize_company_code(value: str) -> str:
    return str(value or "").strip().upper()


def _company_impacts(result: dict[str, Any]) -> list[dict[str, Any]]:
    impacts = []
    for item in result.get("company_impact", []) or []:
        if not isinstance(item, dict):
            continue
        code = _normalize_company_code(item.get("code", "") or "")
        company = str(item.get("company", "") or "").strip()
        stance = str(item.get("stance", "") or "").strip().lower()
        confidence = float(item.get("confidence", 0.0) or 0.0)
        reason = str(item.get("reason", "") or "").strip()
        if not code and not company:
            continue
        impacts.append(
            {
                "code": code,
                "company": company,
                "stance": stance if stance in {"benefit", "threat", "neutral"} else "neutral",
                "confidence": round(confidence, 3),
                "reason": reason,
                "supported_in_evidence": bool(item.get("supported_in_evidence", False)),
            }
        )
    return impacts


def _score_tech_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected", {}) or {}
    blob = _text_blob(result)

    catalyst = str((result.get("catalyst_imminence") or {}).get("dominant_window", "")).strip().lower()
    longevity = str((result.get("longevity") or {}).get("dominant_horizon", "")).strip().lower()
    bottleneck = str((result.get("bottleneck") or {}).get("importance", "")).strip().lower()
    grade = str((result.get("evidence_quality") or {}).get("grade", "")).strip().upper()

    req_terms = [str(x).lower() for x in expected.get("required_terms", []) or []]
    term_hits = {term: (term in blob) for term in req_terms}
    term_score = sum(term_hits.values()) / len(req_terms) if req_terms else 1.0

    cat_target = str(expected.get("catalyst_dominant", expected.get("catalyst_window", ""))).strip().lower()
    lon_target = str(expected.get("longevity_dominant", "")).strip().lower()
    catalyst_score = 1.0 if cat_target and catalyst == cat_target else 0.0 if cat_target else None
    longevity_score = 1.0 if lon_target and longevity == lon_target else 0.0 if lon_target else None
    if catalyst_score is None and longevity_score is None:
        longevity_score = 0.0
        catalyst_score = 0.0
    elif catalyst_score is None:
        catalyst_score = longevity_score
    elif longevity_score is None:
        longevity_score = catalyst_score

    bottleneck_target = str(expected.get("bottleneck_importance", "")).strip().lower()
    bottleneck_score = 1.0 if bottleneck == bottleneck_target else 0.0

    evidence_score = GRADE_SCORE.get(grade, 0) / 4.0
    coverage = result.get("evidence_pack", {}).get("coverage", {}) if isinstance(result.get("evidence_pack"), dict) else {}
    source_diversity = min(1.0, len(coverage.get("sources", []) or []) / 4.0)

    tech_total = (
        0.25 * catalyst_score
        + 0.25 * bottleneck_score
        + 0.20 * term_score
        + 0.20 * evidence_score
        + 0.10 * source_diversity
    )

    return {
        "catalyst": catalyst,
        "longevity": longevity,
        "bottleneck": bottleneck,
        "grade": grade,
        "term_hits": term_hits,
        "term_score": round(term_score, 3),
        "catalyst_score": catalyst_score,
        "longevity_score": longevity_score,
        "bottleneck_score": bottleneck_score,
        "evidence_score": round(evidence_score, 3),
        "source_diversity": round(source_diversity, 3),
        "total": round(tech_total, 3),
    }


def _match_company_alias(expected_code: str, impact: dict[str, Any]) -> bool:
    code = _normalize_company_code(expected_code)
    if not code:
        return False
    impact_code = _normalize_company_code(impact.get("code", "") or "")
    impact_company = str(impact.get("company", "") or "").strip().lower()
    if impact_code == code:
        return True
    aliases = {
        "000660": ["sk hynix", "hynix", "sk하이닉스"],
        "005930": ["samsung", "samsung electronics", "삼성전자"],
        "042700": ["hanmi semiconductor", "hanmi", "한미반도체"],
        "NVDA": ["nvidia"],
        "TSM": ["tsmc"],
        "ASML": ["asml"],
        "AMAT": ["applied materials"],
        "LRCX": ["lam research", "lam"],
        "KLAC": ["kla"],
        "MU": ["micron"],
        "INTC": ["intel"],
    }
    return any(alias in impact_company for alias in aliases.get(code, []))


def _score_market_case(result: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expected = case.get("expected", {}) or {}
    impacts = _company_impacts(result)
    exp_benef = [str(x).upper() for x in expected.get("expected_beneficiaries", []) or []]

    matched = []
    match_scores = []
    for code in exp_benef:
        hit = None
        for impact in impacts:
            if _match_company_alias(code, impact):
                hit = impact
                break
        if hit is None:
            match_scores.append(0.0)
            continue
        matched.append(
            {
                "expected_code": code,
                "matched_code": hit.get("code", ""),
                "company": hit.get("company", ""),
                "stance": hit.get("stance", ""),
                "confidence": hit.get("confidence", 0.0),
                "reason": hit.get("reason", ""),
            }
        )
        stance = str(hit.get("stance", "")).lower()
        confidence = float(hit.get("confidence", 0.0))
        if stance in {"benefit", "threat"} and confidence >= 0.75:
            match_scores.append(1.0)
        elif stance in {"benefit", "threat"} and confidence >= 0.55:
            match_scores.append(0.8)
        elif stance == "neutral" and confidence >= 0.55:
            match_scores.append(0.35)
        else:
            match_scores.append(0.15)

    beneficiary_coverage = sum(1 for s in match_scores if s > 0.0) / len(exp_benef) if exp_benef else 1.0
    beneficiary_quality = sum(match_scores) / len(match_scores) if match_scores else 1.0

    non_neutral = [x for x in impacts if str(x.get("stance", "")).lower() in {"benefit", "threat"}]
    if non_neutral:
        quality_samples = []
        for item in non_neutral:
            conf = float(item.get("confidence", 0.0))
            if conf >= 0.80:
                quality_samples.append(1.0)
            elif conf >= 0.65:
                quality_samples.append(0.8)
            elif conf >= 0.50:
                quality_samples.append(0.6)
            else:
                quality_samples.append(0.3)
        impact_quality = sum(quality_samples) / len(quality_samples)
    else:
        impact_quality = 0.0

    market_total = 0.65 * beneficiary_quality + 0.35 * impact_quality

    return {
        "expected_beneficiaries": exp_benef,
        "matched_beneficiaries": matched,
        "beneficiary_coverage": round(beneficiary_coverage, 3),
        "beneficiary_quality": round(beneficiary_quality, 3),
        "impact_quality": round(impact_quality, 3),
        "total": round(market_total, 3),
        "impact_count": len(impacts),
        "non_neutral_count": len(non_neutral),
    }


def _score_case(case: dict[str, Any], tech_result: dict[str, Any], market_result: dict[str, Any]) -> dict[str, Any]:
    tech = _score_tech_case(tech_result, case)
    market = _score_market_case(market_result, case)
    return {
        "tech": tech,
        "market": market,
        "combined": round(0.7 * tech["total"] + 0.3 * market["total"], 3),
    }


def _result_row(case: dict[str, Any], baseline_raw: dict[str, Any], rag_raw: dict[str, Any]) -> dict[str, Any]:
    baseline_scores = _score_case(case, baseline_raw, baseline_raw)
    rag_scores = _score_case(case, rag_raw, rag_raw)
    return {
        "id": case.get("id", case.get("topic", "")),
        "difficulty": case.get("difficulty", "unknown"),
        "question": case.get("question", case.get("topic", "")),
        "expected": case.get("expected", {}),
        "baseline": {
            "score": baseline_scores["tech"],
            "market_score": baseline_scores["market"],
            "combined": baseline_scores["combined"],
            "raw": baseline_raw,
        },
        "rag": {
            "score": rag_scores["tech"],
            "market_score": rag_scores["market"],
            "combined": rag_scores["combined"],
            "raw": rag_raw,
        },
        "delta": {
            "tech_total": round(rag_scores["tech"]["total"] - baseline_scores["tech"]["total"], 3),
            "market_total": round(rag_scores["market"]["total"] - baseline_scores["market"]["total"], 3),
            "combined": round(rag_scores["combined"] - baseline_scores["combined"], 3),
            "catalyst_score": round(rag_scores["tech"]["catalyst_score"] - baseline_scores["tech"]["catalyst_score"], 3),
            "tech_grade": f"{baseline_scores['tech']['grade']} -> {rag_scores['tech']['grade']}",
            "tech_evidence": round(rag_scores["tech"]["evidence_score"] - baseline_scores["tech"]["evidence_score"], 3),
            "market_beneficiary": round(
                rag_scores["market"]["beneficiary_quality"] - baseline_scores["market"]["beneficiary_quality"],
                3,
            ),
        },
    }


def render_tech_potential_benchmark_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Tech Potential Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at: {report.get('generated_at', 'unknown')}")
    lines.append(f"- Cases evaluated: {report.get('cases_evaluated', 0)}")
    lines.append(f"- Tech baseline avg: {report.get('tech_baseline_avg', 0.0):.3f}")
    lines.append(f"- Tech RAG avg: {report.get('tech_rag_avg', 0.0):.3f}")
    lines.append(f"- Tech improvement: {report.get('tech_improvement', 0.0):+.3f}")
    lines.append(f"- Market baseline avg: {report.get('market_baseline_avg', 0.0):.3f}")
    lines.append(f"- Market RAG avg: {report.get('market_rag_avg', 0.0):.3f}")
    lines.append(f"- Market improvement: {report.get('market_improvement', 0.0):+.3f}")
    lines.append("")
    for row in report.get("results", []):
        lines.append(f"## {row.get('id', '')}")
        lines.append(f"- Difficulty: {row.get('difficulty', '')}")
        lines.append(f"- Question: {row.get('question', '')}")
        lines.append("")
        lines.append("| Component | Baseline | RAG | Delta |")
        lines.append("| --- | ---: | ---: | ---: |")
        lines.append(f"| Catalyst Imminence | {row['baseline']['score']['catalyst_score']:.3f} | {row['rag']['score']['catalyst_score']:.3f} | {row['delta']['catalyst_score']:+.3f} |")
        lines.append(f"| Bottleneck | {row['baseline']['score']['bottleneck_score']:.3f} | {row['rag']['score']['bottleneck_score']:.3f} |  |")
        lines.append(f"| Term coverage | {row['baseline']['score']['term_score']:.3f} | {row['rag']['score']['term_score']:.3f} |  |")
        lines.append(f"| Evidence quality | {row['baseline']['score']['evidence_score']:.3f} | {row['rag']['score']['evidence_score']:.3f} | {row['delta']['tech_evidence']:+.3f} |")
        lines.append(f"| Source diversity | {row['baseline']['score']['source_diversity']:.3f} | {row['rag']['score']['source_diversity']:.3f} |  |")
        lines.append(f"| Tech total | {row['baseline']['score']['total']:.3f} | {row['rag']['score']['total']:.3f} | {row['delta']['tech_total']:+.3f} |")
        lines.append(f"| Market beneficiary coverage | {row['baseline']['market_score']['beneficiary_quality']:.3f} | {row['rag']['market_score']['beneficiary_quality']:.3f} | {row['delta']['market_beneficiary']:+.3f} |")
        lines.append(f"| Market total | {row['baseline']['market_score']['total']:.3f} | {row['rag']['market_score']['total']:.3f} | {row['delta']['market_total']:+.3f} |")
        lines.append("")
        lines.append("### RAG market matches")
        if row["rag"]["market_score"]["matched_beneficiaries"]:
            for item in row["rag"]["market_score"]["matched_beneficiaries"]:
                lines.append(
                    f"- {item['expected_code']} -> {item['company']} ({item['matched_code']}) "
                    f"{item['stance']} conf={item['confidence']:.2f}"
                )
        else:
            lines.append("- none")
        lines.append("")
    return "\n".join(lines)


def run_tech_potential_benchmark(limit: int | None = None) -> dict[str, Any]:
    cases = _load_cases()
    if limit is not None:
        cases = cases[: max(1, limit)]

    results = []
    for case in cases:
        if remaining_call_budget() is not None and remaining_call_budget() < 2:
            break

        baseline_raw = assess_technology_potential_without_rag(
            case["topic"],
            company_hint=case.get("company_hint", ""),
            domain_hint=case.get("domain_hint", ""),
        )
        rag_raw = assess_technology_potential(
            case["topic"],
            company_hint=case.get("company_hint", ""),
            domain_hint=case.get("domain_hint", ""),
            top_k=int(case.get("top_k", 8)),
        )

        results.append(_result_row(case, baseline_raw, rag_raw.__dict__))

    tech_baseline_avg = sum(r["baseline"]["score"]["total"] for r in results) / len(results) if results else 0.0
    tech_rag_avg = sum(r["rag"]["score"]["total"] for r in results) / len(results) if results else 0.0
    market_baseline_avg = sum(r["baseline"]["market_score"]["total"] for r in results) / len(results) if results else 0.0
    market_rag_avg = sum(r["rag"]["market_score"]["total"] for r in results) / len(results) if results else 0.0

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cases_evaluated": len(results),
        "tech_baseline_avg": round(tech_baseline_avg, 3),
        "tech_rag_avg": round(tech_rag_avg, 3),
        "tech_improvement": round(tech_rag_avg - tech_baseline_avg, 3),
        "market_baseline_avg": round(market_baseline_avg, 3),
        "market_rag_avg": round(market_rag_avg, 3),
        "market_improvement": round(market_rag_avg - market_baseline_avg, 3),
        "baseline_avg": round(tech_baseline_avg, 3),
        "rag_avg": round(tech_rag_avg, 3),
        "improvement": round(tech_rag_avg - tech_baseline_avg, 3),
        "results": results,
    }
    return report


def save_tech_potential_benchmark_report(
    path: str = "tech_potential_benchmark_report.json",
    limit: int | None = None,
) -> dict[str, Any]:
    report = run_tech_potential_benchmark(limit=limit)
    Path(path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = Path(path).with_suffix(".md")
    md_path.write_text(render_tech_potential_benchmark_markdown(report), encoding="utf-8")
    return report
