from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agent.semiconductor_event_utils import classify_event_type
from app.db.postgres import get_pg_conn


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if isinstance(value, str):
        return [value] if value.strip() else []
    return []


def _coerce_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [x for x in value if isinstance(x, dict)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _normalize_forecast_payloads(
    scenarios_raw: Any,
    signals_raw: Any,
    features_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenarios = _coerce_json_list(scenarios_raw)
    signals = _coerce_json_dict(signals_raw)
    features = _coerce_json_dict(features_raw)

    # Recover older rows where scenarios/signals were stored in the opposite
    # columns. Keeping this here lets the experience store learn from older
    # backtests instead of silently dropping their context.
    if not scenarios and not signals:
        alt_scenarios = _coerce_json_list(signals_raw)
        alt_signals = _coerce_json_dict(scenarios_raw)
        if alt_scenarios and alt_signals:
            return alt_scenarios, alt_signals, features

    return scenarios, signals, features


def _direction_matches(expected_return: float, realized_return: float) -> bool:
    # Binary target: UP (> 0) versus NON_UP (<= 0), including a flat forecast.
    return (expected_return > 0) == (realized_return > 0)


def _success_label(interval_hit: bool, abs_error: float, band_width: float, expected_return: float, realized_return: float) -> str:
    if interval_hit and abs_error <= max(0.02, band_width * 0.25):
        return "success"
    if interval_hit:
        return "partial"
    if _direction_matches(expected_return, realized_return):
        return "partial"
    return "failure"


def _first_or_empty(items: list[tuple[Any, Any]]) -> str:
    if not items:
        return ""
    return str(items[0][0] or "")


def _derive_domain(signals: dict[str, Any]) -> str:
    tech_domains = signals.get("tech_top_domains") or []
    event_domains = signals.get("top_domains") or []
    if isinstance(tech_domains, list) and tech_domains:
        return _first_or_empty(tech_domains)
    if isinstance(event_domains, list) and event_domains:
        return _first_or_empty(event_domains)
    return ""


def _event_signature(signals: dict[str, Any]) -> str:
    domain = _derive_domain(signals) or "general"
    recent_30 = int(signals.get("recent_30d_count", 0) or 0)
    recent_180 = int(signals.get("recent_180d_count", 0) or 0)
    tech_events = int(signals.get("tech_event_count", 0) or 0)
    tech_docs = int(signals.get("tech_doc_count", 0) or 0)
    novelty_bucket = "fresh" if recent_30 > 0 else "historical" if recent_180 > 0 else "sparse"
    tech_bucket = "rich" if tech_docs >= 40 else "mid" if tech_docs >= 15 else "thin"
    event_bucket = "eventful" if tech_events >= 3 else "light" if tech_events >= 1 else "none"
    return f"{domain}|{novelty_bucket}|{tech_bucket}|{event_bucket}"


def _infer_event_type(signals: dict[str, Any], features: dict[str, Any], notes: str, feedback: str) -> str:
    explicit = str(
        signals.get("event_type")
        or (signals.get("normalized_event") or {}).get("event_type")
        or ""
    ).strip()
    if explicit:
        return explicit
    domain = _derive_domain(signals)
    top_sources = " ".join(str(x[0]) for x in (signals.get("tech_top_sources") or []) if isinstance(x, (list, tuple)) and x)
    top_domains = " ".join(str(x[0]) for x in (signals.get("top_domains") or []) if isinstance(x, (list, tuple)) and x)
    text = " ".join(
        [
            str(notes or ""),
            str(feedback or ""),
            str(top_sources or ""),
            str(top_domains or ""),
            str(domain or ""),
        ]
    )
    event_type = classify_event_type(text, domain)
    if event_type == "technology_update":
        tech_event_count = int(signals.get("tech_event_count", 0) or 0)
        if tech_event_count >= 2 and domain in {"hbm", "packaging", "litho", "logic"}:
            return "technology_breakthrough"
    return event_type


def _extract_error_tags(
    *,
    interval_hit: bool,
    expected_return: float,
    realized_return: float,
    abs_error: float,
    low_return: float,
    high_return: float,
    feedback: str,
    features: dict[str, Any],
    signals: dict[str, Any],
) -> list[str]:
    tags: list[str] = []
    band_width = max(0.0, float(high_return) - float(low_return))
    fb = str(feedback or "").lower()
    trend_strength = float(features.get("trend_strength", 0.0) or 0.0)
    return_20d = float(features.get("return_20d", 0.0) or 0.0)
    event_bias = float(signals.get("event_bias", 0.0) or 0.0)
    tech_bias = float(signals.get("tech_bias", 0.0) or 0.0)
    macro_bias = float(signals.get("macro_bias", 0.0) or 0.0)
    tech_doc_count = int(signals.get("tech_doc_count", 0) or 0)
    tech_event_count = int(signals.get("tech_event_count", 0) or 0)
    recent_30 = int(signals.get("recent_30d_count", 0) or 0)
    tech_sources = signals.get("tech_top_sources") or []
    source_names = {str(src[0]).lower() for src in tech_sources if isinstance(src, (list, tuple)) and src}
    company_specific_source = any(
        key in source
        for source in source_names
        for key in ["samsung", "skhynix", "hynix", "nvidia", "asml", "tsmc", "micron", "hanmi", "applied", "lam", "kla"]
    )
    expected_abs = abs(expected_return)
    realized_abs = abs(realized_return)

    if not _direction_matches(expected_return, realized_return):
        tags.append("direction_flip")

    if (
        expected_abs >= 0.04
        and realized_abs < expected_abs * 0.45
        and ((expected_return > 0 and return_20d > 0.08) or (expected_return < 0 and return_20d < -0.08))
    ):
        tags.append("already_priced_in_miss")

    if "macro spillover" in fb or abs(macro_bias) >= 0.08:
        tags.append("macro_override")

    if realized_abs > max(abs(low_return), abs(high_return), 0.01) * 1.5:
        tags.append("volatility_underweighted")
        tags.append("band_too_narrow")

    if tech_doc_count > 0 and tech_event_count == 0:
        tags.append("missing_tech_event_context")

    if abs(tech_bias) >= 0.25 and not _direction_matches(tech_bias, realized_return):
        tags.append("tech_signal_misread")

    if recent_30 > 0 and abs(tech_bias) >= 0.20 and realized_abs < max(0.015, expected_abs * 0.35):
        tags.append("novelty_overestimated")

    if (
        abs(tech_bias) >= 0.25
        and expected_abs >= 0.04
        and realized_abs < expected_abs * 0.5
        and tech_event_count <= 1
        and abs(event_bias) < 0.15
    ):
        tags.append("revenue_linkage_overestimated")

    if (
        expected_abs >= 0.08
        and abs(tech_bias) >= 0.15
        and tech_doc_count < 12
        and tech_event_count == 0
        and not company_specific_source
    ):
        tags.append("wrong_company_mapping")

    return list(dict.fromkeys(tags))


_PRIMARY_PRIORITY = [
    "direction_flip",
    "already_priced_in_miss",
    "macro_override",
    "volatility_underweighted",
    "band_too_narrow",
    "missing_tech_event_context",
    "tech_signal_misread",
    "novelty_overestimated",
    "revenue_linkage_overestimated",
    "wrong_company_mapping",
]


def _primary_pattern(tags: list[str]) -> str:
    for key in _PRIMARY_PRIORITY:
        if key in tags:
            return key
    return tags[0] if tags else ""


@dataclass
class ExperienceCase:
    forecast_id: int
    company_code: str
    as_of: Any
    horizon_days: int
    target_date: Any
    success_label: str
    primary_pattern: str
    error_tags: list[str]
    related_domain: str
    event_signature: str
    context: dict[str, Any]
    outcome: dict[str, Any]


def _fetch_evaluated_cases(limit: int | None = None) -> list[dict[str, Any]]:
    sql = """
        SELECT
          e.forecast_id,
          e.company_code,
          e.as_of,
          e.horizon_days,
          e.target_date,
          e.realized_at,
          e.base_price,
          e.realized_close,
          e.realized_return,
          e.expected_return,
          e.abs_error,
          e.interval_hit,
          e.scenario_hit,
          e.feedback,
          e.evaluated_at,
          e.extra,
          f.notes,
          f.method,
          f.signals,
          f.features,
          f.scenarios,
          f.low_return,
          f.high_return
        FROM price_forecast_evaluations e
        JOIN price_forecasts f ON f.id = e.forecast_id
        ORDER BY e.evaluated_at DESC
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows or []]


def _to_experience_case(row: dict[str, Any]) -> ExperienceCase:
    extra = _coerce_json_dict(row.get("extra"))
    scenarios, signals, features = _normalize_forecast_payloads(
        row.get("scenarios"),
        row.get("signals"),
        row.get("features"),
    )
    signals = signals or _coerce_json_dict(extra.get("signals"))
    features = features or _coerce_json_dict(extra.get("features"))
    expected_return = float(row.get("expected_return") or 0.0)
    realized_return = float(row.get("realized_return") or 0.0)
    abs_error = float(row.get("abs_error") or 0.0)
    low_return = float(row.get("low_return") or extra.get("low_return") or 0.0)
    high_return = float(row.get("high_return") or extra.get("high_return") or 0.0)
    interval_hit = bool(row.get("interval_hit"))
    band_width = max(0.0, high_return - low_return)
    success_label = _success_label(interval_hit, abs_error, band_width, expected_return, realized_return)
    tags = _extract_error_tags(
        interval_hit=interval_hit,
        expected_return=expected_return,
        realized_return=realized_return,
        abs_error=abs_error,
        low_return=low_return,
        high_return=high_return,
        feedback=str(row.get("feedback") or ""),
        features=features,
        signals=signals,
    )
    related_domain = _derive_domain(signals)
    primary_pattern = _primary_pattern(tags)
    event_signature = _event_signature(signals)
    event_type = _infer_event_type(
        signals,
        features,
        str(row.get("notes") or ""),
        str(row.get("feedback") or ""),
    )
    context = {
        "taxonomy_version": "semiconductor_error_taxonomy_v1",
        "method": row.get("method") or "",
        "notes": row.get("notes") or "",
        "scenario_hit": row.get("scenario_hit") or "",
        "feedback": row.get("feedback") or "",
        "event_type": event_type,
        "signals": signals,
        "features": features,
        "scenarios": scenarios,
    }
    outcome = {
        "realized_at": row.get("realized_at").isoformat() if hasattr(row.get("realized_at"), "isoformat") else str(row.get("realized_at") or ""),
        "base_price": float(row.get("base_price") or 0.0),
        "realized_close": float(row.get("realized_close") or 0.0),
        "realized_return": realized_return,
        "expected_return": expected_return,
        "abs_error": abs_error,
        "interval_hit": interval_hit,
        "feedback": row.get("feedback") or "",
    }
    return ExperienceCase(
        forecast_id=int(row["forecast_id"]),
        company_code=str(row.get("company_code") or ""),
        as_of=row.get("as_of"),
        horizon_days=int(row.get("horizon_days") or 0),
        target_date=row.get("target_date"),
        success_label=success_label,
        primary_pattern=primary_pattern,
        error_tags=tags,
        related_domain=related_domain,
        event_signature=event_signature,
        context=context,
        outcome=outcome,
    )


def _upsert_experience_case(case: ExperienceCase) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO forecast_experience_cases(
                  forecast_id, company_code, as_of, horizon_days, target_date,
                  success_label, primary_pattern, error_tags, related_domain, event_signature,
                  context, outcome, created_at
                ) VALUES (
                  %s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,
                  %s::jsonb,%s::jsonb,%s
                )
                ON CONFLICT (forecast_id) DO UPDATE SET
                  success_label=EXCLUDED.success_label,
                  primary_pattern=EXCLUDED.primary_pattern,
                  error_tags=EXCLUDED.error_tags,
                  related_domain=EXCLUDED.related_domain,
                  event_signature=EXCLUDED.event_signature,
                  context=EXCLUDED.context,
                  outcome=EXCLUDED.outcome
                """,
                (
                    case.forecast_id,
                    case.company_code,
                    case.as_of,
                    case.horizon_days,
                    case.target_date,
                    case.success_label,
                    case.primary_pattern,
                    case.error_tags,
                    case.related_domain or None,
                    case.event_signature,
                    json.dumps(case.context, ensure_ascii=False),
                    json.dumps(case.outcome, ensure_ascii=False),
                    _now_utc(),
                ),
            )
        conn.commit()


def _stat_key(stat_group: str, company_code: str | None, horizon_days: int | None, related_domain: str | None, primary_pattern: str | None) -> str:
    return "|".join(
        [
            stat_group,
            str(company_code or "*"),
            str(horizon_days or "*"),
            str(related_domain or "*"),
            str(primary_pattern or "*"),
        ]
    )


def _replace_stats(rows: list[dict[str, Any]]) -> int:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE forecast_experience_stats")
            for row in rows:
                cur.execute(
                    """
                    INSERT INTO forecast_experience_stats(
                      stat_key, stat_group, company_code, horizon_days, related_domain, primary_pattern,
                      sample_size, success_rate, interval_hit_rate, direction_accuracy,
                      avg_abs_error, avg_expected_return, avg_realized_return, updated_at, extra
                    ) VALUES (
                      %s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s,%s,%s::jsonb
                    )
                    """,
                    (
                        row["stat_key"],
                        row["stat_group"],
                        row.get("company_code"),
                        row.get("horizon_days"),
                        row.get("related_domain"),
                        row.get("primary_pattern"),
                        row["sample_size"],
                        row["success_rate"],
                        row["interval_hit_rate"],
                        row["direction_accuracy"],
                        row["avg_abs_error"],
                        row["avg_expected_return"],
                        row["avg_realized_return"],
                        _now_utc(),
                        json.dumps(row.get("extra", {}), ensure_ascii=False),
                    ),
                )
        conn.commit()
    return len(rows)


def _direction_accuracy_from_cases(cases: list[ExperienceCase]) -> float:
    if not cases:
        return 0.0
    matches = 0
    for case in cases:
        expected = float(case.outcome.get("expected_return") or 0.0)
        realized = float(case.outcome.get("realized_return") or 0.0)
        if _direction_matches(expected, realized):
            matches += 1
    return round(matches / len(cases), 4)


def _build_stats(cases: list[ExperienceCase]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str | None, int | None, str | None, str | None], list[ExperienceCase]] = defaultdict(list)
    for case in cases:
        groups[("company_horizon", case.company_code, case.horizon_days, None, None)].append(case)
        groups[("domain_horizon", None, case.horizon_days, case.related_domain or None, None)].append(case)
        if case.primary_pattern:
            groups[("pattern", None, case.horizon_days, case.related_domain or None, case.primary_pattern)].append(case)
            groups[("company_pattern", case.company_code, case.horizon_days, case.related_domain or None, case.primary_pattern)].append(case)

    rows: list[dict[str, Any]] = []
    for (stat_group, company_code, horizon_days, related_domain, primary_pattern), bucket in groups.items():
        sample_size = len(bucket)
        success_rate = round(sum(1 for x in bucket if x.success_label == "success") / sample_size, 4)
        interval_hit_rate = round(sum(1 for x in bucket if bool(x.outcome.get("interval_hit"))) / sample_size, 4)
        avg_abs_error = round(sum(float(x.outcome.get("abs_error") or 0.0) for x in bucket) / sample_size, 4)
        avg_expected_return = round(sum(float(x.outcome.get("expected_return") or 0.0) for x in bucket) / sample_size, 4)
        avg_realized_return = round(sum(float(x.outcome.get("realized_return") or 0.0) for x in bucket) / sample_size, 4)
        tag_counter: dict[str, int] = defaultdict(int)
        for case in bucket:
            for tag in case.error_tags:
                tag_counter[tag] += 1
        rows.append(
            {
                "stat_key": _stat_key(stat_group, company_code, horizon_days, related_domain, primary_pattern),
                "stat_group": stat_group,
                "company_code": company_code,
                "horizon_days": horizon_days,
                "related_domain": related_domain,
                "primary_pattern": primary_pattern,
                "sample_size": sample_size,
                "success_rate": success_rate,
                "interval_hit_rate": interval_hit_rate,
                "direction_accuracy": _direction_accuracy_from_cases(bucket),
                "avg_abs_error": avg_abs_error,
                "avg_expected_return": avg_expected_return,
                "avg_realized_return": avg_realized_return,
                "extra": {
                    "top_tags": sorted(tag_counter.items(), key=lambda kv: kv[1], reverse=True)[:6],
                    "event_signatures": sorted({case.event_signature for case in bucket}),
                },
            }
        )
    return rows


def _load_existing_experience_cases(limit: int | None = None) -> list[ExperienceCase]:
    sql = """
        SELECT
          forecast_id, company_code, as_of, horizon_days, target_date,
          success_label, primary_pattern, error_tags, related_domain, event_signature,
          context, outcome
        FROM forecast_experience_cases
        ORDER BY as_of DESC
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    cases: list[ExperienceCase] = []
    for (
        forecast_id,
        company_code,
        as_of,
        horizon_days,
        target_date,
        success_label,
        primary_pattern,
        error_tags,
        related_domain,
        event_signature,
        context,
        outcome,
    ) in rows or []:
        cases.append(
            ExperienceCase(
                forecast_id=int(forecast_id),
                company_code=str(company_code or ""),
                as_of=as_of,
                horizon_days=int(horizon_days or 0),
                target_date=target_date,
                success_label=str(success_label or ""),
                primary_pattern=str(primary_pattern or ""),
                error_tags=list(error_tags or []),
                related_domain=str(related_domain or ""),
                event_signature=str(event_signature or ""),
                context=_coerce_json_dict(context),
                outcome=_coerce_json_dict(outcome),
            )
        )
    return cases


def refresh_experience_stats(limit: int | None = None) -> dict[str, Any]:
    cases = _load_existing_experience_cases(limit=limit)
    stat_rows = _build_stats(cases)
    stat_count = _replace_stats(stat_rows) if stat_rows else 0
    return {
        "cases_loaded": len(cases),
        "stats_built": stat_count,
    }


def build_experience_cases_only(limit: int | None = None) -> dict[str, Any]:
    raw_rows = _fetch_evaluated_cases(limit=limit)
    cases = [_to_experience_case(row) for row in raw_rows]
    for case in cases:
        _upsert_experience_case(case)
    success = sum(1 for case in cases if case.success_label == "success")
    partial = sum(1 for case in cases if case.success_label == "partial")
    failure = sum(1 for case in cases if case.success_label == "failure")
    return {
        "cases_built": len(cases),
        "success_count": success,
        "partial_count": partial,
        "failure_count": failure,
    }


def build_forecast_experience_memory(limit: int | None = None) -> dict[str, Any]:
    case_summary = build_experience_cases_only(limit=limit)
    stats_summary = refresh_experience_stats()
    primary_counter: dict[str, int] = defaultdict(int)
    for case in _load_existing_experience_cases():
        if case.primary_pattern:
            primary_counter[case.primary_pattern] += 1
    return {
        **case_summary,
        "stats_built": stats_summary.get("stats_built", 0),
        "top_primary_patterns": sorted(primary_counter.items(), key=lambda kv: kv[1], reverse=True)[:10],
    }


def backfill_experience_event_types(limit: int | None = None) -> dict[str, Any]:
    sql = """
        SELECT fec.forecast_id, fec.related_domain, fec.context, pf.signals, pf.features
        FROM forecast_experience_cases fec
        LEFT JOIN price_forecasts pf ON pf.id = fec.forecast_id
        ORDER BY fec.as_of DESC
    """
    params: list[Any] = []
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))

    updated = 0
    skipped = 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            for forecast_id, related_domain, context, forecast_signals, forecast_features in rows or []:
                context_d = _coerce_json_dict(context)
                context_signals = _coerce_json_dict(context_d.get("signals"))
                latest_signals = _coerce_json_dict(forecast_signals)
                signals = latest_signals if latest_signals.get("normalized_event") else (context_signals or latest_signals)
                context_features = _coerce_json_dict(context_d.get("features"))
                latest_features = _coerce_json_dict(forecast_features)
                features = latest_features or context_features
                notes = str(context_d.get("notes") or "")
                feedback = str(context_d.get("feedback") or "")
                inferred = _infer_event_type(signals, features, notes, feedback)
                if str(context_d.get("event_type") or "") == inferred:
                    skipped += 1
                    continue
                context_d["event_type"] = inferred
                if signals and context_d.get("signals") != signals:
                    context_d["signals"] = signals
                if features and context_d.get("features") != features:
                    context_d["features"] = features
                if related_domain and not context_d.get("related_domain"):
                    context_d["related_domain"] = related_domain
                cur.execute(
                    """
                    UPDATE forecast_experience_cases
                    SET context = %s::jsonb
                    WHERE forecast_id = %s
                    """,
                    (json.dumps(context_d, ensure_ascii=False), forecast_id),
                )
                updated += 1
        conn.commit()
    return {
        "updated": updated,
        "skipped": skipped,
    }


def get_experience_profile(
    *,
    company_code: str | None = None,
    horizon_days: int | None = None,
    related_domain: str | None = None,
    stat_group: str = "company_horizon",
) -> list[dict[str, Any]]:
    clauses = ["stat_group = %s"]
    params: list[Any] = [stat_group]
    if company_code:
        clauses.append("company_code = %s")
        params.append(str(company_code))
    if horizon_days is not None:
        clauses.append("horizon_days = %s")
        params.append(int(horizon_days))
    if related_domain:
        clauses.append("related_domain = %s")
        params.append(str(related_domain))

    sql = f"""
        SELECT
          stat_key, stat_group, company_code, horizon_days, related_domain, primary_pattern,
          sample_size, success_rate, interval_hit_rate, direction_accuracy,
          avg_abs_error, avg_expected_return, avg_realized_return, updated_at, extra
        FROM forecast_experience_stats
        WHERE {" AND ".join(clauses)}
        ORDER BY sample_size DESC, success_rate DESC NULLS LAST
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows or []]


def get_case_profile(
    *,
    company_code: str | None = None,
    horizon_days: int | None = None,
    related_domain: str | None = None,
    event_type: str | None = None,
) -> dict[str, Any]:
    clauses = ["1=1"]
    params: list[Any] = []
    if company_code:
        clauses.append("company_code = %s")
        params.append(str(company_code))
    if horizon_days is not None:
        clauses.append("horizon_days = %s")
        params.append(int(horizon_days))
    if related_domain:
        clauses.append("related_domain = %s")
        params.append(str(related_domain))
    if event_type:
        clauses.append("context->>'event_type' = %s")
        params.append(str(event_type))

    sql = f"""
        SELECT success_label, primary_pattern, error_tags, outcome, context
        FROM forecast_experience_cases
        WHERE {" AND ".join(clauses)}
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    sample_size = len(rows or [])
    if sample_size == 0:
        return {
            "company_code": company_code,
            "horizon_days": horizon_days,
            "related_domain": related_domain,
            "event_type": event_type or "",
            "sample_size": 0,
            "success_rate": 0.0,
            "interval_hit_rate": 0.0,
            "direction_accuracy": 0.0,
            "avg_abs_error": 0.0,
            "avg_expected_return": 0.0,
            "avg_realized_return": 0.0,
            "top_tags": [],
        }

    success = 0
    interval_hits = 0
    direction_matches = 0
    abs_error_sum = 0.0
    expected_sum = 0.0
    realized_sum = 0.0
    tag_counter: dict[str, int] = defaultdict(int)

    for success_label, _primary_pattern, error_tags, outcome, _context in rows:
        outcome_d = _coerce_json_dict(outcome)
        expected = float(outcome_d.get("expected_return") or 0.0)
        realized = float(outcome_d.get("realized_return") or 0.0)
        abs_error = float(outcome_d.get("abs_error") or 0.0)
        if success_label == "success":
            success += 1
        if bool(outcome_d.get("interval_hit")):
            interval_hits += 1
        if _direction_matches(expected, realized):
            direction_matches += 1
        abs_error_sum += abs_error
        expected_sum += expected
        realized_sum += realized
        for tag in _coerce_str_list(error_tags):
            tag_counter[tag] += 1

    return {
        "company_code": company_code,
        "horizon_days": horizon_days,
        "related_domain": related_domain,
        "event_type": event_type or "",
        "sample_size": sample_size,
        "success_rate": round(success / sample_size, 4),
        "interval_hit_rate": round(interval_hits / sample_size, 4),
        "direction_accuracy": round(direction_matches / sample_size, 4),
        "avg_abs_error": round(abs_error_sum / sample_size, 4),
        "avg_expected_return": round(expected_sum / sample_size, 4),
        "avg_realized_return": round(realized_sum / sample_size, 4),
        "top_tags": sorted(tag_counter.items(), key=lambda kv: kv[1], reverse=True)[:8],
    }


def find_similar_experience_cases(
    *,
    company_code: str,
    horizon_days: int,
    related_domain: str | None = None,
    event_type: str | None = None,
    primary_pattern: str | None = None,
    success_label: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    clauses = ["company_code = %s", "horizon_days = %s"]
    params: list[Any] = [str(company_code), int(horizon_days)]
    if related_domain:
        clauses.append("related_domain = %s")
        params.append(str(related_domain))
    if event_type:
        clauses.append("context->>'event_type' = %s")
        params.append(str(event_type))
    if primary_pattern:
        clauses.append("primary_pattern = %s")
        params.append(str(primary_pattern))
    if success_label:
        clauses.append("success_label = %s")
        params.append(str(success_label))

    sql = f"""
        SELECT
          forecast_id, company_code, as_of, horizon_days, target_date,
          success_label, primary_pattern, error_tags, related_domain, event_signature,
          context, outcome
        FROM forecast_experience_cases
        WHERE {" AND ".join(clauses)}
        ORDER BY as_of DESC
        LIMIT %s
    """
    params.append(max(1, int(limit)))
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in rows or []]
