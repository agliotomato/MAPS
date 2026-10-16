"""
Scenario-based stock forecast engine.

The goal is not to "guess" a single price. Instead we:
1. Build data-driven scenarios from price history, event signals, and market metrics.
2. Store every forecast with an as_of timestamp.
3. Later evaluate realized returns against the predicted range.
4. Produce feedback when the forecast band was too wide / too narrow / wrong-direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
import json
import logging
import math
from statistics import mean, pstdev
from typing import Any

from app.agent.semiconductor_event_utils import normalize_event_from_evidence
from app.db.postgres import get_pg_conn

_log = logging.getLogger(__name__)

DEFAULT_COMPANIES = [
    "005930",
    "000660",
    "042700",
    "NVDA",
    "TSM",
    "ASML",
    "AMAT",
    "LRCX",
    "KLAC",
    "MU",
    "INTC",
]

DEFAULT_HORIZONS = [7, 14, 30]

COMPANY_NAMES = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "042700": "Hanmi Semiconductor",
    "NVDA": "NVIDIA",
    "TSM": "TSMC",
    "ASML": "ASML",
    "AMAT": "Applied Materials",
    "LRCX": "Lam Research",
    "KLAC": "KLA",
    "MU": "Micron",
    "INTC": "Intel",
}

MEMORY_CODES = {"005930", "000660", "MU"}

TECH_THEMES = {
    "005930": {
        "keywords": [
            "HBM",
            "DRAM",
            "GAA",
            "EUV",
            "foundry",
            "CIS",
            "memory",
            "packaging",
            "lithography",
            "BPDN",
            "High-NA",
            "yield",
        ],
        "positive": [
            "mass production",
            "ramp",
            "launch",
            "expand",
            "improve",
            "record",
            "approval",
            "adopt",
            "ship",
            "volume production",
            "roadmap",
            "breakthrough",
            "yield improvement",
        ],
        "negative": [
            "delay",
            "cut",
            "pause",
            "decline",
            "shortage",
            "issue",
            "problem",
            "slow",
            "failure",
            "investigation",
            "suspend",
            "underweight",
        ],
    },
    "000660": {
        "keywords": [
            "HBM",
            "DRAM",
            "CXL",
            "PIM",
            "hybrid bonding",
            "TSV",
            "CoWoS",
            "packaging",
            "memory",
            "LPDDR",
            "DDR",
            "backside power",
        ],
        "positive": [
            "mass production",
            "ramp",
            "launch",
            "expand",
            "improve",
            "record",
            "approval",
            "adopt",
            "ship",
            "volume production",
            "roadmap",
            "breakthrough",
            "yield improvement",
        ],
        "negative": [
            "delay",
            "cut",
            "pause",
            "decline",
            "shortage",
            "issue",
            "problem",
            "slow",
            "failure",
            "investigation",
            "suspend",
            "underweight",
        ],
    },
    "042700": {
        "keywords": [
            "advanced packaging",
            "hybrid bonding",
            "TSV",
            "interposer",
            "substrate",
            "CoWoS",
            "fan-out",
            "OSAT",
            "HBM",
            "yield",
            "metrology",
            "inspection",
        ],
        "positive": [
            "mass production",
            "ramp",
            "launch",
            "expand",
            "improve",
            "record",
            "approval",
            "adopt",
            "ship",
            "volume production",
            "roadmap",
            "breakthrough",
            "yield improvement",
        ],
        "negative": [
            "delay",
            "cut",
            "pause",
            "decline",
            "shortage",
            "issue",
            "problem",
            "slow",
            "failure",
            "investigation",
            "suspend",
            "underweight",
        ],
    },
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _norm_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    return mean(values) if values else default


def _safe_stdev(values: list[float], default: float = 0.0) -> float:
    if len(values) < 2:
        return default
    try:
        return pstdev(values)
    except Exception:
        return default


def _to_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)
    if value is None:
        return _now_utc()
    raw = str(value).strip()
    if not raw:
        return _now_utc()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except Exception:
        return _now_utc()


def _company_name(code: str) -> str:
    return COMPANY_NAMES.get(_norm_code(code), _norm_code(code))


@dataclass
class ForecastScenario:
    name: str
    probability: float
    expected_return: float
    low_return: float
    high_return: float
    rationale: str


@dataclass
class PriceForecast:
    company_code: str
    company_name: str
    as_of: str
    published_cutoff: str
    horizon_days: int
    target_date: str
    base_price: float
    expected_return: float
    low_return: float
    high_return: float
    expected_price: float
    low_price: float
    high_price: float
    method: str
    scenarios: list[ForecastScenario] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    signals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    created_at: str = ""


@dataclass
class ForecastEvaluation:
    forecast_id: int
    company_code: str
    as_of: str
    horizon_days: int
    target_date: str
    realized_at: str
    base_price: float
    realized_close: float
    realized_return: float
    expected_return: float
    abs_error: float
    interval_hit: bool
    scenario_hit: str
    feedback: str
    evaluated_at: str
    extra: dict[str, Any] = field(default_factory=dict)


def select_price_history(history: list[dict[str, Any]], as_of: datetime,
                         lookback_days: int = 90) -> list[dict[str, Any]]:
    """Keep the calendar window, extending only when needed for 60 observed intervals.

    The caller supplies the market-local cutoff. Unknown availability stays unknown;
    this date filter alone does not certify the historical data vintage.
    """
    if as_of.tzinfo is None:
        raise ValueError("Price cutoff requires a timezone")
    end = as_of.date()
    eligible = []
    for row in history:
        day = date.fromisoformat(str(row["trade_date"])[:10])
        close = row.get("close")
        if day >= end or close is None or not math.isfinite(float(close)) or float(close) <= 0:
            continue
        if row.get("available_at") is not None:
            available = datetime.fromisoformat(str(row["available_at"]).replace("Z", "+00:00"))
            if available.tzinfo is None:
                raise ValueError("Price availability requires a timezone")
            if available >= as_of:
                continue
        eligible.append((day, row))
    eligible.sort(key=lambda item: item[0])
    if len({day for day, _ in eligible}) != len(eligible):
        raise ValueError("Duplicate price trade dates")
    start = end - timedelta(days=lookback_days)
    recent = [row for day, row in eligible if day >= start]
    return recent if len(recent) >= 61 else [row for _, row in eligible[-61:]]


def _fetch_price_history(company_code: str, as_of: datetime, lookback_days: int = 90) -> list[dict[str, Any]]:
    # Strict cutoff: only use data strictly before as_of's date.
    # This avoids leaking the same day's close into a forecast made "at" as_of.
    start_date = (as_of.date() - timedelta(days=lookback_days)).isoformat()
    end_date = as_of.date().isoformat()
    sql = """
        WITH recent_prices AS (
            SELECT trade_date, close, volume FROM price_daily
            WHERE company_code = %s AND trade_date >= %s::date
              AND trade_date < %s::date AND close > 0
        ), minimum_prices AS (
            SELECT trade_date, close, volume FROM price_daily
            WHERE company_code = %s AND trade_date < %s::date AND close > 0
            ORDER BY trade_date DESC LIMIT %s
        )
        SELECT trade_date, close, volume FROM recent_prices
        UNION
        SELECT trade_date, close, volume FROM minimum_prices
        ORDER BY trade_date ASC
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (company_code, start_date, end_date, company_code, end_date, 61))
            rows = cur.fetchall()
    history = []
    for trade_date, close, volume in rows or []:
        if close is None:
            continue
        history.append(
            {
                "trade_date": trade_date,
                "close": float(close),
                "volume": float(volume or 0.0),
            }
        )
    return select_price_history(history, as_of, lookback_days)


def _fetch_trade_dates(company_code: str, start_date: date, end_date: date) -> list[date]:
    sql = """
        SELECT trade_date
        FROM price_daily
        WHERE company_code = %s
          AND trade_date BETWEEN %s::date AND %s::date
        ORDER BY trade_date ASC
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (company_code, start_date.isoformat(), end_date.isoformat()))
            rows = cur.fetchall()
    return [row[0] for row in rows or []]


def _closest_close_on_or_after(company_code: str, target_date: date) -> tuple[date | None, float | None]:
    sql = """
        SELECT trade_date, close
        FROM price_daily
        WHERE company_code = %s
          AND trade_date >= %s::date
        ORDER BY trade_date ASC
        LIMIT 1
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (company_code, target_date.isoformat()))
            row = cur.fetchone()
    if not row:
        return None, None
    return row[0], float(row[1]) if row[1] is not None else None


def _compute_price_features(history: list[dict[str, Any]]) -> dict[str, float | None]:
    closes = [float(x["close"]) for x in history if x.get("close") is not None]
    volumes = [float(x.get("volume", 0.0) or 0.0) for x in history]
    if len(closes) < 2:
        return {}

    daily_returns = []
    for prev, cur in zip(closes[:-1], closes[1:]):
        if prev:
            daily_returns.append((cur - prev) / prev)

    latest = closes[-1]

    def _return_back(n: int) -> float | None:
        if len(closes) <= n:
            return None
        base = closes[-(n + 1)]
        return (latest - base) / base if base else None

    vol_20 = _safe_stdev(daily_returns[-20:], 0.0)
    vol_60 = _safe_stdev(daily_returns[-60:], 0.0)
    max_60 = max(closes[-60:]) if len(closes) >= 1 else latest
    min_60 = min(closes[-60:]) if len(closes) >= 1 else latest
    volume_avg_20 = _safe_mean(volumes[-20:], 0.0)
    volume_ratio_20 = (volumes[-1] / volume_avg_20) if volume_avg_20 else 1.0
    return_20 = _return_back(20)

    return {
        "latest_close": latest,
        "return_7d": _return_back(7),
        "return_20d": return_20,
        "return_60d": _return_back(60),
        "volatility_20d": vol_20,
        "volatility_60d": vol_60,
        "drawdown_60d": (latest - max_60) / max_60 if max_60 else 0.0,
        "range_60d": (max_60 - min_60) / min_60 if min_60 else 0.0,
        "volume_ratio_20d": volume_ratio_20,
        "trend_strength": (_clamp(return_20 / max(vol_20, 0.01), -3.0, 3.0)
                           if return_20 is not None else None),
    }


def _fetch_event_signals(company_code: str, as_of: datetime) -> dict[str, Any]:
    since_30 = (as_of.date() - timedelta(days=30)).isoformat()
    since_180 = (as_of.date() - timedelta(days=180)).isoformat()
    since_365 = (as_of.date() - timedelta(days=365)).isoformat()

    sql = """
        SELECT ec.event_date, ec.confidence, ec.related_domain, eo.label, eo.ret_5d
        FROM event_candidates ec
        LEFT JOIN event_outcomes eo ON eo.event_id = ec.event_id
          AND eo.computed_at < %s
        WHERE ec.related_company = %s
          AND ec.event_date < %s::date
          AND ec.event_date >= %s::date
        ORDER BY ec.event_date DESC
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (as_of, company_code, as_of.date().isoformat(), since_365))
            rows = cur.fetchall()

    recent_30 = []
    recent_180 = []
    labels = []
    ret_5d_values = []
    domain_counter: dict[str, int] = {}
    for event_date, confidence, related_domain, label, ret_5d in rows or []:
        event_date = event_date.date() if hasattr(event_date, "date") else event_date
        confidence = float(confidence or 0.0)
        related_domain = str(related_domain or "")
        label = str(label or "")
        ret_5d_val = float(ret_5d) if ret_5d is not None else None
        recent_180.append((event_date, confidence, related_domain, label, ret_5d_val))
        if event_date and hasattr(event_date, "isoformat") and event_date.isoformat() >= since_30:
            recent_30.append((event_date, confidence, related_domain, label, ret_5d_val))
        if label:
            labels.append(label)
        if ret_5d_val is not None:
            ret_5d_values.append(ret_5d_val)
        if related_domain:
            domain_counter[related_domain] = domain_counter.get(related_domain, 0) + 1

    positive = sum(1 for x in labels if x == "positive")
    negative = sum(1 for x in labels if x == "negative")
    neutral = sum(1 for x in labels if x == "neutral")
    total = len(labels)
    label_bias = ((positive - negative) / total) if total else 0.0
    avg_ret_5d = _safe_mean(ret_5d_values, 0.0)
    confidence_bias = _safe_mean([x[1] for x in recent_30], 0.0)
    event_bias = _clamp(0.6 * label_bias + 0.4 * _clamp(avg_ret_5d / 0.05, -1.0, 1.0), -1.0, 1.0)

    return {
        "recent_30d_count": len(recent_30),
        "recent_180d_count": len(recent_180),
        "positive_count": positive,
        "negative_count": negative,
        "neutral_count": neutral,
        "label_bias": round(label_bias, 4),
        "avg_ret_5d": round(avg_ret_5d, 4),
        "confidence_bias": round(confidence_bias, 4),
        "event_bias": round(event_bias, 4),
        "top_domains": sorted(domain_counter.items(), key=lambda kv: kv[1], reverse=True)[:3],
    }


def _pick_domain_hint(
    company_code: str,
    event_signals: dict[str, Any] | None = None,
    tech_signals: dict[str, Any] | None = None,
) -> str:
    for bucket in (
        (event_signals or {}).get("top_domains") or [],
        (tech_signals or {}).get("tech_top_domains") or [],
    ):
        if bucket and isinstance(bucket[0], (list, tuple)) and bucket[0]:
            return str(bucket[0][0] or "").strip()
    if _norm_code(company_code) in MEMORY_CODES:
        return "hbm"
    if _norm_code(company_code) == "042700":
        return "packaging"
    return "general"


@lru_cache(maxsize=512)
def _fetch_normalized_event_signal(
    company_code: str,
    as_of: datetime,
    domain_hint: str = "",
) -> dict[str, Any]:
    since_365 = (as_of.date() - timedelta(days=365)).isoformat()
    params: list[Any] = [company_code, as_of.date().isoformat(), since_365]
    domain_clause = ""
    if domain_hint and domain_hint != "general":
        domain_clause = " AND (ec.related_domain = %s OR COALESCE(ec.extra->>'technology', '') ILIKE %s) "
        params.extend([domain_hint, f"%{domain_hint}%"])

    sql = f"""
        SELECT
          ec.event_id,
          ec.event_date,
          ec.event_type,
          ec.source,
          ec.source_tier,
          ec.title,
          ec.summary,
          ec.related_company,
          ec.related_domain,
          ec.evidence_doc_uid,
          ec.confidence
        FROM event_candidates ec
        WHERE ec.related_company = %s
          AND ec.event_date < %s::date
          AND ec.event_date >= %s::date
          {domain_clause}
        ORDER BY ec.event_date DESC, ec.confidence DESC
        LIMIT 5
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()

    if not rows:
        return {}

    recent_90d_cutoff = as_of.date() - timedelta(days=90)
    recent_90d_count = 0
    existing_events: list[dict[str, Any]] = []
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
        ) = row
        event_dt = _to_datetime(event_date)
        if event_dt.date() >= recent_90d_cutoff:
            recent_90d_count += 1
        existing_events.append(
            {
                "event_id": event_id,
                "event_date": event_dt.isoformat(),
                "event_type": str(event_type or ""),
                "source": str(source or ""),
                "source_tier": int(source_tier or 0),
                "title": str(title or ""),
                "summary": str(summary or ""),
                "related_company": str(related_company or company_code),
                "related_domain": str(related_domain or domain_hint or ""),
                "evidence_doc_uid": str(evidence_doc_uid or ""),
                "confidence": float(confidence or 0.0),
            }
        )

    normalized = normalize_event_from_evidence(
        query="",
        items=[],
        company=company_code,
        domain=domain_hint,
        existing_event=existing_events[0],
        recent_90d_count=recent_90d_count,
    )
    if not normalized:
        return {}

    normalized["recent_event_count"] = len(existing_events)
    normalized["recent_90d_event_count"] = recent_90d_count
    normalized["recent_event_titles"] = [evt["title"] for evt in existing_events[:3] if evt.get("title")]
    return normalized


def enrich_signals_with_normalized_event(
    company_code: str,
    as_of: datetime | str,
    signals: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(signals or {})
    as_of_dt = _to_datetime(as_of)
    domain_hint = _pick_domain_hint(company_code, enriched, enriched)
    normalized_event = _fetch_normalized_event_signal(company_code, as_of_dt, domain_hint)
    if not normalized_event:
        return enriched
    enriched.update(
        {
            "normalized_event": normalized_event,
            "event_type": str(normalized_event.get("event_type") or ""),
            "event_technology": str(normalized_event.get("technology") or ""),
            "event_technology_category": str(normalized_event.get("technology_category") or ""),
            "event_catalyst_imminence": str(normalized_event.get("catalyst_imminence") or ""),
            "event_revenue_linkage": str(normalized_event.get("revenue_linkage") or ""),
            "event_market_transmission_speed": str(normalized_event.get("market_transmission_speed") or ""),
            "event_novelty_hint": str(normalized_event.get("novelty_hint") or ""),
            "event_detail_reason": str(normalized_event.get("detail_reason") or ""),
        }
    )
    return enriched


def _fetch_tech_signals(company_code: str, as_of: datetime) -> dict[str, Any]:
    theme = TECH_THEMES.get(_norm_code(company_code), TECH_THEMES["005930"])
    keywords = theme["keywords"]
    positive_terms = [t.lower() for t in theme["positive"]]
    negative_terms = [t.lower() for t in theme["negative"]]

    clauses: list[str] = []
    params: list[Any] = [as_of]
    for kw in keywords:
        pattern = f"%{kw}%"
        clauses.append("(title ILIKE %s OR COALESCE(summary,'') ILIKE %s OR COALESCE(content,'') ILIKE %s)")
        params.extend([pattern, pattern, pattern])

    if not clauses:
        return {
            "tech_doc_count": 0,
            "tech_event_count": 0,
            "tech_doc_bias": 0.0,
            "tech_event_bias": 0.0,
            "tech_bias": 0.0,
            "tech_notes": [],
            "tech_top_sources": [],
            "tech_top_domains": [],
        }

    since_365 = (as_of.date() - timedelta(days=365)).isoformat()

    doc_sql = f"""
        SELECT source, source_type, title, COALESCE(summary, '') AS summary, COALESCE(content, '') AS content, published_at, confidence
        FROM tech_documents
        WHERE published_at IS NOT NULL
          AND published_at < %s
          AND published_at >= %s::date
          AND ({' OR '.join(clauses)})
        ORDER BY published_at DESC
        LIMIT 400
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(doc_sql, [params[0], since_365, *params[1:]])
            doc_rows = cur.fetchall()

    doc_scores: list[float] = []
    source_counter: dict[str, int] = {}
    doc_notes: list[str] = []
    for source, source_type, title, summary, content, published_at, confidence in doc_rows or []:
        text = f"{title or ''} {summary or ''} {content or ''}".lower()
        pos_hits = sum(1 for term in positive_terms if term in text)
        neg_hits = sum(1 for term in negative_terms if term in text)
        age_days = max(0, (as_of.date() - _to_datetime(published_at).date()).days)
        recency = max(0.20, 1.0 - age_days / 365.0)
        source_type_norm = str(source_type or "").lower()
        if source_type_norm in {"irds", "jedec", "hotchips", "paper", "openalex", "arxiv"}:
            source_weight = 0.45
        elif source_type_norm in {"company_newsroom", "official", "equip_doc"} or "newsroom" in source_type_norm:
            source_weight = 0.35
        elif source_type_norm in {"blog", "semi_blog"}:
            source_weight = 0.25
        elif source_type_norm in {"news", "rss"}:
            source_weight = 0.20
        else:
            source_weight = 0.15
        sentiment = 0.18
        if pos_hits > neg_hits:
            sentiment += 0.24 + 0.04 * min(pos_hits, 3)
        elif neg_hits > pos_hits:
            sentiment -= 0.24 + 0.04 * min(neg_hits, 3)
        score = source_weight * recency * sentiment * (0.85 + 0.15 * float(confidence or 0.0))
        doc_scores.append(score)
        source_counter[str(source or source_type or "")] = source_counter.get(str(source or source_type or ""), 0) + 1
        if len(doc_notes) < 8:
            doc_notes.append(f"{source_type}:{str(title or '')[:60]}")

    doc_bias = _clamp(sum(doc_scores) / max(4.0, len(doc_scores) or 1), -1.0, 1.0)

    tech_event_sql = """
        SELECT ec.event_date, ec.confidence, ec.related_domain, eo.label, eo.ret_5d, ec.source, ec.extra
        FROM event_candidates ec
        LEFT JOIN event_outcomes eo ON eo.event_id = ec.event_id
          AND eo.computed_at < %s
        WHERE ec.related_company = %s
          AND ec.event_date < %s::date
          AND ec.event_date >= %s::date
          AND COALESCE(ec.extra->>'origin', '') = 'tech_documents'
        ORDER BY ec.event_date DESC
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(tech_event_sql, (as_of, company_code, as_of.date().isoformat(), since_365))
            event_rows = cur.fetchall()

    labels: list[str] = []
    ret_5d_values: list[float] = []
    event_scores: list[float] = []
    event_domains: dict[str, int] = {}
    for event_date, confidence, related_domain, label, ret_5d, source, extra in event_rows or []:
        label = str(label or "")
        confidence = float(confidence or 0.0)
        ret_5d_val = float(ret_5d) if ret_5d is not None else None
        age_days = max(0, (as_of.date() - _to_datetime(event_date).date()).days)
        recency = max(0.20, 1.0 - age_days / 365.0)
        label_score = 0.0
        if label == "positive":
            label_score = 0.30
        elif label == "negative":
            label_score = -0.30
        if ret_5d_val is not None:
            label_score += _clamp(ret_5d_val / 0.05, -0.35, 0.35)
        event_scores.append(label_score * recency * (0.75 + 0.25 * confidence))
        if label:
            labels.append(label)
        if ret_5d_val is not None:
            ret_5d_values.append(ret_5d_val)
        event_domains[str(related_domain or "")] = event_domains.get(str(related_domain or ""), 0) + 1

    positive = sum(1 for x in labels if x == "positive")
    negative = sum(1 for x in labels if x == "negative")
    neutral = sum(1 for x in labels if x == "neutral")
    total = len(labels)
    event_label_bias = ((positive - negative) / total) if total else 0.0
    event_ret_bias = _clamp(_safe_mean(ret_5d_values, 0.0) / 0.05, -1.0, 1.0)
    tech_event_bias = _clamp(0.55 * event_label_bias + 0.45 * event_ret_bias + sum(event_scores), -1.0, 1.0)
    tech_bias = _clamp(0.65 * doc_bias + 0.35 * tech_event_bias, -1.0, 1.0)

    return {
        "tech_doc_count": len(doc_rows),
        "tech_event_count": len(event_rows),
        "tech_doc_bias": round(doc_bias, 4),
        "tech_event_bias": round(tech_event_bias, 4),
        "tech_bias": round(tech_bias, 4),
        "tech_event_label_bias": round(event_label_bias, 4),
        "tech_event_ret_bias": round(event_ret_bias, 4),
        "tech_positive_count": positive,
        "tech_negative_count": negative,
        "tech_neutral_count": neutral,
        "tech_notes": doc_notes[:8],
        "tech_top_sources": sorted(source_counter.items(), key=lambda kv: kv[1], reverse=True)[:5],
        "tech_top_domains": sorted(event_domains.items(), key=lambda kv: kv[1], reverse=True)[:3],
    }


def _fetch_market_bias(as_of: datetime, company_code: str) -> dict[str, Any]:
    sql = """
        SELECT metric_name, metric_value, observed_at
        FROM metric_observations
        WHERE company_code = 'MARKET'
          AND observed_at < %s
        ORDER BY metric_name ASC, observed_at DESC
        LIMIT 300
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (as_of,))
            rows = cur.fetchall()

    grouped: dict[str, list[tuple[datetime, float]]] = {}
    for metric_name, metric_value, observed_at in rows or []:
        grouped.setdefault(str(metric_name), []).append((_to_datetime(observed_at), float(metric_value)))

    bias = 0.0
    notes: list[str] = []
    for metric_name, series in grouped.items():
        if len(series) < 2:
            continue
        latest = series[0][1]
        prev = series[1][1]
        if prev == 0:
            continue
        change = (latest - prev) / abs(prev)
        metric = metric_name.lower()
        weight = 0.0
        sign = 1.0
        if "book_to_bill_ratio" in metric:
            weight = 0.35
            sign = 1.0
        elif "global_semiconductor_sales" in metric or "industry_production_semiconductor" in metric:
            weight = 0.25
            sign = 1.0
        elif "ppi_semiconductor_devices" in metric or "industrial_production_semiconductor" in metric:
            weight = 0.20
            sign = 1.0
        elif "spot_price_" in metric:
            if _norm_code(company_code) in MEMORY_CODES:
                weight = 0.20
                sign = 1.0
        elif "exchange_rate" in metric or "usdkrw" in metric:
            if _norm_code(company_code) in {"005930", "000660", "042700", "TSM", "MU"}:
                weight = 0.15
                sign = 1.0
        if weight <= 0:
            continue
        bias += weight * _clamp(change * sign, -0.3, 0.3)
        notes.append(f"{metric_name}:{change:+.3f}")

    return {
        "macro_bias": round(_clamp(bias, -0.15, 0.15), 4),
        "macro_notes": notes[:8],
    }


def _scenario_probabilities(micro_bias: float) -> tuple[float, float, float]:
    bull_raw = math.exp(1.1 * micro_bias)
    base_raw = math.exp(0.15)
    bear_raw = math.exp(-1.1 * micro_bias)
    total = bull_raw + base_raw + bear_raw
    return bull_raw / total, base_raw / total, bear_raw / total


def _forecast_for_company(company_code: str, horizon_days: int, as_of: datetime) -> PriceForecast | None:
    history = _fetch_price_history(company_code, as_of, lookback_days=max(90, horizon_days * 4))
    if len(history) < 21:
        return None

    signals = _fetch_event_signals(company_code, as_of)
    tech = _fetch_tech_signals(company_code, as_of)
    market = _fetch_market_bias(as_of, company_code)
    forecast = forecast_from_inputs(company_code, horizon_days, as_of, history, signals, tech, market)
    if forecast is not None:
        forecast.signals = enrich_signals_with_normalized_event(company_code, as_of, forecast.signals)
    return forecast


def forecast_from_inputs(
    company_code: str, horizon_days: int, as_of: datetime,
    history: list[dict[str, Any]], signals: dict[str, Any],
    tech: dict[str, Any], market: dict[str, Any],
) -> PriceForecast | None:
    """Shared scenario calculation over frozen inputs; no retrieval or persistence.

    Callers own point-in-time eligibility and the per-horizon history window.
    Event normalization is display enrichment performed only by the DB wrapper.
    Require the 20-observation trend; omit unavailable returns from momentum and
    renormalize existing weights. Missing history is not an observed flat return.
    """
    if len(history) < 21:
        return None
    features = _compute_price_features(history)
    if features.get("return_7d") is None or features.get("trend_strength") is None:
        return None
    latest_close = float(features.get("latest_close", 0.0))
    if latest_close <= 0:
        return None

    horizon_scale = math.sqrt(max(horizon_days, 1) / 7.0)
    trend_strength = float(features.get("trend_strength", 0.0))
    event_bias = float(signals.get("event_bias", 0.0))
    tech_bias = float(tech.get("tech_bias", 0.0))
    macro_bias = float(market.get("macro_bias", 0.0))
    return_weights = {"return_7d": 0.20, "return_20d": 0.45, "return_60d": 0.35}
    available_weights = {key: weight for key, weight in return_weights.items() if features[key] is not None}
    momentum_mix = sum(weight * features[key] for key, weight in available_weights.items()) / sum(available_weights.values())
    # Tech-first view: semiconductor technical signals dominate the direction.
    micro_bias = _clamp(0.70 * tech_bias + 0.20 * trend_strength + 0.10 * macro_bias, -2.0, 2.0)
    bull_prob, base_prob, bear_prob = _scenario_probabilities(micro_bias)

    expected_return = (
        0.70 * tech_bias
        + 0.20 * momentum_mix * min(1.8, 0.65 + 0.12 * horizon_days)
        + 0.10 * macro_bias
    )
    expected_return = _clamp(expected_return, -0.35, 0.35)

    vol_band = max(
        0.025,
        float(features.get("volatility_20d", 0.0)) * horizon_scale * 1.35 + abs(tech_bias) * 0.025,
    )
    low_return = _clamp(expected_return - vol_band, -0.60, 0.80)
    high_return = _clamp(expected_return + vol_band, -0.60, 0.80)

    bull_return = _clamp(expected_return + vol_band * (0.85 + max(micro_bias, 0.0) * 0.10), -0.60, 0.80)
    bear_return = _clamp(expected_return - vol_band * (0.85 + max(-micro_bias, 0.0) * 0.10), -0.60, 0.80)

    scenarios = [
        ForecastScenario(
            name="bull",
            probability=round(bull_prob, 3),
            expected_return=round(bull_return, 4),
            low_return=round(expected_return, 4),
            high_return=round(high_return, 4),
            rationale="Momentum and recent signals continue to support upside.",
        ),
        ForecastScenario(
            name="base",
            probability=round(base_prob, 3),
            expected_return=round(expected_return, 4),
            low_return=round(low_return, 4),
            high_return=round(high_return, 4),
            rationale="Trend persists but partially mean-reverts into the forecast window.",
        ),
        ForecastScenario(
            name="bear",
            probability=round(bear_prob, 3),
            expected_return=round(bear_return, 4),
            low_return=round(low_return, 4),
            high_return=round(expected_return, 4),
            rationale="Momentum fades or event/macro shocks dominate the horizon.",
        ),
    ]

    target_date = (as_of + timedelta(days=horizon_days)).date()
    expected_price = latest_close * (1.0 + expected_return)
    low_price = latest_close * (1.0 + low_return)
    high_price = latest_close * (1.0 + high_return)

    notes = "; ".join(
        [
            f"trend_strength={trend_strength:+.2f}",
            f"event_bias={event_bias:+.2f}",
            f"tech_bias={tech_bias:+.2f}",
            f"macro_bias={macro_bias:+.2f}",
            f"vol_20d={float(features.get('volatility_20d', 0.0)):.4f}",
        ]
    )
    missing_returns = [key for key in return_weights if key not in available_weights]
    if missing_returns:
        notes += (f"; unavailable_returns={','.join(missing_returns)}"
                  f"; momentum_weight_coverage={sum(available_weights.values()):.2f}"
                  "; available momentum weights renormalized; missing returns are not zero")

    merged_signals = signals | tech | market

    return PriceForecast(
        company_code=_norm_code(company_code),
        company_name=_company_name(company_code),
        as_of=as_of.isoformat(),
        published_cutoff=as_of.isoformat(),
        horizon_days=int(horizon_days),
        target_date=target_date.isoformat(),
        base_price=round(latest_close, 4),
        expected_return=round(expected_return, 4),
        low_return=round(low_return, 4),
        high_return=round(high_return, 4),
        expected_price=round(expected_price, 4),
        low_price=round(low_price, 4),
        high_price=round(high_price, 4),
        method="scenario_v3_available_returns",
        scenarios=scenarios,
        features={k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
        signals=merged_signals,
        notes=notes,
        created_at=_now_utc().isoformat(),
    )


def _upsert_forecast(forecast: PriceForecast) -> int:
    sql = """
        INSERT INTO price_forecasts(
          company_code, horizon_days, as_of, published_cutoff, target_date,
          base_price, expected_return, low_return, high_return,
          expected_price, low_price, high_price, method,
          signals, features, scenarios, notes, created_at
        ) VALUES (
          %s,%s,%s,%s,%s,
          %s,%s,%s,%s,
          %s,%s,%s,%s,
          %s::jsonb,%s::jsonb,%s::jsonb,%s,%s
        )
        ON CONFLICT (company_code, horizon_days, as_of) DO UPDATE SET
          published_cutoff=EXCLUDED.published_cutoff,
          target_date=EXCLUDED.target_date,
          base_price=EXCLUDED.base_price,
          expected_return=EXCLUDED.expected_return,
          low_return=EXCLUDED.low_return,
          high_return=EXCLUDED.high_return,
          expected_price=EXCLUDED.expected_price,
          low_price=EXCLUDED.low_price,
          high_price=EXCLUDED.high_price,
          method=EXCLUDED.method,
          signals=EXCLUDED.signals,
          features=EXCLUDED.features,
          scenarios=EXCLUDED.scenarios,
          notes=EXCLUDED.notes
        RETURNING id
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                (
                    forecast.company_code,
                    forecast.horizon_days,
                    forecast.as_of,
                    forecast.published_cutoff,
                    forecast.target_date,
                    forecast.base_price,
                    forecast.expected_return,
                    forecast.low_return,
                    forecast.high_return,
                    forecast.expected_price,
                    forecast.low_price,
                    forecast.high_price,
                    forecast.method,
                    json.dumps(forecast.signals | {"company_name": forecast.company_name}, ensure_ascii=False),
                    json.dumps(forecast.features, ensure_ascii=False),
                    json.dumps([asdict(x) for x in forecast.scenarios], ensure_ascii=False),
                    forecast.notes,
                    forecast.created_at,
                ),
            )
            forecast_id = int(cur.fetchone()[0])
        conn.commit()
    return forecast_id


def generate_price_forecasts(
    company_codes: list[str] | None = None,
    horizons: list[int] | None = None,
    as_of: datetime | str | None = None,
) -> list[dict[str, Any]]:
    as_of_dt = _to_datetime(as_of)
    companies = [_norm_code(c) for c in (company_codes or DEFAULT_COMPANIES)]
    horizon_list = [int(h) for h in (horizons or DEFAULT_HORIZONS)]
    results: list[dict[str, Any]] = []

    for company_code in companies:
        for horizon in horizon_list:
            forecast = _forecast_for_company(company_code, horizon, as_of_dt)
            if forecast is None:
                continue
            forecast_id = _upsert_forecast(forecast)
            results.append(
                {
                    "forecast_id": forecast_id,
                    "company_code": forecast.company_code,
                    "company_name": forecast.company_name,
                    "as_of": forecast.as_of,
                    "horizon_days": forecast.horizon_days,
                    "target_date": forecast.target_date,
                    "expected_return": forecast.expected_return,
                    "low_return": forecast.low_return,
                    "high_return": forecast.high_return,
                    "expected_price": forecast.expected_price,
                    "low_price": forecast.low_price,
                    "high_price": forecast.high_price,
                    "scenarios": [asdict(x) for x in forecast.scenarios],
                    "signals": forecast.signals,
                    "features": forecast.features,
                    "notes": forecast.notes,
                }
            )

    return results


def _scenario_hit(realized_return: float, scenarios: list[dict[str, Any]]) -> str:
    best_name = "unknown"
    best_dist = float("inf")
    for scenario in scenarios:
        expected = float(scenario.get("expected_return", 0.0))
        dist = abs(realized_return - expected)
        if dist < best_dist:
            best_dist = dist
            best_name = str(scenario.get("name", "unknown"))
    return best_name


def _coerce_json_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def _coerce_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _normalize_stored_forecast_payloads(
    scenarios_raw: Any,
    features_raw: Any,
    signals_raw: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    scenarios = _coerce_json_list(scenarios_raw)
    features = _coerce_json_dict(features_raw)
    signals = _coerce_json_dict(signals_raw)

    # Backward-compatible recovery for legacy rows where signals and scenarios
    # were accidentally persisted in swapped columns.
    if not scenarios and not signals:
        alt_scenarios = _coerce_json_list(signals_raw)
        alt_signals = _coerce_json_dict(scenarios_raw)
        if alt_scenarios and alt_signals:
            return alt_scenarios, features, alt_signals

    return scenarios, features, signals


def _build_feedback(forecast_row: dict[str, Any], realized_return: float) -> str:
    low_return = float(forecast_row["low_return"])
    high_return = float(forecast_row["high_return"])
    expected_return = float(forecast_row["expected_return"])
    features = forecast_row.get("features", {}) or {}
    signals = forecast_row.get("signals", {}) or {}

    if low_return <= realized_return <= high_return:
        if abs(realized_return - expected_return) <= max(0.02, abs(high_return - low_return) * 0.25):
            return "Forecast range held and point estimate was close. The band was calibrated well."
        return "Forecast range held, but the point estimate was off. The scenario band was useful."

    realized_abs = abs(realized_return)
    band_abs = max(abs(low_return), abs(high_return), 0.01)
    if realized_abs > band_abs * 1.5:
        return "The move was larger than the band. Volatility and event risk were underweighted."

    if realized_return * expected_return < 0:
        return "Direction flipped. Trend reversal or a delayed event shock dominated the horizon."

    if abs(float(signals.get("event_bias", 0.0))) >= 0.35:
        return "Event bias mattered, but the mapping to price impact was still too uncertain."

    if abs(float(features.get("trend_strength", 0.0))) >= 1.0:
        return "Momentum was strong, but mean reversion or liquidity effects distorted the move."

    return "Forecast miss likely came from a missing catalyst, macro spillover, or regime shift."


def _save_forecast_evaluation(
    *,
    forecast_id: int,
    company_code: str,
    as_of: datetime,
    horizon_days: int,
    target_date: date,
    realized_at: date | None,
    base_price: float,
    realized_close: float,
    realized_return: float,
    expected_return: float,
    low_return: float,
    high_return: float,
    interval_hit: bool,
    scenario_hit: str,
    feedback: str,
    features: dict[str, Any],
    signals: dict[str, Any],
) -> None:
    evaluated_at = _now_utc()
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO price_forecast_evaluations(
                  forecast_id, company_code, as_of, horizon_days, target_date,
                  realized_at, base_price, realized_close, realized_return,
                  expected_return, abs_error, interval_hit, scenario_hit,
                  feedback, evaluated_at, extra
                ) VALUES (
                  %s,%s,%s,%s,%s,
                  %s,%s,%s,%s,
                  %s,%s,%s,%s,
                  %s,%s,%s::jsonb
                )
                ON CONFLICT (forecast_id) DO UPDATE SET
                  realized_at=EXCLUDED.realized_at,
                  realized_close=EXCLUDED.realized_close,
                  realized_return=EXCLUDED.realized_return,
                  expected_return=EXCLUDED.expected_return,
                  abs_error=EXCLUDED.abs_error,
                  interval_hit=EXCLUDED.interval_hit,
                  scenario_hit=EXCLUDED.scenario_hit,
                  feedback=EXCLUDED.feedback,
                  evaluated_at=EXCLUDED.evaluated_at,
                  extra=EXCLUDED.extra
                """,
                (
                    forecast_id,
                    company_code,
                    as_of,
                    horizon_days,
                    target_date,
                    realized_at,
                    base_price,
                    realized_close,
                    realized_return,
                    expected_return,
                    abs_error := abs(realized_return - expected_return),
                    interval_hit,
                    scenario_hit,
                    feedback,
                    evaluated_at,
                    json.dumps(
                        {
                            "low_return": low_return,
                            "high_return": high_return,
                            "features": features,
                            "signals": signals,
                        },
                        ensure_ascii=False,
                    ),
                ),
            )
        conn.commit()


def evaluate_due_forecasts(limit: int = 1000) -> list[dict[str, Any]]:
    sql = """
        SELECT
          f.id, f.company_code, f.as_of, f.horizon_days, f.target_date,
          f.base_price, f.expected_return, f.low_return, f.high_return, f.scenarios, f.features, f.signals
        FROM price_forecasts f
        LEFT JOIN price_forecast_evaluations e ON e.forecast_id = f.id
        WHERE e.forecast_id IS NULL
          AND f.target_date <= CURRENT_DATE
        ORDER BY f.as_of DESC
        LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    results: list[dict[str, Any]] = []
    for row in rows or []:
        (
            forecast_id,
            company_code,
            as_of,
            horizon_days,
            target_date,
            base_price,
            expected_return,
            low_return,
            high_return,
            scenarios_raw,
            features_raw,
            signals_raw,
        ) = row
        realized_at, realized_close = _closest_close_on_or_after(company_code, target_date)
        if realized_close is None:
            continue

        base_price_f = float(base_price)
        realized_return = (realized_close - base_price_f) / base_price_f if base_price_f else 0.0
        interval_hit = float(low_return) <= realized_return <= float(high_return)
        scenarios, features, signals = _normalize_stored_forecast_payloads(
            scenarios_raw,
            features_raw,
            signals_raw,
        )
        feedback = _build_feedback(
            {
                "low_return": float(low_return),
                "high_return": float(high_return),
                "expected_return": float(expected_return),
                "features": features,
                "signals": signals,
            },
            realized_return,
        )
        scenario_hit = _scenario_hit(realized_return, scenarios)
        abs_error = abs(realized_return - float(expected_return))
        evaluated_at = _now_utc()

        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO price_forecast_evaluations(
                      forecast_id, company_code, as_of, horizon_days, target_date,
                      realized_at, base_price, realized_close, realized_return,
                      expected_return, abs_error, interval_hit, scenario_hit,
                      feedback, evaluated_at, extra
                    ) VALUES (
                      %s,%s,%s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s,%s,
                      %s,%s,%s::jsonb
                    )
                    ON CONFLICT (forecast_id) DO UPDATE SET
                      realized_at=EXCLUDED.realized_at,
                      realized_close=EXCLUDED.realized_close,
                      realized_return=EXCLUDED.realized_return,
                      expected_return=EXCLUDED.expected_return,
                      abs_error=EXCLUDED.abs_error,
                      interval_hit=EXCLUDED.interval_hit,
                      scenario_hit=EXCLUDED.scenario_hit,
                      feedback=EXCLUDED.feedback,
                      evaluated_at=EXCLUDED.evaluated_at,
                      extra=EXCLUDED.extra
                    """,
                    (
                        forecast_id,
                        company_code,
                        as_of,
                        horizon_days,
                        target_date,
                        realized_at,
                        base_price,
                        realized_close,
                        realized_return,
                        expected_return,
                        abs_error,
                        interval_hit,
                        scenario_hit,
                        feedback,
                        evaluated_at,
                        json.dumps(
                            {
                                "low_return": float(low_return),
                                "high_return": float(high_return),
                                "features": features,
                                "signals": signals,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
            conn.commit()

        results.append(
            {
                "forecast_id": forecast_id,
                "company_code": company_code,
                "as_of": as_of.isoformat() if hasattr(as_of, "isoformat") else str(as_of),
                "horizon_days": horizon_days,
                "target_date": target_date.isoformat() if hasattr(target_date, "isoformat") else str(target_date),
                "realized_at": realized_at.isoformat() if hasattr(realized_at, "isoformat") else str(realized_at),
                "realized_return": round(realized_return, 4),
                "expected_return": round(float(expected_return), 4),
                "abs_error": round(abs_error, 4),
                "interval_hit": interval_hit,
                "scenario_hit": scenario_hit,
                "feedback": feedback,
            }
        )

    return results


def render_forecast_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# Price Forecast Summary", ""]
    if not rows:
        lines.append("No forecasts available.")
        return "\n".join(lines)
    for row in rows:
        lines.append(
            f"## {row['company_code']} | {row['horizon_days']}d | as_of={row['as_of']} | target={row['target_date']}"
        )
        lines.append(
            f"- expected_return={row['expected_return']:+.2%} | band=[{row['low_return']:+.2%}, {row['high_return']:+.2%}]"
        )
        lines.append(
            f"- expected_price={row['expected_price']:.2f} | band=[{row['low_price']:.2f}, {row['high_price']:.2f}]"
        )
        lines.append(f"- signals={json.dumps(row.get('signals', {}), ensure_ascii=False)}")
        lines.append(f"- features={json.dumps(row.get('features', {}), ensure_ascii=False)}")
        lines.append("")
    return "\n".join(lines)


def backtest_price_forecasts(
    company_codes: list[str] | None = None,
    horizons: list[int] | None = None,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    step_days: int = 7,
    persist: bool = True,
) -> dict[str, Any]:
    companies = [_norm_code(c) for c in (company_codes or DEFAULT_COMPANIES)]
    horizon_list = [int(h) for h in (horizons or DEFAULT_HORIZONS)]
    start_dt = start_date if isinstance(start_date, date) else date.fromisoformat(str(start_date)) if start_date else (date.today() - timedelta(days=180))
    end_dt = end_date if isinstance(end_date, date) else date.fromisoformat(str(end_date)) if end_date else date.today()
    step_days = max(1, int(step_days))

    rows: list[dict[str, Any]] = []
    for company in companies:
        trade_dates = _fetch_trade_dates(company, start_dt, end_dt)
        for idx, trade_date in enumerate(trade_dates):
            if idx % step_days != 0:
                continue
            as_of_dt = datetime.combine(trade_date, datetime.min.time(), tzinfo=timezone.utc)
            for horizon in horizon_list:
                forecast = _forecast_for_company(company, horizon, as_of_dt)
                if forecast is None:
                    continue
                forecast_id = _upsert_forecast(forecast) if persist else -1
                target_dt = date.fromisoformat(forecast.target_date)
                realized_at, realized_close = _closest_close_on_or_after(company, target_dt)
                if realized_close is None:
                    continue
                realized_return = (realized_close - forecast.base_price) / forecast.base_price if forecast.base_price else 0.0
                interval_hit = forecast.low_return <= realized_return <= forecast.high_return
                abs_error = abs(realized_return - forecast.expected_return)
                scenario_hit = _scenario_hit(realized_return, [asdict(x) for x in forecast.scenarios])
                feedback = _build_feedback(
                    {
                        "low_return": forecast.low_return,
                        "high_return": forecast.high_return,
                        "expected_return": forecast.expected_return,
                        "features": forecast.features,
                        "signals": forecast.signals,
                    },
                    realized_return,
                )
                if persist and forecast_id > 0:
                    _save_forecast_evaluation(
                        forecast_id=forecast_id,
                        company_code=company,
                        as_of=as_of_dt,
                        horizon_days=horizon,
                        target_date=target_dt,
                        realized_at=realized_at,
                        base_price=forecast.base_price,
                        realized_close=realized_close,
                        realized_return=realized_return,
                        expected_return=forecast.expected_return,
                        low_return=forecast.low_return,
                        high_return=forecast.high_return,
                        interval_hit=interval_hit,
                        scenario_hit=scenario_hit,
                        feedback=feedback,
                        features=forecast.features,
                        signals=forecast.signals,
                    )
                rows.append(
                    {
                        "company_code": company,
                        "company_name": forecast.company_name,
                        "as_of": forecast.as_of,
                        "horizon_days": horizon,
                        "target_date": forecast.target_date,
                        "realized_at": realized_at.isoformat() if hasattr(realized_at, "isoformat") else str(realized_at),
                        "expected_return": forecast.expected_return,
                        "realized_return": round(realized_return, 4),
                        "abs_error": round(abs_error, 4),
                        "interval_hit": interval_hit,
                        "scenario_hit": scenario_hit,
                        "feedback": feedback,
                        "forecast_id": forecast_id,
                        "low_return": forecast.low_return,
                        "high_return": forecast.high_return,
                    }
                )

    def _avg(values: list[float]) -> float:
        return round(_safe_mean(values, 0.0), 4) if values else 0.0

    def _hit_rate(items: list[dict[str, Any]]) -> float:
        return round(sum(1 for r in items if r["interval_hit"]) / len(items), 3) if items else 0.0

    def _direction_accuracy(items: list[dict[str, Any]]) -> float:
        if not items:
            return 0.0
        return round(
            sum(
                1 for r in items
                if (float(r["expected_return"]) > 0) == (float(r["realized_return"]) > 0)
            ) / len(items),
            3,
        )

    by_company: dict[str, dict[str, Any]] = {}
    by_horizon: dict[int, dict[str, Any]] = {}

    for company in sorted(set(r["company_code"] for r in rows)):
        company_rows = [r for r in rows if r["company_code"] == company]
        by_company[company] = {
            "count": len(company_rows),
            "avg_expected_return": _avg([float(r["expected_return"]) for r in company_rows]),
            "avg_realized_return": _avg([float(r["realized_return"]) for r in company_rows]),
            "avg_abs_error": _avg([float(r["abs_error"]) for r in company_rows]),
            "interval_hit_rate": _hit_rate(company_rows),
            "direction_accuracy": _direction_accuracy(company_rows),
        }

    for horizon in sorted(set(int(r["horizon_days"]) for r in rows)):
        horizon_rows = [r for r in rows if int(r["horizon_days"]) == horizon]
        by_horizon[horizon] = {
            "count": len(horizon_rows),
            "avg_expected_return": _avg([float(r["expected_return"]) for r in horizon_rows]),
            "avg_realized_return": _avg([float(r["realized_return"]) for r in horizon_rows]),
            "avg_abs_error": _avg([float(r["abs_error"]) for r in horizon_rows]),
            "interval_hit_rate": _hit_rate(horizon_rows),
            "direction_accuracy": _direction_accuracy(horizon_rows),
        }

    return {
        "cases": len(rows),
        "avg_expected_return": _avg([float(r["expected_return"]) for r in rows]),
        "avg_realized_return": _avg([float(r["realized_return"]) for r in rows]),
        "avg_abs_error": _avg([float(r["abs_error"]) for r in rows]),
        "interval_hit_rate": _hit_rate(rows),
        "direction_accuracy": _direction_accuracy(rows),
        "by_company": by_company,
        "by_horizon": by_horizon,
        "rows": rows[:200],
    }


def render_backtest_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Forecast Backtest Summary", ""]
    lines.append(f"- cases: {summary.get('cases', 0)}")
    lines.append(f"- avg_expected_return: {summary.get('avg_expected_return', 0.0):+.2%}")
    lines.append(f"- avg_realized_return: {summary.get('avg_realized_return', 0.0):+.2%}")
    lines.append(f"- avg_abs_error: {summary.get('avg_abs_error', 0.0):.2%}")
    lines.append(f"- interval_hit_rate: {summary.get('interval_hit_rate', 0.0):.1%}")
    lines.append(f"- direction_accuracy: {summary.get('direction_accuracy', 0.0):.1%}")
    lines.append("")
    lines.append("## By Company")
    lines.append("| Company | Count | Expected | Realized | Abs Error | Hit Rate | Dir Acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for company, stats in (summary.get("by_company") or {}).items():
        lines.append(
            f"| {company} | {stats.get('count', 0)} | {float(stats.get('avg_expected_return', 0.0)):+.2%} | "
            f"{float(stats.get('avg_realized_return', 0.0)):+.2%} | {float(stats.get('avg_abs_error', 0.0)):.2%} | "
            f"{float(stats.get('interval_hit_rate', 0.0)):.1%} | {float(stats.get('direction_accuracy', 0.0)):.1%} |"
        )
    lines.append("")
    lines.append("## By Horizon")
    lines.append("| Horizon | Count | Expected | Realized | Abs Error | Hit Rate | Dir Acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for horizon, stats in (summary.get("by_horizon") or {}).items():
        lines.append(
            f"| {horizon}d | {stats.get('count', 0)} | {float(stats.get('avg_expected_return', 0.0)):+.2%} | "
            f"{float(stats.get('avg_realized_return', 0.0)):+.2%} | {float(stats.get('avg_abs_error', 0.0)):.2%} | "
            f"{float(stats.get('interval_hit_rate', 0.0)):.1%} | {float(stats.get('direction_accuracy', 0.0)):.1%} |"
        )
    lines.append("")
    lines.append("## Sample Misses")
    for row in sorted(summary.get("rows", []), key=lambda r: abs(float(r.get("abs_error", 0.0))), reverse=True)[:10]:
        lines.append(
            f"- {row.get('company_code')} {row.get('horizon_days')}d as_of={row.get('as_of')} "
            f"expected={float(row.get('expected_return', 0.0)):+.2%} realized={float(row.get('realized_return', 0.0)):+.2%} "
            f"error={float(row.get('abs_error', 0.0)):.2%} | {row.get('feedback', '')}"
        )
    return "\n".join(lines)
