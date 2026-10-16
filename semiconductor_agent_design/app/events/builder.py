"""
Build event candidates and post-event outcome labels.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import date, datetime, timezone
from typing import Any

from app.agent.semiconductor_event_utils import classify_event_type, classify_technology_category
from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

COMPANIES = {
    "005930": ["samsung", "삼성전자", "삼성", "samsung electronics"],
    "000660": ["sk hynix", "sk하이닉스", "하이닉스", "hynix"],
    "042700": ["hanmi semiconductor", "한미반도체", "hanmi"],
}

GLOBAL_COMPANY_HINTS = {
    "NVDA": ["nvidia", "gpu", "blackwell", "rubin"],
    "TSM": ["tsmc", "cowos", "foundry"],
    "ASML": ["asml", "euv", "high-na"],
    "AMAT": ["applied materials", "amat"],
    "LRCX": ["lam research", "lam"],
    "MU": ["micron"],
}

DOMAIN_KEYWORDS = {
    "hbm": ["hbm", "hbm3", "hbm3e", "hbm4", "high bandwidth memory"],
    "packaging": ["advanced packaging", "cowos", "hybrid bonding", "tc bonding", "tsv", "chiplet"],
    "litho": ["euv", "high-na", "lithography", "asml"],
    "nand": ["nand", "3d nand", "flash"],
    "dram": ["dram", "ddr5", "lpddr", "memory"],
    "logic": ["gaa", "finfet", "nanosheet", "foundry"],
    "financials": ["earnings", "revenue", "guidance", "sales", "results", "공시", "실적"],
}

EVENT_KEYWORDS = [
    "hbm", "hbm3", "hbm3e", "hbm4", "euv", "gaa", "cowos", "hybrid bonding",
    "advanced packaging", "revenue", "earnings", "guidance", "capex", "supply",
    "양산", "수주", "공급", "실적", "투자", "증설", "매출", "영업이익",
]

SOURCE_TIER_BY_SOURCE = {
    "dart": 1,
    "samsung_global_newsroom": 1,
    "skhynix_newsroom": 1,
    "tsmc_monthly_revenue": 1,
    "asml_press_releases": 1,
    "nvidia_ir_rss": 1,
    "micron_ir": 1,
    "applied_materials_ir": 1,
    "arxiv": 2,
    "openalex": 2,
    "semantic_scholar": 2,
    "semiconductor_engineering": 3,
    "eetimes": 3,
    "ieee_spectrum": 3,
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _event_id(*parts: object) -> str:
    raw = "|".join(str(p or "") for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _source_tier(source: str, extra: dict[str, Any] | None = None) -> int:
    if extra and extra.get("source_tier"):
        try:
            return max(1, min(4, int(extra["source_tier"])))
        except Exception:
            pass
    return SOURCE_TIER_BY_SOURCE.get(source, 3)


def _infer_domain(title: str, summary: str, tags: list[str] | None = None) -> str:
    text = f"{title} {summary} {' '.join(tags or [])}".lower()
    best = "general"
    best_count = 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        count = sum(1 for kw in kws if kw in text)
        if count > best_count:
            best = domain
            best_count = count
    return best


def _infer_company(title: str, summary: str, tags: list[str] | None, extra: dict[str, Any] | None) -> str | None:
    if extra:
        c = extra.get("company_code") or extra.get("company")
        if c:
            return str(c)
    text = f"{title} {summary} {' '.join(tags or [])}".lower()
    for code, hints in COMPANIES.items():
        if any(h in text for h in hints):
            return code
    for code, hints in GLOBAL_COMPANY_HINTS.items():
        if any(h in text for h in hints):
            return code
    return None


def _is_event_like(title: str, summary: str, source_type: str) -> bool:
    text = f"{title} {summary}".lower()
    if source_type in {"company_official", "paper", "rss_news", "tech_blog", "conference_metadata"}:
        return any(k in text for k in EVENT_KEYWORDS)
    return source_type in {"patent", "dart", "disclosure"}


def _upsert_event(
    *,
    event_id: str,
    event_date: datetime,
    event_type: str,
    source: str,
    source_tier: int,
    title: str,
    summary: str,
    related_company: str | None,
    related_domain: str,
    evidence_doc_uid: str | None,
    confidence: float,
    extra: dict[str, Any],
) -> bool:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO event_candidates(
                  event_id, event_date, event_type, source, source_tier,
                  title, summary, related_company, related_domain, evidence_doc_uid,
                  confidence, status, created_at, extra
                ) VALUES (
                  %s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s,
                  %s,'new',%s,%s::jsonb
                )
                ON CONFLICT (event_id) DO UPDATE SET
                  title=EXCLUDED.title,
                  summary=EXCLUDED.summary,
                  related_company=EXCLUDED.related_company,
                  related_domain=EXCLUDED.related_domain,
                  confidence=EXCLUDED.confidence,
                  extra=event_candidates.extra || EXCLUDED.extra
                """,
                (
                    event_id, event_date, event_type, source, source_tier,
                    title, summary, related_company, related_domain, evidence_doc_uid,
                    confidence, _now_utc(), json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()
    return True


def build_events_from_documents(limit: int = 1000) -> int:
    """Promote high-signal tech_documents to event_candidates."""
    sql = """
        SELECT doc_uid, source, source_type, title, url, published_at, collected_at,
               summary, tags, confidence, extra
        FROM tech_documents
        ORDER BY COALESCE(published_at, collected_at) DESC
        LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    inserted = 0
    for row in rows:
        doc_uid, source, source_type, title, url, published_at, collected_at, summary, tags, confidence, extra = row
        title = title or ""
        summary = summary or ""
        tags = list(tags or [])
        extra = dict(extra or {})
        if not _is_event_like(title, summary, source_type):
            continue
        event_date = published_at or collected_at or _now_utc()
        company = _infer_company(title, summary, tags, extra)
        domain = _infer_domain(title, summary, tags)
        tier = _source_tier(source, extra)
        score = min(0.98, float(confidence or 0.70) * (1.0 if tier == 1 else 0.92 if tier == 2 else 0.78))
        eid = _event_id("doc", doc_uid, company, domain)
        _upsert_event(
            event_id=eid,
            event_date=event_date,
            event_type=classify_event_type(f"{title} {summary}", domain),
            source=source,
            source_tier=tier,
            title=title,
            summary=summary[:2000],
            related_company=company,
            related_domain=domain,
            evidence_doc_uid=doc_uid,
            confidence=score,
            extra={
                "url": url,
                "tags": tags,
                "origin": "tech_documents",
                "technology_category": classify_technology_category(f"{title} {summary}", domain),
            },
        )
        inserted += 1
    logger.info("build_events_from_documents done | candidates=%d", inserted)
    return inserted


def build_events_from_disclosures(limit: int = 1000) -> int:
    """Promote DART disclosures to event_candidates."""
    sql = """
        SELECT company_code, rcept_no, report_type, title, published_at, extracted
        FROM disclosures
        ORDER BY published_at DESC
        LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            rows = cur.fetchall()

    inserted = 0
    for company_code, rcept_no, report_type, title, published_at, extracted in rows:
        title = title or report_type or rcept_no
        summary = json.dumps(extracted or {}, ensure_ascii=False)[:2000]
        domain = _infer_domain(title, summary, ["dart", "financials"])
        eid = _event_id("dart", rcept_no, company_code)
        _upsert_event(
            event_id=eid,
            event_date=published_at,
            event_type="disclosure",
            source="dart",
            source_tier=1,
            title=title,
            summary=summary,
            related_company=company_code,
            related_domain=domain,
            evidence_doc_uid=None,
            confidence=0.95,
            extra={"rcept_no": rcept_no, "report_type": report_type, "origin": "disclosures"},
        )
        inserted += 1
    logger.info("build_events_from_disclosures done | candidates=%d", inserted)
    return inserted


def _price_after(cur, company_code: str, start_date: date, offset: int) -> tuple[date, float, float | None] | None:
    cur.execute(
        """
        SELECT trade_date, close, volume
        FROM price_daily
        WHERE company_code=%s AND trade_date >= %s
        ORDER BY trade_date ASC
        OFFSET %s LIMIT 1
        """,
        (company_code, start_date, offset),
    )
    row = cur.fetchone()
    if not row:
        return None
    return row[0], float(row[1]), float(row[2]) if row[2] is not None else None


def label_event_outcomes(limit: int = 5000, positive_threshold: float = 0.03, negative_threshold: float = -0.03) -> int:
    """Compute post-event returns for event candidates with company price data."""
    sql = """
        SELECT event_id, related_company, event_date
        FROM event_candidates
        WHERE related_company IS NOT NULL
        ORDER BY event_date DESC
        LIMIT %s
    """
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (limit,))
            events = cur.fetchall()

            upserted = 0
            for event_id, company_code, event_dt in events:
                if company_code not in COMPANIES:
                    continue
                start_date = event_dt.date() if hasattr(event_dt, "date") else event_dt
                base = _price_after(cur, company_code, start_date, 0)
                if not base:
                    continue
                _, base_close, base_vol = base

                def ret_at(days: int) -> float | None:
                    target = _price_after(cur, company_code, start_date, days)
                    if not target or base_close == 0:
                        return None
                    return (target[1] - base_close) / base_close

                ret_1d = ret_at(1)
                ret_5d = ret_at(5)
                ret_20d = ret_at(20)
                ret_60d = ret_at(60)
                vol_target = _price_after(cur, company_code, start_date, 5)
                vol_change = None
                if base_vol and vol_target and vol_target[2] is not None:
                    vol_change = (vol_target[2] - base_vol) / base_vol if base_vol else None
                label_base = ret_5d if ret_5d is not None else ret_20d
                label = None
                if label_base is not None:
                    if label_base >= positive_threshold:
                        label = "positive"
                    elif label_base <= negative_threshold:
                        label = "negative"
                    else:
                        label = "neutral"
                cur.execute(
                    """
                    INSERT INTO event_outcomes(
                      event_id, related_company, event_date,
                      ret_1d, ret_5d, ret_20d, ret_60d,
                      volume_change_5d, label, computed_at, extra
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (event_id) DO UPDATE SET
                      ret_1d=EXCLUDED.ret_1d,
                      ret_5d=EXCLUDED.ret_5d,
                      ret_20d=EXCLUDED.ret_20d,
                      ret_60d=EXCLUDED.ret_60d,
                      volume_change_5d=EXCLUDED.volume_change_5d,
                      label=EXCLUDED.label,
                      computed_at=EXCLUDED.computed_at
                    """,
                    (
                        event_id, company_code, start_date,
                        ret_1d, ret_5d, ret_20d, ret_60d,
                        vol_change, label, _now_utc(),
                        json.dumps({"positive_threshold": positive_threshold, "negative_threshold": negative_threshold}),
                    ),
                )
                upserted += 1
        conn.commit()
    logger.info("label_event_outcomes done | upserted=%d", upserted)
    return upserted


def build_event_dataset() -> dict[str, int]:
    docs = build_events_from_documents()
    disclosures = build_events_from_disclosures()
    outcomes = label_event_outcomes()
    return {"document_events": docs, "disclosure_events": disclosures, "outcomes": outcomes}
