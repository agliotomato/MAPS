import hashlib
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import feedparser

from app.db.postgres import get_pg_conn
from app.taxonomy import taxonomy_collection

logger = logging.getLogger(__name__)

SEMICONDUCTOR_KEYWORDS = [
    "semiconductor",
    "chip",
    "hbm",
    "dram",
    "nand",
    "euv",
    "lithography",
    "foundry",
    "packaging",
    "tsv",
    "thermo-compression",
    "bonding",
    "yield",
    "cmos",
    "finfet",
    "gaa",
    "nanosheet",
    "cowos",
    "cxl",
    "pim",
    "processing-in-memory",
    "ucie",
    "chiplet",
    "interposer",
    "hybrid bonding",
    "high-na",
    "backside power",
    "bpdn",
    "glass substrate",
    "sic",
    "gan",
    "power semiconductor",
    "rfic",
    "pmic",
    "cis",
    "eda",
    "mpw",
    "metrology",
    "inspection",
    "reliability",
    "electromigration",
    "tem",
    "fib",
    "jedec",
    "irds",
    "semi",
    "wsts",
    "capex",
    "backlog",
    "roadmap",
    "standard",
]

RSS_NEWS_SOURCES = {
    "semiconductor_engineering": "https://semiengineering.com/feed",
    "eetimes": "https://www.eetimes.com/feed/",
    "ieee_spectrum": "https://spectrum.ieee.org/feeds/feed.rss",
}

TECH_BLOG_SOURCES = {
    "chips_and_cheese": "https://chipsandcheese.substack.com/feed",
    "real_world_tech": "https://www.realworldtech.com/index.xml",
    "skhynix_newsroom": "https://news.skhynix.com/feed/",
}

_TAXONOMY_COLLECTION = taxonomy_collection()
if _TAXONOMY_COLLECTION:
    SEMICONDUCTOR_KEYWORDS = list(dict.fromkeys((SEMICONDUCTOR_KEYWORDS or []) + (_TAXONOMY_COLLECTION.get("news_keywords") or [])))


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _has_semiconductor_signal(title: str, summary: str) -> tuple[bool, list[str]]:
    text = f"{title} {summary}".lower()
    matched = [kw for kw in SEMICONDUCTOR_KEYWORDS if kw in text]
    return (len(matched) > 0), matched


def _make_doc_uid(source: str, url: str, published_at: datetime | None) -> str:
    base = f"{source}|{url}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _upsert_doc(
    source: str,
    source_type: str,
    title: str,
    url: str,
    published_at: datetime | None,
    summary: str,
    tags: list[str],
    confidence: float,
    extra: dict,
) -> None:
    doc_uid = _make_doc_uid(source, url, published_at)
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tech_documents(
                  doc_uid, source, source_type, title, url, published_at, collected_at,
                  summary, content, tags, confidence, extra
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (doc_uid)
                DO UPDATE SET
                  title = EXCLUDED.title,
                  summary = EXCLUDED.summary,
                  tags = EXCLUDED.tags,
                  confidence = EXCLUDED.confidence,
                  extra = EXCLUDED.extra,
                  collected_at = EXCLUDED.collected_at
                """,
                (
                    doc_uid,
                    source,
                    source_type,
                    title,
                    url,
                    published_at,
                    _now_utc(),
                    summary,
                    None,
                    tags,
                    confidence,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()


def _collect_from_sources(sources: dict[str, str], source_type: str, confidence: float) -> None:
    inserted = 0
    for source, url in sources.items():
        logger.info("RSS collect start | source=%s url=%s", source, url)
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
        logger.info("RSS parsed | source=%s entries=%s", source, len(entries))

        for entry in entries:
            title = str(getattr(entry, "title", "") or "").strip()
            link = str(getattr(entry, "link", "") or "").strip()
            summary = str(getattr(entry, "summary", "") or "").strip()
            published_raw = str(getattr(entry, "published", "") or "").strip()
            published_at = _to_dt(published_raw)
            ok, tags = _has_semiconductor_signal(title, summary)
            if not ok:
                continue
            if not link:
                continue

            _upsert_doc(
                source=source,
                source_type=source_type,
                title=title or link,
                url=link,
                published_at=published_at,
                summary=summary,
                tags=tags,
                confidence=confidence,
                extra={"published_raw": published_raw},
            )
            inserted += 1

        logger.info("RSS source done | source=%s inserted=%s", source, inserted)

    logger.info("RSS collect done | source_type=%s inserted_total=%s", source_type, inserted)


def collect_rss_all_sources() -> None:
    _collect_from_sources(RSS_NEWS_SOURCES, source_type="rss_news", confidence=0.70)


def collect_tech_blogs() -> None:
    _collect_from_sources(TECH_BLOG_SOURCES, source_type="tech_blog", confidence=0.75)
