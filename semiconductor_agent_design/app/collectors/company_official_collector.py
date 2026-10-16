"""
Official company IR/news collector.

These sources are high-trust inputs for the agent. They are stored as
tech_documents and chunked immediately so the RAG layer can cite them.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from io import BytesIO
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import feedparser
import httpx

from app.db.postgres import get_pg_conn
from app.taxonomy import taxonomy_collection

logger = logging.getLogger(__name__)


OFFICIAL_RSS_SOURCES: list[dict[str, Any]] = [
    {
        "source": "samsung_global_newsroom",
        "company_code": "005930",
        "url": "https://news.samsung.com/global/feed",
        "tags": ["official", "samsung", "press_release", "ir"],
        "confidence": 0.92,
    },
    {
        "source": "skhynix_newsroom",
        "company_code": "000660",
        "url": "https://news.skhynix.com/feed/",
        "tags": ["official", "sk_hynix", "press_release", "ir"],
        "confidence": 0.92,
        "fallback_urls": [
            "https://news.skhynix.com/press-center/press-release/",
            "https://news.skhynix.com/insight/",
        ],
    },
    {
        "source": "nvidia_ir_rss",
        "company_code": "NVDA",
        "url": "https://investor.nvidia.com/investor-resources/rss/default.aspx",
        "tags": ["official", "nvidia", "ir", "ai_demand"],
        "confidence": 0.88,
        "fallback_urls": [
            "https://nvidianews.nvidia.com/rss",
        ],
    },
    {
        "source": "nvidia_newsroom",
        "company_code": "NVDA",
        "url": "https://www.nvidia.com/en-us/about-nvidia/rss/",
        "tags": ["official", "nvidia", "press_release", "ai_demand"],
        "confidence": 0.86,
    },
    {
        "source": "nvidia_blog_feed",
        "company_code": "NVDA",
        "url": "https://blogs.nvidia.com/feed/",
        "tags": ["official", "nvidia", "blog", "ai_demand"],
        "confidence": 0.84,
    },
    {
        "source": "nvidia_developer_blog",
        "company_code": "NVDA",
        "url": "https://developer.nvidia.com/blog/feed/",
        "tags": ["official", "nvidia", "developer_blog", "ai_demand"],
        "confidence": 0.82,
    },
    {
        "source": "micron_ir",
        "company_code": "MU",
        "url": "https://micron.gcs-web.com/rss/news-releases.xml",
        "tags": ["official", "micron", "memory", "ir"],
        "confidence": 0.88,
        "fallback_urls": [
            "https://investors.micron.com/",
            "https://investors.micron.com/news-releases",
        ],
    },
    {
        "source": "micron_insight_feed",
        "company_code": "MU",
        "url": "https://www.micron.com/insight/feed",
        "tags": ["official", "micron", "insight", "memory"],
        "confidence": 0.82,
    },
    {
        "source": "applied_materials_ir",
        "company_code": "AMAT",
        "url": "https://ir.appliedmaterials.com/rss/news-releases.xml",
        "tags": ["official", "amat", "equipment", "ir"],
        "confidence": 0.86,
        "fallback_urls": [
            "https://ir.appliedmaterials.com/news-releases",
            "https://www.appliedmaterials.com/us/en/newsroom.html",
        ],
    },
]


HTML_SOURCES: list[dict[str, Any]] = [
    {
        "source": "tsmc_monthly_revenue",
        "company_code": "TSM",
        "url": "https://investor.tsmc.com/english/monthly-revenue/2026",
        "tags": ["official", "tsmc", "monthly_revenue", "foundry", "ai_demand"],
        "confidence": 0.90,
        "scrape_mode": "table",
        "fallback_urls": [
            "https://investor.tsmc.com/english/monthly-revenue/2025",
            "https://investor.tsmc.com/english/financial-calendar",
        ],
    },
    {
        "source": "asml_press_releases",
        "company_code": "ASML",
        "url": "https://www.asml.com/en/news/press-releases",
        "tags": ["official", "asml", "equipment", "lithography"],
        "confidence": 0.88,
        "scrape_mode": "links",
    },
    {
        "source": "skhynix_press_center",
        "company_code": "000660",
        "url": "https://news.skhynix.com/press-center/press-release/",
        "tags": ["official", "sk_hynix", "press_release", "ir"],
        "confidence": 0.90,
        "scrape_mode": "links",
    },
    {
        "source": "nvidia_newsroom_home",
        "company_code": "NVDA",
        "url": "https://www.nvidia.com/en-us/about-nvidia/",
        "tags": ["official", "nvidia", "newsroom", "ai_demand"],
        "confidence": 0.84,
        "scrape_mode": "links",
    },
    {
        "source": "micron_investor_home",
        "company_code": "MU",
        "url": "https://investors.micron.com/",
        "tags": ["official", "micron", "newsroom", "memory"],
        "confidence": 0.84,
        "scrape_mode": "links",
    },
    {
        "source": "applied_materials_newsroom",
        "company_code": "AMAT",
        "url": "https://www.appliedmaterials.com/us/en/insights.html",
        "tags": ["official", "amat", "insights", "equipment"],
        "confidence": 0.84,
        "scrape_mode": "links",
    },
]


PDF_SOURCES: list[dict[str, Any]] = [
    {
        "source": "tsmc_1q26_earnings_release",
        "company_code": "TSM",
        "url": "https://investor.tsmc.com/english/encrypt/files/encrypt_file/qr/phase4_reports/2026-04/bd8eb0403902fdea59a2f5e390e48d010b50edc9/1Q26%20EarningsRelease_WoG.pdf",
        "tags": ["official", "tsmc", "earnings_release", "foundry", "ai_demand"],
        "confidence": 0.93,
    },
    {
        "source": "tsmc_1q26_management_report",
        "company_code": "TSM",
        "url": "https://investor.tsmc.com/schinese/encrypt/files/encrypt_file/qr/phase4_reports/2026-04/9f060092ba29ff3630cfdaefd67774026195e135/1Q26ManagementReport.pdf",
        "tags": ["official", "tsmc", "management_report", "foundry", "ai_demand"],
        "confidence": 0.93,
    },
    {
        "source": "tsmc_4q25_earnings_release",
        "company_code": "TSM",
        "url": "https://investor.tsmc.com/english/encrypt/files/encrypt_file/reports/2026-01/3e49621566a3ca53bdf8aee2586929b666c17fd6/4Q25EarningsRelease.pdf",
        "tags": ["official", "tsmc", "earnings_release", "foundry", "ai_demand"],
        "confidence": 0.93,
    },
]


DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "hbm": ["hbm", "high bandwidth memory", "hbm3", "hbm3e", "hbm4"],
    "packaging": ["advanced packaging", "cowos", "chiplet", "hybrid bonding", "tc bonding", "tsv", "interposer", "glass substrate", "wlp", "fan-out", "bpdn"],
    "litho": ["euv", "high-na", "lithography", "asml"],
    "memory": ["dram", "nand", "memory", "ddr5", "lpddr", "micron", "sk hynix", "cxl", "pim", "wide io", "mram", "reram"],
    "ai_demand": ["ai", "accelerated computing", "data center", "gpu", "nvidia"],
    "equipment": ["equipment", "wafer", "deposition", "etch", "inspection", "metrology", "pellicle", "scanner", "stepper", "ald", "ale"],
    "financials": ["results", "revenue", "earnings", "guidance", "sales"],
    "materials": ["photoresist", "substrate", "silicon", "glass substrate", "etch gas", "cmp slurry", "precursor"],
    "power": ["sic", "gan", "power semiconductor", "mosfet", "igbt"],
    "standards": ["jedec", "irds", "ucie", "cxl", "semi"],
    "reliability": ["yield", "reliability", "electromigration", "failure analysis", "defect", "metrology"],
    "design": ["eda", "pdk", "mpw", "fabless", "design house"],
}

_TAXONOMY_COLLECTION = taxonomy_collection()
if _TAXONOMY_COLLECTION:
    domain_keywords = _TAXONOMY_COLLECTION.get("domain_keywords") or {}
    merged = dict(DOMAIN_KEYWORDS)
    for key, values in domain_keywords.items():
        if not isinstance(values, list):
            continue
        merged.setdefault(key, [])
        merged[key] = list(dict.fromkeys(merged[key] + values))
    DOMAIN_KEYWORDS = merged


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
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None


def _doc_uid(source: str, url: str, published_at: datetime | None) -> str:
    raw = f"{source}|{url}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _sanitize(text: str | None) -> str:
    return (text or "").replace("\x00", "").strip()


def _domain_hits(title: str, summary: str) -> list[str]:
    text = f"{title} {summary}".lower()
    hits = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in text for k in keywords):
            hits.append(domain)
    return hits or ["general"]


def _chunk_text(text: str, size: int = 1000, overlap: int = 120) -> list[str]:
    text = _sanitize(text)
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _doc_exists(doc_uid: str) -> bool:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM tech_documents WHERE doc_uid=%s LIMIT 1", (doc_uid,))
            return cur.fetchone() is not None


def _ensure_company(company_code: str) -> None:
    names = {
        "005930": "Samsung Electronics",
        "000660": "SK hynix",
        "042700": "Hanmi Semiconductor",
        "NVDA": "NVIDIA",
        "TSM": "TSMC",
        "MU": "Micron Technology",
        "ASML": "ASML",
        "AMAT": "Applied Materials",
    }
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO companies(company_code, company_name, market, country)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (company_code) DO UPDATE SET company_name=EXCLUDED.company_name
                """,
                (company_code, names.get(company_code, company_code), "GLOBAL", ""),
            )
        conn.commit()


def _upsert_doc_and_chunks(
    *,
    source: str,
    source_type: str,
    company_code: str,
    title: str,
    url: str,
    published_at: datetime | None,
    summary: str,
    tags: list[str],
    confidence: float,
    extra: dict[str, Any],
) -> bool:
    if not url:
        return False
    _ensure_company(company_code)
    domains = _domain_hits(title, summary)
    doc_uid = _doc_uid(source, url, published_at)
    if _doc_exists(doc_uid) and not _env_bool("OFFICIAL_FORCE_REFRESH", False):
        return False

    content = _sanitize(f"{title}\n\n{summary}")
    all_tags = sorted(set(tags + domains + [company_code, "official"]))
    extra = {
        **extra,
        "company_code": company_code,
        "source_tier": 1,
        "domain_hits": domains,
    }

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tech_documents(
                  doc_uid, source, source_type, title, url, published_at, collected_at,
                  summary, content, tags, confidence, extra
                ) VALUES (
                  %s,%s,%s,%s,%s,%s,%s,
                  %s,%s,%s,%s,%s::jsonb
                )
                ON CONFLICT (doc_uid) DO UPDATE SET
                  title=EXCLUDED.title,
                  summary=EXCLUDED.summary,
                  content=EXCLUDED.content,
                  tags=EXCLUDED.tags,
                  confidence=EXCLUDED.confidence,
                  extra=tech_documents.extra || EXCLUDED.extra,
                  collected_at=EXCLUDED.collected_at
                """,
                (
                    doc_uid,
                    source,
                    source_type,
                    _sanitize(title) or url,
                    url,
                    published_at,
                    _now_utc(),
                    _sanitize(summary)[:3000],
                    content,
                    all_tags,
                    confidence,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
            cur.execute("DELETE FROM tech_document_chunks WHERE doc_uid=%s", (doc_uid,))
            for idx, chunk in enumerate(_chunk_text(content)):
                cur.execute(
                    """
                    INSERT INTO tech_document_chunks(
                      doc_uid, chunk_index, chunk_text, char_len, token_estimate, created_at, extra
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                    """,
                    (
                        doc_uid,
                        idx,
                        chunk,
                        len(chunk),
                        max(1, len(chunk) // 4),
                        _now_utc(),
                        json.dumps({"source": source, "company_code": company_code}, ensure_ascii=False),
                    ),
                )
        conn.commit()
    return True


def _collect_rss_source(cfg: dict[str, Any]) -> int:
    urls = [cfg["url"]] + list(cfg.get("fallback_urls", []) or [])
    inserted_total = 0
    for url in urls:
        try:
            with httpx.Client(
                timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20")),
                follow_redirects=True,
                headers={"User-Agent": "SemiconductorAgentBot/1.0"},
            ) as client:
                resp = client.get(url)
                resp.raise_for_status()
            feed = feedparser.parse(resp.text)
        except Exception as ex:
            logger.warning("Official RSS fetch failed source=%s url=%s error=%s", cfg["source"], url, ex)
            continue

        entries = getattr(feed, "entries", []) or []
        inserted = 0
        for e in entries:
            title = _sanitize(getattr(e, "title", ""))
            link = _sanitize(getattr(e, "link", ""))
            summary = _sanitize(getattr(e, "summary", "") or getattr(e, "description", ""))
            published_raw = _sanitize(getattr(e, "published", "") or getattr(e, "updated", ""))
            published_at = _to_dt(published_raw)
            if not title and not link:
                continue
            try:
                ok = _upsert_doc_and_chunks(
                    source=cfg["source"],
                    source_type="company_official",
                    company_code=cfg["company_code"],
                    title=title or link,
                    url=link,
                    published_at=published_at,
                    summary=summary,
                    tags=cfg.get("tags", []),
                    confidence=float(cfg.get("confidence", 0.85)),
                    extra={"published_raw": published_raw, "feed_url": url},
                )
                if ok:
                    inserted += 1
            except Exception as ex:
                logger.warning("Official RSS upsert failed source=%s link=%s error=%s", cfg["source"], link, ex)

        inserted_total += inserted
        logger.info("Official RSS done | source=%s url=%s entries=%d inserted=%d", cfg["source"], url, len(entries), inserted)
        if inserted > 0:
            break
    return inserted_total


def _collect_html_source(cfg: dict[str, Any]) -> int:
    try:
        from bs4 import BeautifulSoup
    except Exception as ex:
        logger.warning("BeautifulSoup unavailable source=%s error=%s", cfg["source"], ex)
        return 0

    urls = [cfg["url"]] + list(cfg.get("fallback_urls", []) or [])
    mode = str(cfg.get("scrape_mode", "links")).strip().lower()
    inserted_total = 0

    for url in urls:
        try:
            with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20")), follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "SemiconductorAgentBot/1.0"})
                resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as ex:
            logger.warning("Official HTML fetch failed source=%s url=%s error=%s", cfg["source"], url, ex)
            continue

        inserted = 0
        if mode == "table":
            page_title = _sanitize((soup.title.get_text(" ", strip=True) if soup.title else "") or cfg["source"] or url)
            tables: list[str] = []
            for table in soup.find_all("table")[:5]:
                rows: list[str] = []
                for tr in table.find_all("tr"):
                    cells = [
                        _sanitize(cell.get_text(" ", strip=True))
                        for cell in tr.find_all(["th", "td"])
                    ]
                    cells = [cell for cell in cells if cell]
                    if cells:
                        rows.append(" | ".join(cells))
                if rows:
                    tables.append("\n".join(rows))

            if tables:
                summary = "\n\n".join(tables[:2])
                content = f"{page_title}\n\n{summary}"
                try:
                    ok = _upsert_doc_and_chunks(
                        source=cfg["source"],
                        source_type="company_official",
                        company_code=cfg["company_code"],
                        title=page_title,
                        url=url,
                        published_at=None,
                        summary=summary[:3000],
                        tags=cfg.get("tags", []),
                        confidence=float(cfg.get("confidence", 0.85)),
                        extra={"page_url": url, "scrape_method": "table", "table_count": len(tables)},
                    )
                    if ok:
                        inserted += 1
                except Exception as ex:
                    logger.warning("Official table upsert failed source=%s url=%s error=%s", cfg["source"], url, ex)
            else:
                logger.info("Official HTML table mode found no tables | source=%s url=%s", cfg["source"], url)
        else:
            candidates = []
            for a in soup.find_all("a", href=True):
                text = _sanitize(a.get_text(" ", strip=True))
                href = str(a.get("href", "")).strip()
                if len(text) < 12:
                    continue
                if not re.search(r"revenue|results|press|financial|hbm|ai|euv|sales|monthly|report|news", text, re.I):
                    continue
                if href.startswith("/"):
                    base = re.match(r"^(https?://[^/]+)", url)
                    href = (base.group(1) if base else url.rstrip("/")) + href
                if href.startswith("http"):
                    candidates.append((text, href))

            seen = set()
            for title, link in candidates[:40]:
                if link in seen:
                    continue
                seen.add(link)
                try:
                    ok = _upsert_doc_and_chunks(
                        source=cfg["source"],
                        source_type="company_official",
                        company_code=cfg["company_code"],
                        title=title,
                        url=link,
                        published_at=None,
                        summary=title,
                        tags=cfg.get("tags", []),
                        confidence=float(cfg.get("confidence", 0.85)),
                        extra={"page_url": url, "scrape_method": "link_title"},
                    )
                    if ok:
                        inserted += 1
                except Exception as ex:
                    logger.warning("Official HTML upsert failed source=%s link=%s error=%s", cfg["source"], link, ex)
            logger.info("Official HTML done | source=%s url=%s candidates=%d inserted=%d", cfg["source"], url, len(candidates), inserted)

        inserted_total += inserted
        if inserted > 0:
            break
    return inserted_total


def _collect_pdf_source(cfg: dict[str, Any]) -> int:
    try:
        from pypdf import PdfReader
    except Exception as ex:
        logger.warning("PDF parser unavailable source=%s error=%s", cfg["source"], ex)
        return 0

    urls = [cfg["url"]] + list(cfg.get("fallback_urls", []) or [])
    inserted_total = 0
    for url in urls:
        try:
            with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "30")), follow_redirects=True) as client:
                resp = client.get(url, headers={"User-Agent": "SemiconductorAgentBot/1.0"})
                resp.raise_for_status()
            reader = PdfReader(BytesIO(resp.content))
            texts: list[str] = []
            for page in reader.pages[:8]:
                try:
                    texts.append(_sanitize(page.extract_text() or ""))
                except Exception:
                    continue
            text = "\n\n".join(t for t in texts if t)
        except Exception as ex:
            logger.warning("Official PDF fetch failed source=%s url=%s error=%s", cfg["source"], url, ex)
            continue

        if not text:
            logger.info("Official PDF empty source=%s url=%s", cfg["source"], url)
            continue

        title = cfg.get("title") or cfg["source"]
        summary = text[:3000]
        try:
            ok = _upsert_doc_and_chunks(
                source=cfg["source"],
                source_type="company_official",
                company_code=cfg["company_code"],
                title=title,
                url=url,
                published_at=None,
                summary=summary,
                tags=cfg.get("tags", []),
                confidence=float(cfg.get("confidence", 0.85)),
                extra={"source_url": url, "scrape_method": "pdf", "page_count": len(getattr(reader, "pages", []))},
            )
            if ok:
                inserted_total += 1
        except Exception as ex:
            logger.warning("Official PDF upsert failed source=%s url=%s error=%s", cfg["source"], url, ex)
        if inserted_total > 0:
            break
    return inserted_total


def collect_company_official_sources() -> None:
    """Collect high-trust official company documents."""
    logger.info("=== company official collector START ===")
    total = 0
    for cfg in OFFICIAL_RSS_SOURCES:
        try:
            total += _collect_rss_source(cfg)
        except Exception as ex:
            logger.warning("Official RSS source failed source=%s error=%s", cfg["source"], ex)
    for cfg in HTML_SOURCES:
        total += _collect_html_source(cfg)
    for cfg in PDF_SOURCES:
        total += _collect_pdf_source(cfg)
    logger.info("=== company official collector DONE | inserted=%d ===", total)
