import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator
import xml.etree.ElementTree as ET

import httpx
import yaml

from app.db.postgres import get_pg_conn
from app.taxonomy import taxonomy_collection

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    from PyPDF2 import PdfReader

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"

SEMI_ROOT_KEYWORDS = [
    "semiconductor",
    "cmos",
    "foundry",
    "fab",
    "wafer",
    "integrated circuit",
    "vlsi",
    "memory",
    "packaging",
    "interconnect",
    "device",
    "materials",
    "reliability",
    "metrology",
    "inspection",
    "power",
    "sensor",
    "standard",
]

SEMI_DOMAIN_KEYWORDS = [
    "hbm",
    "dram",
    "nand",
    "tsv",
    "hybrid bonding",
    "thermo-compression",
    "cowos",
    "chiplet",
    "2.5d",
    "3d integration",
    "euv",
    "lithography",
    "yield",
    "process variation",
    "defect density",
    "finfet",
    "gaa",
    "nanosheet",
    "advanced packaging",
    "cxl",
    "pim",
    "ucie",
    "interposer",
    "glass substrate",
    "backside power",
    "bpdn",
    "sic",
    "gan",
    "rfic",
    "pmic",
    "cis",
    "eda",
    "mpw",
    "reliability",
    "electromigration",
]

SEMI_EXCLUDE_KEYWORDS = [
    "openfoam",
    "cfd",
    "aerodynamics",
    "meteorology",
    "protein",
    "biomedical",
    "air-quality sensors",
    "long-context language models",
    "sustainable large-scale computing",
    "large language model",
    "llm inference",
    "kv cache",
    "flashattention",
    "quantum teleportation",
    "forever chemicals",
    "medical image",
    "brain tumor",
    "protein folding",
    "battery",
]

SEMI_MEMORY_KEYWORDS = [
    "dram",
    "nand",
    "hbm",
    "flash memory",
    "3d nand",
    "lpddr",
    "ddr5",
    "gddr",
    "tsv",
    "cxl",
    "pim",
    "mram",
    "reram",
]

SEMI_PROCESS_KEYWORDS = [
    "semiconductor",
    "foundry",
    "fab",
    "wafer",
    "euv",
    "lithography",
    "etch",
    "deposition",
    "cmp",
    "metrology",
    "yield",
    "defect density",
    "process variation",
    "advanced packaging",
    "hybrid bonding",
    "thermo-compression",
    "cowos",
    "chiplet",
    "beol",
    "feol",
    "ald",
    "ale",
    "pellicle",
    "opc",
    "patterning",
    "mask",
    "reticle",
    "bpdn",
    "backside power",
    "metrology",
    "inspection",
]

SEMI_DEVICE_KEYWORDS = [
    "finfet",
    "gaa",
    "nanosheet",
    "fefet",
    "mosfet",
    "transistor",
    "ferroelectric",
    "igzo",
    "oxide semiconductor",
    "power device",
    "sic",
    "gan",
    "sensor",
    "rfic",
    "pmic",
    "cis",
]

SEMI_ALLOWED_CATEGORY_PREFIX = [
    "cond-mat",
    "physics.app-ph",
    "eess",
    "cs.ar",
    "cs.et",
    "physics.comp-ph",
    "physics.ins-det",
    "physics.ins-det",
]

SEMI_COMPANY_KEYWORDS = [
    "samsung",
    "sk hynix",
    "hynix",
    "tsmc",
    "micron",
    "intel",
    "amd",
    "nvidia",
    "lam research",
    "applied materials",
    "kla",
    "asml",
    "jedec",
    "irds",
    "semi",
]

SEMI_VENUE_WEIGHTS = {
    "iedm": 1.0,
    "isscc": 1.0,
    "vlsi symposium": 0.95,
    "irps": 0.9,
    "iitc": 0.85,
    "ectc": 0.85,
    "dac": 0.8,
    "iccad": 0.8,
    "hot chips": 0.8,
    "spie advanced lithography": 0.85,
}

_TAXONOMY_COLLECTION = taxonomy_collection()
if _TAXONOMY_COLLECTION:
    SEMI_ROOT_KEYWORDS = _TAXONOMY_COLLECTION.get("root_keywords") or SEMI_ROOT_KEYWORDS
    SEMI_DOMAIN_KEYWORDS = _TAXONOMY_COLLECTION.get("domain_keywords", {}).get("memory") or SEMI_DOMAIN_KEYWORDS
    # Preserve broader coverage across multiple axes.
    for key in ("packaging", "litho", "memory", "logic", "process", "materials", "power", "design", "reliability", "standards", "business"):
        SEMI_DOMAIN_KEYWORDS.extend([k for k in _TAXONOMY_COLLECTION.get("domain_keywords", {}).get(key, []) if k not in SEMI_DOMAIN_KEYWORDS])
    SEMI_EXCLUDE_KEYWORDS = _TAXONOMY_COLLECTION.get("exclude_keywords") or SEMI_EXCLUDE_KEYWORDS
    SEMI_MEMORY_KEYWORDS = _TAXONOMY_COLLECTION.get("domain_keywords", {}).get("memory") or SEMI_MEMORY_KEYWORDS
    SEMI_PROCESS_KEYWORDS = list(dict.fromkeys((SEMI_PROCESS_KEYWORDS or []) + (_TAXONOMY_COLLECTION.get("domain_keywords", {}).get("process") or [])))
    SEMI_DEVICE_KEYWORDS = list(dict.fromkeys((SEMI_DEVICE_KEYWORDS or []) + (_TAXONOMY_COLLECTION.get("domain_keywords", {}).get("logic") or []) + (_TAXONOMY_COLLECTION.get("domain_keywords", {}).get("power") or [])))
    SEMI_ALLOWED_CATEGORY_PREFIX = list(dict.fromkeys(SEMI_ALLOWED_CATEGORY_PREFIX + ["physics.comp-ph", "physics.ins-det"]))
    SEMI_COMPANY_KEYWORDS = list(dict.fromkeys((SEMI_COMPANY_KEYWORDS or []) + (_TAXONOMY_COLLECTION.get("company_keywords") or [])))
    SEMI_VENUE_WEIGHTS = dict(_TAXONOMY_COLLECTION.get("venue_weights") or SEMI_VENUE_WEIGHTS)

def _taxonomy_collection() -> dict[str, Any]:
    return _TAXONOMY_COLLECTION if isinstance(_TAXONOMY_COLLECTION, dict) else {}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int_optional(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _in_year_range(dt: datetime | None, year_min: int | None, year_max: int | None) -> bool:
    if year_min is None and year_max is None:
        return True
    if dt is None:
        return False
    y = dt.year
    if year_min is not None and y < year_min:
        return False
    if year_max is not None and y > year_max:
        return False
    return True


def _safe_int(value: object, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _load_arxiv_query_config() -> dict:
    root = Path(__file__).resolve().parents[2]
    query_path = root / "arxiv_query_set.yaml"
    if not query_path.exists():
        logger.warning("arxiv_query_set.yaml not found: %s", query_path)
        return {"queries": [], "post_filters": {}}

    data = yaml.safe_load(query_path.read_text(encoding="utf-8")) or {}
    queries = data.get("queries", []) or []
    return {
        "queries": [q for q in queries if q.get("enabled", True)],
        "post_filters": data.get("post_filters", {}) or {},
    }


def _load_arxiv_queries() -> list[dict]:
    cfg = _load_arxiv_query_config()
    return cfg.get("queries", [])


_COMPANY_PAPER_WATCHLIST: dict[str, Any] | None = None


def _load_company_paper_watchlist() -> dict[str, Any]:
    global _COMPANY_PAPER_WATCHLIST
    if _COMPANY_PAPER_WATCHLIST is not None:
        return _COMPANY_PAPER_WATCHLIST
    root = Path(__file__).resolve().parents[2]
    watchlist_path = root / "data" / "company_paper_watchlist.yaml"
    if not watchlist_path.exists():
        logger.warning("company_paper_watchlist.yaml not found: %s", watchlist_path)
        _COMPANY_PAPER_WATCHLIST = {"companies": []}
        return _COMPANY_PAPER_WATCHLIST
    try:
        data = yaml.safe_load(watchlist_path.read_text(encoding="utf-8")) or {}
        _COMPANY_PAPER_WATCHLIST = data if isinstance(data, dict) else {"companies": []}
    except Exception as e:
        logger.warning("company paper watchlist load failed: %s", e)
        _COMPANY_PAPER_WATCHLIST = {"companies": []}
    return _COMPANY_PAPER_WATCHLIST


def _company_watchlist_entries() -> list[dict[str, Any]]:
    data = _load_company_paper_watchlist()
    entries = data.get("companies", []) or []
    return [x for x in entries if isinstance(x, dict)]


def _company_openalex_queries() -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for entry in _company_watchlist_entries():
        code = str(entry.get("code", "")).strip().upper()
        name = str(entry.get("name", "")).strip()
        priority = int(entry.get("priority", 2) or 2)
        for q in entry.get("openalex_queries", []) or []:
            if not isinstance(q, dict):
                continue
            query_id = str(q.get("id", "")).strip()
            search = str(q.get("search", "")).strip()
            if not query_id or not search:
                continue
            queries.append(
                {
                    "id": f"company_{query_id}",
                    "search": search,
                    "tags": list(dict.fromkeys((q.get("tags", []) or []) + ["company_priority", code, name.lower()])),
                    "min_year": int(q.get("min_year", 2022) or 2022),
                    "company_code": code,
                    "company_name": name,
                    "priority": priority,
                }
            )
    queries.sort(key=lambda x: (x.get("priority", 99), x.get("company_code", ""), x.get("id", "")))
    return queries


def _company_arxiv_queries() -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for entry in _company_watchlist_entries():
        code = str(entry.get("code", "")).strip().upper()
        name = str(entry.get("name", "")).strip()
        priority = int(entry.get("priority", 2) or 2)
        for q in entry.get("arxiv_queries", []) or []:
            if not isinstance(q, dict):
                continue
            query_id = str(q.get("id", "")).strip()
            query = str(q.get("query", "")).strip()
            if not query_id or not query:
                continue
            queries.append(
                {
                    "id": f"company_{query_id}",
                    "query": query,
                    "max_results": int(q.get("max_results", 40) or 40),
                    "company_code": code,
                    "company_name": name,
                    "priority": priority,
                }
            )
    queries.sort(key=lambda x: (x.get("priority", 99), x.get("company_code", ""), x.get("id", "")))
    return queries


def _parse_iso_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _match_keywords(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    return [kw for kw in keywords if kw in lowered]


def _recency_score(published_at: datetime | None) -> float:
    if not published_at:
        return 0.3
    now = _now_utc()
    days = max(0, (now - published_at).days)
    if days <= 180:
        return 1.0
    if days <= 365:
        return 0.85
    if days <= 730:
        return 0.65
    return 0.4


def _conference_score(text: str) -> tuple[float, str | None]:
    lowered = text.lower()
    best = 0.0
    best_key = None
    venue_weights = _taxonomy_collection().get("venue_weights") or SEMI_VENUE_WEIGHTS
    for k, w in venue_weights.items():
        if k in lowered and w > best:
            best = w
            best_key = k
    return best, best_key


def _evaluate_semiconductor_relevance(paper: dict) -> tuple[bool, float, dict]:
    threshold = float(os.getenv("ARXIV_SEMI_MIN_SCORE", "0.55"))
    title = str(paper.get("title", "") or "")
    summary = str(paper.get("summary", "") or "")
    journal_ref = str(paper.get("journal_ref", "") or "")
    comment = str(paper.get("comment", "") or "")
    categories = " ".join(paper.get("categories", []) or [])
    affiliations = " ".join(paper.get("affiliations", []) or [])
    authors = " ".join(paper.get("authors", []) or [])

    base_text = " ".join([title, summary, journal_ref, comment, categories, affiliations, authors])
    base_l = base_text.lower()

    tax = _taxonomy_collection()
    root_keywords = tax.get("root_keywords") or SEMI_ROOT_KEYWORDS
    domain_keywords = list(dict.fromkeys(
        (tax.get("domain_keywords", {}).get("memory") or SEMI_DOMAIN_KEYWORDS)
        + (tax.get("domain_keywords", {}).get("packaging") or [])
        + (tax.get("domain_keywords", {}).get("litho") or [])
        + (tax.get("domain_keywords", {}).get("logic") or [])
        + (tax.get("domain_keywords", {}).get("process") or [])
        + (tax.get("domain_keywords", {}).get("materials") or [])
        + (tax.get("domain_keywords", {}).get("power") or [])
        + (tax.get("domain_keywords", {}).get("design") or [])
        + (tax.get("domain_keywords", {}).get("reliability") or [])
        + (tax.get("domain_keywords", {}).get("standards") or [])
        + (tax.get("domain_keywords", {}).get("business") or [])
    ))
    exclude_keywords = tax.get("exclude_keywords") or SEMI_EXCLUDE_KEYWORDS
    memory_keywords = tax.get("domain_keywords", {}).get("memory") or SEMI_MEMORY_KEYWORDS
    process_keywords = list(dict.fromkeys((SEMI_PROCESS_KEYWORDS or []) + (tax.get("domain_keywords", {}).get("process") or [])))
    device_keywords = list(dict.fromkeys((SEMI_DEVICE_KEYWORDS or []) + (tax.get("domain_keywords", {}).get("logic") or []) + (tax.get("domain_keywords", {}).get("power") or [])))
    company_keywords = tax.get("company_keywords") or SEMI_COMPANY_KEYWORDS

    root_hits = _match_keywords(base_l, root_keywords)
    domain_hits = _match_keywords(base_l, domain_keywords)
    exclude_hits = _match_keywords(base_l, exclude_keywords)
    memory_hits = _match_keywords(base_l, memory_keywords)
    process_hits = _match_keywords(base_l, process_keywords)
    device_hits = _match_keywords(base_l, device_keywords)
    company_hits = _match_keywords(base_l, company_keywords)
    venue_s, venue_name = _conference_score(base_l)
    recency = _recency_score(paper.get("published"))
    categories_list = [str(c).lower() for c in (paper.get("categories", []) or [])]
    allowed_prefixes = tax.get("allowed_category_prefix") or SEMI_ALLOWED_CATEGORY_PREFIX
    category_allowed = any(
        any(c.startswith(prefix) for prefix in allowed_prefixes)
        for c in categories_list
    )

    # hard gates
    if not root_hits:
        return False, 0.0, {"reason": "missing_root_keyword"}
    if not domain_hits:
        return False, 0.0, {"reason": "missing_domain_keyword"}
    if exclude_hits:
        return False, 0.0, {"reason": "exclude_keyword", "exclude_hits": exclude_hits}
    # Focus gate: keep truly semiconductor-tech papers and reduce HPC/LLM memory papers.
    if not (len(memory_hits) >= 1 or len(process_hits) >= 2 or len(device_hits) >= 2):
        return False, 0.0, {"reason": "insufficient_semiconductor_focus"}
    if (not category_allowed) and len(memory_hits) == 0 and len(process_hits) < 3:
        return False, 0.0, {"reason": "category_out_of_focus", "categories": categories_list}

    kw_score = min(1.0, (len(set(root_hits)) * 0.25 + len(set(domain_hits)) * 0.15))
    focus_score = min(
        1.0,
        (len(set(memory_hits)) * 0.4 + len(set(process_hits)) * 0.12 + len(set(device_hits)) * 0.2),
    )
    company_score = min(1.0, len(set(company_hits)) * 0.35)
    score = (
        (0.30 * kw_score)
        + (0.25 * focus_score)
        + (0.15 * recency)
        + (0.15 * venue_s)
        + (0.15 * company_score)
    )
    score = max(0.0, min(1.0, score))
    accepted = score >= threshold

    detail = {
        "threshold": threshold,
        "score": round(score, 4),
        "root_hits": root_hits,
        "domain_hits": domain_hits,
        "memory_hits": memory_hits,
        "process_hits": process_hits,
        "device_hits": device_hits,
        "focus_score": round(focus_score, 4),
        "company_hits": company_hits,
        "venue_name": venue_name,
        "categories": categories_list,
        "category_allowed": category_allowed,
        "recency_score": recency,
        "keyword_score": round(kw_score, 4),
        "venue_score": round(venue_s, 4),
        "company_score": round(company_score, 4),
    }
    return accepted, score, detail


def _doc_uid(source: str, url: str, published_at: datetime | None) -> str:
    base = f"{source}|{url}|{published_at.isoformat() if published_at else ''}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _sanitize_text(value: str | None) -> str | None:
    if value is None:
        return None
    # PostgreSQL text cannot contain NUL bytes.
    return value.replace("\x00", "")


def _upsert_paper(
    source: str,
    title: str,
    url: str,
    published_at: datetime | None,
    summary: str,
    tags: list[str],
    confidence: float,
    extra: dict,
    content: str | None = None,
) -> str:
    uid = _doc_uid(source, url, published_at)
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
                  content = COALESCE(EXCLUDED.content, tech_documents.content),
                  tags = EXCLUDED.tags,
                  confidence = EXCLUDED.confidence,
                  extra = EXCLUDED.extra,
                  collected_at = EXCLUDED.collected_at
                """,
                (
                    uid,
                    source,
                    "paper",
                    _sanitize_text(title) or "",
                    url,
                    published_at,
                    _now_utc(),
                    _sanitize_text(summary) or "",
                    _sanitize_text(content),
                    tags,
                    confidence,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()
    return uid


def _upsert_doc_content(doc_uid: str, full_text: str, extra: dict) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tech_documents
                SET content = %s,
                    extra = extra || %s::jsonb,
                    collected_at = %s
                WHERE doc_uid = %s
                """,
                (_sanitize_text(full_text) or "", json.dumps(extra, ensure_ascii=False), _now_utc(), doc_uid),
            )
        conn.commit()


def _doc_has_chunks(doc_uid: str) -> bool:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM tech_document_chunks WHERE doc_uid=%s LIMIT 1", (doc_uid,))
            return cur.fetchone() is not None


def _replace_chunks(doc_uid: str, chunks: list[str]) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tech_document_chunks WHERE doc_uid=%s", (doc_uid,))
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO tech_document_chunks(
                      doc_uid, chunk_index, chunk_text, char_len, token_estimate, created_at, extra
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (doc_uid, chunk_index)
                    DO UPDATE SET
                      chunk_text = EXCLUDED.chunk_text,
                      char_len = EXCLUDED.char_len,
                      token_estimate = EXCLUDED.token_estimate,
                      created_at = EXCLUDED.created_at,
                      extra = EXCLUDED.extra
                    """,
                    (
                        doc_uid,
                        idx,
                        _sanitize_text(chunk) or "",
                        len(chunk),
                        max(1, len(chunk) // 4),
                        _now_utc(),
                        json.dumps({}, ensure_ascii=False),
                    ),
                )
        conn.commit()


def _split_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    sections: list[dict] = []
    current_title = "Document"
    current_lines: list[str] = []
    section_index = 0
    char_cursor = 0
    start_char = 0

    heading_num = re.compile(r"^\s*(\d+(\.\d+)*)\s+([A-Z][A-Za-z0-9\-\s]{2,120})\s*$")
    heading_named = re.compile(
        r"^\s*(abstract|introduction|background|related work|method|methods|experiment|experiments|results|discussion|conclusion|references)\s*$",
        re.IGNORECASE,
    )

    def flush_section(end_char: int) -> None:
        nonlocal section_index, start_char, current_lines, current_title
        body = "\n".join(current_lines).strip()
        if not body:
            return
        sections.append(
            {
                "section_index": section_index,
                "section_title": current_title,
                "section_level": 1,
                "start_char": start_char,
                "end_char": end_char,
                "section_text": body,
            }
        )
        section_index += 1
        current_lines = []
        start_char = end_char + 1

    for line in lines:
        stripped = line.strip()
        is_heading = bool(heading_num.match(stripped) or heading_named.match(stripped))
        if is_heading and current_lines:
            flush_section(char_cursor)
            current_title = stripped[:160]
        elif is_heading and not current_lines:
            current_title = stripped[:160]
        else:
            current_lines.append(line)
        char_cursor += len(line) + 1

    flush_section(char_cursor)
    return sections


def _extract_caption_items(text: str, kind: str) -> list[dict]:
    lines = text.splitlines()
    out: list[dict] = []
    idx = 0
    if kind == "table":
        pat = re.compile(r"^\s*(table)\s*([0-9ivx\.\-]+)[:\.\s]+(.+)$", re.IGNORECASE)
    else:
        pat = re.compile(r"^\s*(figure|fig\.?)\s*([0-9ivx\.\-]+)[:\.\s]+(.+)$", re.IGNORECASE)

    for line in lines:
        m = pat.match(line.strip())
        if not m:
            continue
        idx += 1
        out.append(
            {
                "index": idx,
                "label_no": m.group(2),
                "caption": m.group(3).strip()[:1000],
                "raw_text": line.strip()[:2000],
                "page_hint": None,
                "image_path": None,
            }
        )
    return out


def _replace_sections(doc_uid: str, sections: list[dict]) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_sections WHERE doc_uid=%s", (doc_uid,))
            for s in sections:
                cur.execute(
                    """
                    INSERT INTO paper_sections(
                      doc_uid, section_index, section_title, section_level,
                      start_char, end_char, section_text, created_at, extra
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (doc_uid, section_index)
                    DO UPDATE SET
                      section_title = EXCLUDED.section_title,
                      section_level = EXCLUDED.section_level,
                      start_char = EXCLUDED.start_char,
                      end_char = EXCLUDED.end_char,
                      section_text = EXCLUDED.section_text,
                      created_at = EXCLUDED.created_at,
                      extra = EXCLUDED.extra
                    """,
                    (
                        doc_uid,
                        s["section_index"],
                        s["section_title"],
                        s["section_level"],
                        s["start_char"],
                        s["end_char"],
                        _sanitize_text(s["section_text"]) or "",
                        _now_utc(),
                        json.dumps({}, ensure_ascii=False),
                    ),
                )
        conn.commit()


def _replace_tables(doc_uid: str, tables: list[dict]) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_tables WHERE doc_uid=%s", (doc_uid,))
            for t in tables:
                cur.execute(
                    """
                    INSERT INTO paper_tables(
                      doc_uid, table_index, caption, page_hint, raw_text, image_path, created_at, extra
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (doc_uid, table_index)
                    DO UPDATE SET
                      caption = EXCLUDED.caption,
                      page_hint = EXCLUDED.page_hint,
                      raw_text = EXCLUDED.raw_text,
                      image_path = EXCLUDED.image_path,
                      created_at = EXCLUDED.created_at,
                      extra = EXCLUDED.extra
                    """,
                    (
                        doc_uid,
                        t["index"],
                        _sanitize_text(t.get("caption")),
                        t.get("page_hint"),
                        _sanitize_text(t.get("raw_text")),
                        t.get("image_path"),
                        _now_utc(),
                        json.dumps({"label_no": t.get("label_no")}, ensure_ascii=False),
                    ),
                )
        conn.commit()


def _replace_figures(doc_uid: str, figures: list[dict]) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM paper_figures WHERE doc_uid=%s", (doc_uid,))
            for f in figures:
                cur.execute(
                    """
                    INSERT INTO paper_figures(
                      doc_uid, figure_index, caption, page_hint, raw_text, image_path, created_at, extra
                    ) VALUES (
                      %s, %s, %s, %s, %s, %s, %s, %s::jsonb
                    )
                    ON CONFLICT (doc_uid, figure_index)
                    DO UPDATE SET
                      caption = EXCLUDED.caption,
                      page_hint = EXCLUDED.page_hint,
                      raw_text = EXCLUDED.raw_text,
                      image_path = EXCLUDED.image_path,
                      created_at = EXCLUDED.created_at,
                      extra = EXCLUDED.extra
                    """,
                    (
                        doc_uid,
                        f["index"],
                        _sanitize_text(f.get("caption")),
                        f.get("page_hint"),
                        _sanitize_text(f.get("raw_text")),
                        f.get("image_path"),
                        _now_utc(),
                        json.dumps({"label_no": f.get("label_no")}, ensure_ascii=False),
                    ),
                )
        conn.commit()


def _assets_dir(doc_uid: str) -> Path:
    root = Path(__file__).resolve().parents[2]
    rel = os.getenv("ARXIV_ASSET_DIR", "data/raw/arxiv_assets").strip()
    p = root / rel / doc_uid
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ocr_image(image_path: Path) -> str:
    if not _env_bool("ARXIV_ENABLE_OCR", True):
        return ""
    if pytesseract is None or Image is None:
        return ""
    tess_cmd = os.getenv("TESSERACT_CMD", "").strip()
    if tess_cmd and hasattr(pytesseract, "pytesseract"):
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
    try:
        img = Image.open(str(image_path))
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception:
        return ""


def _extract_visual_assets(pdf_path: Path, doc_uid: str) -> tuple[list[dict], list[dict]]:
    if fitz is None:
        logger.warning("PyMuPDF is unavailable. Skipping figure/table image extraction.")
        return [], []

    table_pat = re.compile(r"^\s*(table)\s*([0-9ivx\.\-]+)[:\.\s]+(.+)$", re.IGNORECASE)
    fig_pat = re.compile(r"^\s*(figure|fig\.?)\s*([0-9ivx\.\-]+)[:\.\s]+(.+)$", re.IGNORECASE)
    asset_dir = _assets_dir(doc_uid)

    tables: list[dict] = []
    figures: list[dict] = []
    table_idx = 0
    fig_idx = 0

    doc = fitz.open(str(pdf_path))
    try:
        for page_no, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks") or []
            caption_blocks = []
            for b in blocks:
                x0, y0, x1, y1, text = b[0], b[1], b[2], b[3], (b[4] or "")
                for line in [ln.strip() for ln in text.splitlines() if ln.strip()]:
                    m_table = table_pat.match(line)
                    m_fig = fig_pat.match(line)
                    if m_table:
                        caption_blocks.append(("table", m_table.group(2), m_table.group(3), x0, y0, x1, y1, line))
                        break
                    if m_fig:
                        caption_blocks.append(("figure", m_fig.group(2), m_fig.group(3), x0, y0, x1, y1, line))
                        break

            caption_blocks.sort(key=lambda x: x[4])  # y0
            for i, c in enumerate(caption_blocks):
                kind, label_no, cap, x0, y0, x1, y1, raw_line = c
                next_y0 = page.rect.height
                if i + 1 < len(caption_blocks):
                    next_y0 = caption_blocks[i + 1][4]
                clip = fitz.Rect(0, y1, page.rect.width, max(y1 + 20, next_y0))
                if clip.height < 20:
                    continue

                pix = page.get_pixmap(clip=clip, dpi=200)
                if kind == "table":
                    table_idx += 1
                    img_name = f"table_{table_idx:04d}_p{page_no}.png"
                    img_path = asset_dir / img_name
                    pix.save(str(img_path))
                    ocr_text = _ocr_image(img_path)
                    tables.append(
                        {
                            "index": table_idx,
                            "label_no": label_no,
                            "caption": cap[:1000],
                            "raw_text": (ocr_text[:20000] if ocr_text else raw_line[:2000]),
                            "page_hint": page_no,
                            "image_path": str(img_path),
                        }
                    )
                else:
                    fig_idx += 1
                    img_name = f"figure_{fig_idx:04d}_p{page_no}.png"
                    img_path = asset_dir / img_name
                    pix.save(str(img_path))
                    ocr_text = _ocr_image(img_path)
                    figures.append(
                        {
                            "index": fig_idx,
                            "label_no": label_no,
                            "caption": cap[:1000],
                            "raw_text": (ocr_text[:20000] if ocr_text else raw_line[:2000]),
                            "page_hint": page_no,
                            "image_path": str(img_path),
                        }
                    )
    finally:
        doc.close()

    return tables, figures


def _fetch_arxiv(query_text: str, max_results: int) -> list[dict]:
    timeout = float(max(10, int(max_results / 2)))
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(
            ARXIV_API,
            params={
                "search_query": query_text,
                "start": 0,
                "max_results": max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        resp.raise_for_status()
        xml_text = resp.text

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    out = []
    for entry in root.findall("atom:entry", ns):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
        entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
        updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
        journal_ref = (entry.findtext("arxiv:journal_ref", default="", namespaces=ns) or "").strip()
        comment = (entry.findtext("arxiv:comment", default="", namespaces=ns) or "").strip()
        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns) if c.attrib.get("term")]
        authors = []
        affiliations = []
        for a in entry.findall("atom:author", ns):
            n = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
            aff = (a.findtext("arxiv:affiliation", default="", namespaces=ns) or "").strip()
            if n:
                authors.append(n)
            if aff:
                affiliations.append(aff)
        out.append(
            {
                "title": title,
                "summary": summary,
                "id": entry_id,
                "published": _parse_iso_dt(published),
                "updated": updated,
                "journal_ref": journal_ref,
                "comment": comment,
                "categories": categories,
                "authors": authors,
                "affiliations": affiliations,
            }
        )
    return out


def _fetch_arxiv_by_id(arxiv_id: str) -> dict | None:
    timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        resp = client.get(
            ARXIV_API,
            params={
                "id_list": arxiv_id,
                "max_results": 1,
            },
        )
        resp.raise_for_status()
        xml_text = resp.text

    ns = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }
    root = ET.fromstring(xml_text)
    entry = root.find("atom:entry", ns)
    if entry is None:
        return None

    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
    summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip().replace("\n", " ")
    entry_id = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
    published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
    updated = (entry.findtext("atom:updated", default="", namespaces=ns) or "").strip()
    journal_ref = (entry.findtext("arxiv:journal_ref", default="", namespaces=ns) or "").strip()
    comment = (entry.findtext("arxiv:comment", default="", namespaces=ns) or "").strip()
    categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns) if c.attrib.get("term")]
    authors = []
    affiliations = []
    for a in entry.findall("atom:author", ns):
        n = (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
        aff = (a.findtext("arxiv:affiliation", default="", namespaces=ns) or "").strip()
        if n:
            authors.append(n)
        if aff:
            affiliations.append(aff)
    return {
        "title": title,
        "summary": summary,
        "id": entry_id,
        "published": _parse_iso_dt(published),
        "updated": updated,
        "journal_ref": journal_ref,
        "comment": comment,
        "categories": categories,
        "authors": authors,
        "affiliations": affiliations,
    }


def _normalize_arxiv_ref(ref: str) -> str:
    value = (ref or "").strip()
    value = value.replace("arXiv:", "").strip()
    if value.startswith("http://") or value.startswith("https://"):
        if "/abs/" in value:
            return value
        if "/pdf/" in value and value.endswith(".pdf"):
            aid = value.split("/pdf/")[-1].replace(".pdf", "")
            return f"https://arxiv.org/abs/{aid}"
        return value
    return f"https://arxiv.org/abs/{value}"


def _extract_arxiv_id(url: str) -> str | None:
    m = re.search(r"arxiv\.org/abs/([^\s?#]+)", url)
    if not m:
        return None
    return m.group(1).strip()


def _pdf_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    rel = os.getenv("ARXIV_PDF_DIR", "data/raw/arxiv").strip()
    p = root / rel
    p.mkdir(parents=True, exist_ok=True)
    return p


def _download_pdf(arxiv_id: str) -> Path | None:
    pdf_path = _pdf_dir() / f"{arxiv_id.replace('/', '_')}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))
    with httpx.Client(timeout=timeout, follow_redirects=True) as client:
        r = client.get(url)
        r.raise_for_status()
        pdf_path.write_bytes(r.content)
    return pdf_path


def _extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append(txt)
    return "\n".join(pages).strip()


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if not text:
        return []
    if chunk_size <= 0:
        chunk_size = 1200
    if overlap < 0:
        overlap = 0
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def _process_pdf_and_chunks(doc_uid: str, paper_url: str) -> bool:
    arxiv_id = _extract_arxiv_id(paper_url)
    if not arxiv_id:
        logger.warning("Cannot parse arXiv id from url=%s", paper_url)
        return False

    force_rechunk = _env_bool("ARXIV_FORCE_RECHUNK", False)
    if (not force_rechunk) and _doc_has_chunks(doc_uid):
        return False

    pdf_path = _download_pdf(arxiv_id)
    if not pdf_path:
        return False

    text = _extract_pdf_text(pdf_path)
    if not text:
        logger.warning("Empty extracted text | arxiv_id=%s", arxiv_id)
        return False

    chunk_size = _env_int("ARXIV_CHUNK_SIZE", 1500)
    overlap = _env_int("ARXIV_CHUNK_OVERLAP", 200)
    chunks = _chunk_text(text, chunk_size, overlap)
    if not chunks:
        return False

    _replace_chunks(doc_uid, chunks)
    sections = _split_sections(text)
    tables = _extract_caption_items(text, kind="table")
    figures = _extract_caption_items(text, kind="figure")

    extracted_tables, extracted_figures = _extract_visual_assets(pdf_path, doc_uid)
    if extracted_tables:
        tables = extracted_tables
    if extracted_figures:
        figures = extracted_figures

    _replace_sections(doc_uid, sections)
    _replace_tables(doc_uid, tables)
    _replace_figures(doc_uid, figures)

    _upsert_doc_content(
        doc_uid,
        full_text=text,
        extra={
            "pdf_path": str(pdf_path),
            "chunk_count": len(chunks),
            "section_count": len(sections),
            "table_caption_count": len(tables),
            "figure_caption_count": len(figures),
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "text_char_len": len(text),
        },
    )
    return True


def _get_doc_structure_counts(doc_uid: str) -> dict:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM tech_document_chunks WHERE doc_uid=%s", (doc_uid,))
            chunks = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM paper_sections WHERE doc_uid=%s", (doc_uid,))
            sections = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM paper_tables WHERE doc_uid=%s", (doc_uid,))
            tables = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(*) FROM paper_figures WHERE doc_uid=%s", (doc_uid,))
            figures = int(cur.fetchone()[0])
    return {
        "chunk_count": chunks,
        "section_count": sections,
        "table_count": tables,
        "figure_count": figures,
    }


def ingest_single_arxiv_paper(arxiv_ref: str, query_id: str = "manual_demo") -> dict:
    abs_url = _normalize_arxiv_ref(arxiv_ref)
    arxiv_id = _extract_arxiv_id(abs_url)
    if not arxiv_id:
        raise ValueError(f"Invalid arXiv reference: {arxiv_ref}")

    logger.info("Single ingest start | arxiv_id=%s", arxiv_id)
    meta = _fetch_arxiv_by_id(arxiv_id) or {
        "title": abs_url,
        "summary": "",
        "id": abs_url,
        "published": _now_utc(),
        "updated": "",
        "journal_ref": "",
        "comment": "",
        "categories": [],
        "authors": [],
        "affiliations": [],
    }

    paper_url = meta["id"] or abs_url
    uid = _upsert_paper(
        source="arxiv",
        title=meta["title"] or paper_url,
        url=paper_url,
        published_at=meta["published"],
        summary=meta["summary"],
        tags=["arxiv", query_id],
        confidence=0.85,
        extra={
            "query_id": query_id,
            "updated": meta.get("updated", ""),
            "journal_ref": meta.get("journal_ref", ""),
            "comment": meta.get("comment", ""),
            "categories": meta.get("categories", []),
            "authors": meta.get("authors", []),
            "affiliations": meta.get("affiliations", []),
        },
        content=None,
    )

    processed = _process_pdf_and_chunks(uid, paper_url)
    counts = _get_doc_structure_counts(uid)
    result = {
        "doc_uid": uid,
        "arxiv_id": arxiv_id,
        "url": paper_url,
        "processed": processed,
        **counts,
    }
    logger.info("Single ingest done | %s", result)
    return result


def ingest_top_arxiv_by_query(query_text: str, query_id: str = "manual_query", pick_index: int = 0, max_results: int = 10) -> dict:
    query = (query_text or "").strip()
    if not query:
        raise ValueError("query_text is empty")

    papers = _fetch_arxiv(query, max_results=max_results)
    if not papers:
        raise RuntimeError(f"No papers found for query: {query}")

    ranked = []
    for p in papers:
        accepted, score, detail = _evaluate_semiconductor_relevance(p)
        if accepted:
            ranked.append((p, score, detail))
        else:
            logger.info("paper rejected | title=%s reason=%s", p.get("title", ""), detail.get("reason", "score"))

    if not ranked:
        raise RuntimeError("No semiconductor-focused paper passed filtering rules.")

    ranked.sort(key=lambda x: x[1], reverse=True)
    idx = max(0, min(pick_index, len(ranked) - 1))
    picked, picked_score, picked_detail = ranked[idx]
    result = ingest_single_arxiv_paper(picked["id"], query_id=query_id)
    result["picked_title"] = picked.get("title")
    result["picked_query"] = query
    result["picked_index"] = idx
    result["picked_score"] = round(picked_score, 4)
    result["picked_filter_detail"] = picked_detail
    return result


def ingest_batch_arxiv_by_query(
    query_text: str,
    query_id: str = "batch_query",
    max_results: int = 50,
    ingest_limit: int = 12,
    year_min: int | None = None,
    year_max: int | None = None,
) -> dict:
    query = (query_text or "").strip()
    if not query:
        raise ValueError("query_text is empty")

    hard_cap = _env_int("ARXIV_BATCH_INGEST_CAP", 20)
    ingest_limit = max(1, min(int(ingest_limit), hard_cap))

    env_year_min = _env_int_optional("ARXIV_YEAR_MIN")
    env_year_max = _env_int_optional("ARXIV_YEAR_MAX")
    if year_min is None:
        year_min = env_year_min
    if year_max is None:
        year_max = env_year_max

    papers_all = _fetch_arxiv(query, max_results=max_results)
    papers = [p for p in papers_all if _in_year_range(p.get("published"), year_min, year_max)]
    year_filtered_out = len(papers_all) - len(papers)

    if not papers:
        return {
            "query": query,
            "fetched": len(papers_all),
            "accepted": 0,
            "ingested": 0,
            "failed": 0,
            "rejected": 0,
            "year_filtered_out": year_filtered_out,
            "year_min": year_min,
            "year_max": year_max,
            "limit": ingest_limit,
            "items": [],
        }

    ranked = []
    rejected = 0
    for p in papers:
        accepted, score, detail = _evaluate_semiconductor_relevance(p)
        if not accepted:
            rejected += 1
            logger.info(
                "batch reject | title=%s reason=%s score=%s",
                p.get("title", ""),
                detail.get("reason", "score_below_threshold"),
                detail.get("score"),
            )
            continue
        ranked.append((p, score, detail))

    ranked.sort(key=lambda x: x[1], reverse=True)
    selected = ranked[:ingest_limit]

    items = []
    failed = 0
    for idx, (p, score, detail) in enumerate(selected, start=1):
        try:
            logger.info(
                "batch ingest start | %s/%s | score=%.4f | title=%s",
                idx,
                len(selected),
                score,
                p.get("title", ""),
            )
            result = ingest_single_arxiv_paper(p.get("id", ""), query_id=query_id)
            result["score"] = round(score, 4)
            result["filter_detail"] = detail
            items.append(result)
        except Exception as e:
            failed += 1
            logger.warning("batch ingest failed | title=%s error=%s", p.get("title", ""), e)

    summary = {
        "query": query,
        "fetched": len(papers_all),
        "accepted": len(ranked),
        "ingested": len(items),
        "failed": failed,
        "rejected": rejected,
        "year_filtered_out": year_filtered_out,
        "year_min": year_min,
        "year_max": year_max,
        "limit": ingest_limit,
        "items": items,
    }
    logger.info(
        "batch ingest done | fetched=%s year_filtered_out=%s year_min=%s year_max=%s accepted=%s ingested=%s failed=%s rejected=%s limit=%s",
        summary["fetched"],
        summary["year_filtered_out"],
        summary["year_min"],
        summary["year_max"],
        summary["accepted"],
        summary["ingested"],
        summary["failed"],
        summary["rejected"],
        summary["limit"],
    )
    return summary


def monitor_arxiv_new_papers() -> None:
    cfg = _load_arxiv_query_config()
    queries = cfg.get("queries", [])
    if not queries:
        logger.warning("No arXiv queries found. Skip monitor_arxiv_new_papers.")
        return

    post_filters = cfg.get("post_filters", {}) or {}
    year_min = _env_int_optional("ARXIV_YEAR_MIN")
    year_max = _env_int_optional("ARXIV_YEAR_MAX")
    if year_min is None:
        year_min = _safe_int(post_filters.get("year_min"), None)
    if year_max is None:
        year_max = _safe_int(post_filters.get("year_max"), None)

    download_pdf = _env_bool("ARXIV_DOWNLOAD_PDF", True)
    max_pdf_per_run = _env_int("ARXIV_MAX_PDF_PER_RUN", 25)
    pdf_processed = 0

    total = 0
    for q in queries:
        qid = str(q.get("id", "unknown"))
        qtext = str(q.get("query", "")).strip()
        max_results = int(q.get("max_results", 30))
        if not qtext:
            continue

        logger.info("arXiv query start | id=%s max_results=%s", qid, max_results)
        try:
            papers = _fetch_arxiv(qtext, max_results=max_results)
        except Exception as e:
            logger.warning("arXiv query failed | id=%s error=%s", qid, e)
            continue

        inserted_local = 0
        rejected_local = 0
        year_rejected_local = 0
        for p in papers:
            if not p["id"]:
                continue
            if not _in_year_range(p.get("published"), year_min, year_max):
                year_rejected_local += 1
                rejected_local += 1
                logger.info(
                    "paper rejected | query_id=%s title=%s reason=year_out_of_range year=%s min=%s max=%s",
                    qid,
                    p.get("title", ""),
                    p.get("published").year if p.get("published") else None,
                    year_min,
                    year_max,
                )
                continue
            accepted, score, detail = _evaluate_semiconductor_relevance(p)
            if not accepted:
                rejected_local += 1
                logger.info(
                    "paper rejected | query_id=%s title=%s reason=%s score=%s",
                    qid,
                    p.get("title", ""),
                    detail.get("reason", "score_below_threshold"),
                    detail.get("score"),
                )
                continue
            uid = _upsert_paper(
                source="arxiv",
                title=p["title"] or p["id"],
                url=p["id"],
                published_at=p["published"],
                summary=p["summary"],
                tags=["arxiv", qid],
                confidence=max(0.6, min(0.95, score)),
                extra={
                    "query_id": qid,
                    "updated": p["updated"],
                    "journal_ref": p.get("journal_ref", ""),
                    "comment": p.get("comment", ""),
                    "categories": p.get("categories", []),
                    "authors": p.get("authors", []),
                    "affiliations": p.get("affiliations", []),
                    "filter_detail": detail,
                },
                content=None,
            )
            inserted_local += 1
            total += 1

            if download_pdf and pdf_processed < max_pdf_per_run:
                try:
                    did = _process_pdf_and_chunks(uid, p["id"])
                    if did:
                        pdf_processed += 1
                        logger.info("arXiv PDF processed | doc_uid=%s processed=%s", uid, pdf_processed)
                except Exception as e:
                    logger.warning("arXiv PDF process failed | doc_uid=%s error=%s", uid, e)

        logger.info(
            "arXiv query done | id=%s fetched=%s upserted=%s rejected=%s year_rejected=%s year_min=%s year_max=%s",
            qid,
            len(papers),
            inserted_local,
            rejected_local,
            year_rejected_local,
            year_min,
            year_max,
        )

    logger.info(
        "monitor_arxiv_new_papers done | upserted_total=%s pdf_processed=%s",
        total,
        pdf_processed,
    )


def monitor_arxiv_company_papers() -> None:
    """Company-priority arXiv harvest focused on the latest company-linked papers."""
    cfg = _load_arxiv_query_config()
    post_filters = cfg.get("post_filters", {}) or {}
    year_min = _env_int_optional("ARXIV_YEAR_MIN")
    year_max = _env_int_optional("ARXIV_YEAR_MAX")
    if year_min is None:
        year_min = _safe_int(post_filters.get("year_min"), None)
    if year_max is None:
        year_max = _safe_int(post_filters.get("year_max"), None)

    download_pdf = _env_bool("ARXIV_DOWNLOAD_PDF", True)
    max_pdf_per_run = _env_int("ARXIV_MAX_PDF_PER_RUN", 12)
    query_limit = _env_int("ARXIV_COMPANY_QUERY_LIMIT", 12)
    pdf_processed = 0
    total = 0

    queries = _company_arxiv_queries()[: max(1, query_limit)]
    if not queries:
        logger.warning("No company arXiv queries found. Skip monitor_arxiv_company_papers.")
        return

    for q in queries:
        qid = str(q.get("id", "unknown"))
        qtext = str(q.get("query", "")).strip()
        max_results = int(q.get("max_results", 30))
        if not qtext:
            continue

        logger.info("Company arXiv query start | id=%s max_results=%s", qid, max_results)
        try:
            papers = _fetch_arxiv(qtext, max_results=max_results)
        except Exception as e:
            logger.warning("Company arXiv query failed | id=%s error=%s", qid, e)
            continue

        inserted_local = 0
        rejected_local = 0
        year_rejected_local = 0
        for p in papers:
            if not p["id"]:
                continue
            if not _in_year_range(p.get("published"), year_min, year_max):
                year_rejected_local += 1
                rejected_local += 1
                continue
            accepted, score, detail = _evaluate_semiconductor_relevance(p)
            if not accepted:
                rejected_local += 1
                continue
            uid = _upsert_paper(
                source="arxiv",
                title=p["title"] or p["id"],
                url=p["id"],
                published_at=p["published"],
                summary=p["summary"],
                tags=["arxiv", qid, "company_priority"],
                confidence=max(0.6, min(0.95, score)),
                extra={
                    "query_id": qid,
                    "company_priority": True,
                    "updated": p["updated"],
                    "journal_ref": p.get("journal_ref", ""),
                    "comment": p.get("comment", ""),
                    "categories": p.get("categories", []),
                    "authors": p.get("authors", []),
                    "affiliations": p.get("affiliations", []),
                    "filter_detail": detail,
                },
                content=None,
            )
            inserted_local += 1
            total += 1

            if download_pdf and pdf_processed < max_pdf_per_run:
                try:
                    did = _process_pdf_and_chunks(uid, p["id"])
                    if did:
                        pdf_processed += 1
                except Exception as e:
                    logger.warning("Company arXiv PDF process failed | doc_uid=%s error=%s", uid, e)

        logger.info(
            "Company arXiv query done | id=%s fetched=%s upserted=%s rejected=%s year_rejected=%s year_min=%s year_max=%s",
            qid,
            len(papers),
            inserted_local,
            rejected_local,
            year_rejected_local,
            year_min,
            year_max,
        )

    logger.info(
        "monitor_arxiv_company_papers done | upserted_total=%s pdf_processed=%s",
        total,
        pdf_processed,
    )


# ════════════════════════════════════════════════════════════════════
# Semantic Scholar 논문 수집
# ════════════════════════════════════════════════════════════════════

S2_SEARCH_EP = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,abstract,year,venue,authors,externalIds,citationCount,influentialCitationCount,fieldsOfStudy,publicationDate"

# IEDM/ISSCC/VLSI 중심의 핵심 쿼리 세트
S2_QUERIES: list[dict] = [
    {
        "id": "s2_hbm_packaging",
        "query": "HBM high bandwidth memory 3D stacking TSV",
        "min_year": 2022,
        "tags": ["hbm", "packaging", "tsv"],
    },
    {
        "id": "s2_tc_bonding",
        "query": "thermo-compression bonding flip chip advanced packaging semiconductor",
        "min_year": 2022,
        "tags": ["tc_bonding", "packaging", "flip_chip"],
    },
    {
        "id": "s2_hybrid_bonding",
        "query": "hybrid bonding wafer bonding Cu-Cu direct bonding chiplet",
        "min_year": 2022,
        "tags": ["hybrid_bonding", "chiplet", "packaging"],
    },
    {
        "id": "s2_3d_nand",
        "query": "3D NAND flash memory cell stacking vertical channel",
        "min_year": 2022,
        "tags": ["nand", "3d_nand", "flash_memory"],
    },
    {
        "id": "s2_euv_litho",
        "query": "EUV extreme ultraviolet lithography high-NA patterning",
        "min_year": 2022,
        "tags": ["euv", "lithography", "patterning"],
    },
    {
        "id": "s2_gaa_finfet",
        "query": "gate-all-around nanosheet transistor MOSFET CMOS scaling",
        "min_year": 2022,
        "tags": ["gaa", "nanosheet", "finfet", "transistor"],
    },
    {
        "id": "s2_cowos_packaging",
        "query": "CoWoS chip on wafer on substrate silicon interposer 2.5D packaging",
        "min_year": 2022,
        "tags": ["cowos", "interposer", "packaging"],
    },
    {
        "id": "s2_dram_cell",
        "query": "DRAM cell capacitor process technology next generation memory",
        "min_year": 2022,
        "tags": ["dram", "memory", "capacitor"],
    },
]

# 상위 컨퍼런스 가중치 (venue 문자열 부분 매치)
S2_VENUE_BOOST: dict[str, float] = {
    "iedm": 1.0,
    "isscc": 1.0,
    "vlsi symposium": 0.95,
    "irps": 0.90,
    "iitc": 0.85,
    "ectc": 0.85,
    "hot chips": 0.80,
    "spie": 0.80,
    "dac": 0.75,
}


def _s2_headers() -> dict:
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if key:
        return {"x-api-key": key}
    return {}


def _s2_venue_score(venue: str | None) -> float:
    if not venue:
        return 0.0
    v = venue.lower()
    for kw, score in S2_VENUE_BOOST.items():
        if kw in v:
            return score
    return 0.0


def _fetch_s2_papers(query: str, min_year: int, limit: int = 50) -> list[dict]:
    """Semantic Scholar API 검색 → 논문 메타데이터 리스트."""
    params = {
        "query": query,
        "limit": min(limit, 100),
        "fields": S2_FIELDS,
        "year": f"{min_year}-",
    }

    try:
        timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "30"))
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(S2_SEARCH_EP, params=params, headers=_s2_headers())
            if resp.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit — sleeping 60s")
                time.sleep(60)
                resp = client.get(S2_SEARCH_EP, params=params, headers=_s2_headers())
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", []) or []
    except Exception as e:
        logger.warning("S2 fetch failed | query='%s' error=%s", query[:60], e)
        return []


def _s2_paper_url(paper: dict) -> str:
    """Semantic Scholar 논문 URL 생성."""
    ext_ids = paper.get("externalIds") or {}
    doi = ext_ids.get("DOI")
    if doi:
        return f"https://doi.org/{doi}"
    arxiv_id = ext_ids.get("ArXiv")
    if arxiv_id:
        return f"https://arxiv.org/abs/{arxiv_id}"
    paper_id = paper.get("paperId", "")
    return f"https://www.semanticscholar.org/paper/{paper_id}"


def _s2_doc_uid(paper_id: str) -> str:
    return hashlib.sha1(f"s2|{paper_id}".encode()).hexdigest()


def _s2_evaluate_relevance(paper: dict, tags: list[str]) -> float:
    """간단 점수 계산: venue 가중치 + citation 보너스."""
    venue = paper.get("venue") or ""
    venue_s = _s2_venue_score(venue)

    citations = int(paper.get("citationCount") or 0)
    influential = int(paper.get("influentialCitationCount") or 0)

    # citation 보너스 (log-scale)
    cit_bonus = min(0.3, math.log1p(citations) / 20.0)
    inf_bonus  = min(0.2, math.log1p(influential) / 10.0)

    base = 0.5 + venue_s * 0.3 + cit_bonus + inf_bonus
    return min(1.0, base)


def _ingest_s2_paper(paper: dict, query_id: str, tags: list[str]) -> bool:
    """단일 S2 논문 → tech_documents + (arXiv면) PDF 청크."""
    paper_id = paper.get("paperId") or ""
    if not paper_id:
        return False

    title = (paper.get("title") or "").strip()
    abstract = (paper.get("abstract") or "").strip()
    year = paper.get("year")
    venue = paper.get("venue") or ""
    pub_date_str = paper.get("publicationDate") or ""
    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]
    ext_ids = paper.get("externalIds") or {}
    citations = int(paper.get("citationCount") or 0)
    influential = int(paper.get("influentialCitationCount") or 0)
    fields = paper.get("fieldsOfStudy") or []

    # published_at 파싱
    if pub_date_str:
        pub_dt = _parse_iso_dt(pub_date_str)
    elif year:
        pub_dt = datetime(int(year), 1, 1, tzinfo=timezone.utc)
    else:
        pub_dt = None

    url = _s2_paper_url(paper)
    confidence = _s2_evaluate_relevance(paper, tags)

    all_tags = list(set(["semantic_scholar", query_id] + tags))

    extra = {
        "source": "semantic_scholar",
        "paper_id": paper_id,
        "venue": venue,
        "year": year,
        "authors": authors[:10],
        "citation_count": citations,
        "influential_citation_count": influential,
        "external_ids": ext_ids,
        "fields_of_study": fields,
        "query_id": query_id,
    }

    # arXiv ID가 있으면 arXiv 소스로 등록 (PDF 청크 연동)
    arxiv_id = ext_ids.get("ArXiv")
    source_label = "arxiv" if arxiv_id else "semantic_scholar"
    abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else url

    uid = _upsert_paper(
        source=source_label,
        title=title or abs_url,
        url=abs_url,
        published_at=pub_dt,
        summary=abstract,
        tags=all_tags,
        confidence=confidence,
        extra=extra,
        content=None,
    )

    # arXiv ID가 있으면 PDF도 시도 (ARXIV_DOWNLOAD_PDF=1이면)
    if arxiv_id and _env_bool("ARXIV_DOWNLOAD_PDF", True):
        try:
            _process_pdf_and_chunks(uid, abs_url)
        except Exception as e:
            logger.debug("S2 PDF process skip | arxiv_id=%s error=%s", arxiv_id, e)

    return True


def collect_semantic_scholar_weekly() -> None:
    """Semantic Scholar 반도체 논문 주간 수집 (IEDM/ISSCC/VLSI 중심)."""
    logger.info("=== collect_semantic_scholar_weekly START ===")

    total_fetched = 0
    total_ingested = 0
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if not api_key:
        logger.info("SEMANTIC_SCHOLAR_API_KEY 미설정 — 무료 티어로 수집 (속도 제한 주의)")

    for q in S2_QUERIES:
        qid     = q["id"]
        query   = q["query"]
        min_yr  = q.get("min_year", 2022)
        tags    = q.get("tags", [])

        logger.info("S2 query: id=%s min_year=%d", qid, min_yr)

        papers = _fetch_s2_papers(query, min_year=min_yr, limit=50)
        total_fetched += len(papers)
        ingested_local = 0

        for paper in papers:
            try:
                ok = _ingest_s2_paper(paper, query_id=qid, tags=tags)
                if ok:
                    ingested_local += 1
                    total_ingested += 1
            except Exception as e:
                logger.warning("S2 ingest failed | paper_id=%s error=%s", paper.get("paperId"), e)

        logger.info("S2 query done | id=%s fetched=%d ingested=%d", qid, len(papers), ingested_local)

        # API 속도 제한 준수 (무료: ~1 req/sec)
        sleep_time = 2.0 if not api_key else 0.5
        time.sleep(sleep_time)

    logger.info(
        "=== collect_semantic_scholar_weekly DONE | fetched=%d ingested=%d ===",
        total_fetched, total_ingested,
    )


# ════════════════════════════════════════════════════════════════════
# OpenAlex 반도체 논문 수집 (무료, API 키 불필요)
# https://docs.openalex.org/
# ════════════════════════════════════════════════════════════════════

OPENALEX_BASE = "https://api.openalex.org"

# 반도체 핵심 쿼리 (keyword search + 연도 필터)
OPENALEX_QUERIES: list[dict] = [
    {
        "id": "oa_hbm_tsv",
        "search": "HBM high bandwidth memory TSV 3D stacking",
        "tags": ["hbm", "tsv", "packaging"],
        "min_year": 2020,
    },
    {
        "id": "oa_tc_bonding",
        "search": "thermo-compression bonding flip chip advanced packaging semiconductor",
        "tags": ["tc_bonding", "packaging", "flip_chip"],
        "min_year": 2020,
    },
    {
        "id": "oa_hybrid_bonding",
        "search": "hybrid bonding Cu-Cu direct bonding chiplet wafer",
        "tags": ["hybrid_bonding", "chiplet", "wafer_bonding"],
        "min_year": 2020,
    },
    {
        "id": "oa_3d_nand",
        "search": "3D NAND flash memory vertical channel stacking",
        "tags": ["nand", "3d_nand", "flash_memory"],
        "min_year": 2020,
    },
    {
        "id": "oa_euv_litho",
        "search": "EUV extreme ultraviolet lithography high-NA semiconductor patterning",
        "tags": ["euv", "lithography", "patterning"],
        "min_year": 2020,
    },
    {
        "id": "oa_gaa_nanosheet",
        "search": "gate-all-around nanosheet transistor CMOS process technology",
        "tags": ["gaa", "nanosheet", "transistor"],
        "min_year": 2020,
    },
    {
        "id": "oa_cowos_interposer",
        "search": "CoWoS silicon interposer 2.5D packaging AI accelerator",
        "tags": ["cowos", "interposer", "packaging"],
        "min_year": 2021,
    },
    {
        "id": "oa_dram_scaling",
        "search": "DRAM cell capacitor process technology DDR5 LPDDR5",
        "tags": ["dram", "ddr5", "memory_scaling"],
        "min_year": 2020,
    },
    {
        "id": "oa_semiconductor_yield",
        "search": "semiconductor yield defect density process variation wafer fab",
        "tags": ["yield", "defect_density", "fab"],
        "min_year": 2020,
    },
    {
        "id": "oa_power_delivery",
        "search": "power delivery network semiconductor PDN decoupling capacitor BEOL",
        "tags": ["power_delivery", "beol", "packaging"],
        "min_year": 2021,
    },
    {
        "id": "oa_pim_cxl_memory",
        "search": "processing-in-memory CXL memory expansion bandwidth latency",
        "tags": ["pim", "cxl", "memory_system"],
        "min_year": 2020,
    },
    {
        "id": "oa_materials_reliability",
        "search": "semiconductor materials reliability electromigration failure analysis",
        "tags": ["materials", "reliability", "failure_analysis"],
        "min_year": 2020,
    },
    {
        "id": "oa_power_semiconductors",
        "search": "silicon carbide gallium nitride power semiconductor device",
        "tags": ["sic", "gan", "power_semiconductor"],
        "min_year": 2020,
    },
    {
        "id": "oa_eda_chiplet_ucie",
        "search": "EDA chiplet UCIe die-to-die architecture semiconductor",
        "tags": ["eda", "chiplet", "ucie"],
        "min_year": 2020,
    },
    {
        "id": "oa_advanced_substrate",
        "search": "glass substrate organic interposer advanced packaging semiconductor",
        "tags": ["glass_substrate", "interposer", "packaging"],
        "min_year": 2020,
    },
    {
        "id": "oa_metrology_inspection",
        "search": "metrology inspection defect detection semiconductor manufacturing",
        "tags": ["metrology", "inspection", "yield"],
        "min_year": 2020,
    },
    {
        "id": "oa_device_physics_2d",
        "search": "2D materials TMD graphene semiconductor device physics",
        "tags": ["device_physics", "2d_materials"],
        "min_year": 2020,
    },
]

# venue 가중치 (display_name 부분 매치)
OA_VENUE_WEIGHTS: dict[str, float] = {
    "electron devices meeting": 1.0,
    "iedm": 1.0,
    "solid-state circuits": 1.0,
    "isscc": 1.0,
    "vlsi": 0.95,
    "irps": 0.90,
    "interconnect technology": 0.88,
    "iitc": 0.88,
    "electronic components and technology": 0.85,
    "ectc": 0.85,
    "hot chips": 0.80,
    "spie": 0.78,
    "advanced lithography": 0.78,
    "design automation": 0.75,
    "ieee transactions on electron devices": 0.85,
    "journal of applied physics": 0.70,
    "nature electronics": 0.90,
    "ieee electron device letters": 0.85,
}


def _oa_venue_score(source_name: str | None) -> float:
    if not source_name:
        return 0.0
    s = source_name.lower()
    for kw, score in OA_VENUE_WEIGHTS.items():
        if kw in s:
            return score
    return 0.0


def _oa_paper_url(work: dict) -> str:
    doi = work.get("doi") or ""
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi}"
    oa_url = (work.get("open_access") or {}).get("oa_url") or ""
    if oa_url:
        return oa_url
    return work.get("id", "")


def _oa_extract_abstract(work: dict) -> str:
    """OpenAlex는 abstract를 inverted index로 저장 → 복원."""
    inv = work.get("abstract_inverted_index")
    if not inv:
        return ""
    # {word: [position, ...]} → 위치순 정렬 후 재구성
    try:
        word_pos: list[tuple[int, str]] = []
        for word, positions in inv.items():
            for pos in positions:
                word_pos.append((pos, word))
        word_pos.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_pos)
    except Exception:
        return ""


def _oa_evaluate_relevance(work: dict, tags: list[str]) -> float:
    """venue + citation + OA 점수."""
    source_name = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or ""
    venue_s = _oa_venue_score(source_name)
    cit = int(work.get("cited_by_count") or 0)
    cit_bonus = min(0.25, math.log1p(cit) / 20.0)
    oa_bonus = 0.05 if (work.get("open_access") or {}).get("is_oa") else 0.0
    return min(1.0, 0.5 + venue_s * 0.30 + cit_bonus + oa_bonus)


def _fetch_openalex_works(
    search: str,
    min_year: int,
    per_page: int = 100,
    max_pages: int = 5,
) -> list[dict]:
    """OpenAlex /works 검색 — 페이지네이션 포함."""
    all_works: list[dict] = []
    cursor = "*"
    email = os.getenv("OPENALEX_EMAIL", "taland797@gmail.com")
    filter_str = f"publication_year:{min_year}-"

    for page_num in range(max_pages):
        params = {
            "search": search,
            "filter": filter_str,
            "per-page": per_page,
            "cursor": cursor,
            "mailto": email,
            "select": (
                "id,doi,title,abstract_inverted_index,publication_year,"
                "primary_location,open_access,cited_by_count,"
                "authorships,concepts,topics"
            ),
        }
        try:
            timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "30"))
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                resp = client.get(f"{OPENALEX_BASE}/works", params=params)
                if resp.status_code == 429:
                    logger.warning("OpenAlex rate limit — sleeping 10s")
                    time.sleep(10)
                    continue
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            logger.warning("OpenAlex fetch failed | page=%d error=%s", page_num, e)
            break

        works = data.get("results") or []
        all_works.extend(works)

        next_cursor = (data.get("meta") or {}).get("next_cursor")
        if not next_cursor or len(works) < per_page:
            break
        cursor = next_cursor
        time.sleep(0.2)  # polite pool

    return all_works


def _ingest_oa_work(work: dict, query_id: str, tags: list[str]) -> bool:
    """단일 OpenAlex work → tech_documents."""
    oa_id = work.get("id") or ""
    if not oa_id:
        return False

    title = (work.get("title") or "").strip()
    abstract = _oa_extract_abstract(work)
    year = work.get("publication_year")
    pub_date_str = work.get("publication_date") or f"{year}-01-01" if year else None
    pub_dt = _parse_iso_dt(pub_date_str) if pub_date_str else None

    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    venue_name = src.get("display_name") or ""

    doi = work.get("doi") or ""
    oa_info = work.get("open_access") or {}
    oa_url = oa_info.get("oa_url") or ""
    cit = int(work.get("cited_by_count") or 0)

    authors = [
        (a.get("author") or {}).get("display_name", "")
        for a in (work.get("authorships") or [])[:10]
        if (a.get("author") or {}).get("display_name")
    ]

    concepts = [
        (c.get("display_name") or "")
        for c in (work.get("concepts") or [])[:8]
        if c.get("score", 0) > 0.3
    ]

    paper_url = doi if doi else oa_url if oa_url else oa_id
    if doi and not doi.startswith("http"):
        paper_url = f"https://doi.org/{doi}"

    confidence = _oa_evaluate_relevance(work, tags)
    all_tags = list(set(["openalex", query_id] + tags))

    extra = {
        "source": "openalex",
        "openalex_id": oa_id,
        "venue": venue_name,
        "year": year,
        "doi": doi,
        "oa_url": oa_url,
        "is_oa": oa_info.get("is_oa", False),
        "cited_by_count": cit,
        "authors": authors,
        "concepts": concepts,
        "query_id": query_id,
    }

    # arXiv PDF 연동: oa_url에 arxiv가 있으면 기존 파이프라인 활용
    arxiv_id_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^\s?#]+)", oa_url or "")
    if arxiv_id_match:
        abs_url = f"https://arxiv.org/abs/{arxiv_id_match.group(1).replace('.pdf', '')}"
        uid = _upsert_paper(
            source="arxiv",
            title=title or abs_url,
            url=abs_url,
            published_at=pub_dt,
            summary=abstract,
            tags=all_tags,
            confidence=confidence,
            extra=extra,
            content=None,
        )
        if _env_bool("ARXIV_DOWNLOAD_PDF", True):
            try:
                _process_pdf_and_chunks(uid, abs_url)
            except Exception as e:
                logger.debug("OA→arXiv PDF skip | error=%s", e)
    else:
        uid = _upsert_paper(
            source="openalex",
            title=title or paper_url,
            url=paper_url,
            published_at=pub_dt,
            summary=abstract,
            tags=all_tags,
            confidence=confidence,
            extra=extra,
            content=None,
        )

    return True


def collect_openalex_papers() -> None:
    """OpenAlex 반도체 논문 수집 (무료, 주간 실행 권장)."""
    logger.info("=== collect_openalex_papers START ===")

    max_pages = int(os.getenv("OPENALEX_MAX_PAGES", "3"))
    per_page = int(os.getenv("OPENALEX_PER_PAGE", "100"))
    query_limit = int(os.getenv("OPENALEX_QUERY_LIMIT", str(len(OPENALEX_QUERIES))))
    total_fetched = 0
    total_ingested = 0

    for q in OPENALEX_QUERIES[: max(1, query_limit)]:
        qid   = q["id"]
        search = q["search"]
        tags  = q.get("tags", [])
        min_yr = q.get("min_year", 2020)

        logger.info("OA query: id=%s min_year=%d", qid, min_yr)

        works = _fetch_openalex_works(search, min_year=min_yr, per_page=per_page, max_pages=max_pages)
        total_fetched += len(works)
        ingested_local = 0

        for work in works:
            try:
                ok = _ingest_oa_work(work, query_id=qid, tags=tags)
                if ok:
                    ingested_local += 1
                    total_ingested += 1
            except Exception as e:
                logger.warning("OA ingest failed | id=%s error=%s", work.get("id"), e)

        logger.info("OA query done | id=%s fetched=%d ingested=%d", qid, len(works), ingested_local)
        time.sleep(0.5)

    logger.info(
        "=== collect_openalex_papers DONE | fetched=%d ingested=%d ===",
        total_fetched, total_ingested,
    )


def collect_openalex_company_priority_papers() -> None:
    """Company-priority OpenAlex harvest focused on latest company-linked papers."""
    logger.info("=== collect_openalex_company_priority_papers START ===")

    max_pages = int(os.getenv("OPENALEX_COMPANY_MAX_PAGES", "2"))
    per_page = int(os.getenv("OPENALEX_COMPANY_PER_PAGE", "50"))
    query_limit = int(os.getenv("OPENALEX_COMPANY_QUERY_LIMIT", "12"))
    queries = _company_openalex_queries()[: max(1, query_limit)]
    total_fetched = 0
    total_ingested = 0

    for q in queries:
        qid = str(q["id"])
        search = str(q["search"])
        tags = q.get("tags", [])
        min_yr = int(q.get("min_year", 2022))
        logger.info("Company OA query: id=%s company=%s min_year=%d", qid, q.get("company_code", ""), min_yr)
        works = _fetch_openalex_works(search, min_year=min_yr, per_page=per_page, max_pages=max_pages)
        total_fetched += len(works)
        ingested_local = 0
        for work in works:
            try:
                if _ingest_oa_work(work, query_id=qid, tags=tags):
                    ingested_local += 1
                    total_ingested += 1
            except Exception as e:
                logger.warning("Company OA ingest failed | id=%s error=%s", work.get("id"), e)
        logger.info("Company OA query done | id=%s fetched=%d ingested=%d", qid, len(works), ingested_local)
        time.sleep(0.5)

    logger.info(
        "=== collect_openalex_company_priority_papers DONE | fetched=%d ingested=%d ===",
        total_fetched,
        total_ingested,
    )
