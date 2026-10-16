"""
Lightweight lexical fallback over stored chunk text.

This is a practical fallback for diagnostics and early RAG evaluation when
the full hybrid stack is unavailable or too expensive to use.
"""

from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

_GENERIC_QUERY_TERMS = {
    "memory",
    "semiconductor",
    "chip",
    "standard",
    "roadmap",
    "power",
    "yield",
    "reliability",
    "inspection",
    "metrology",
    "revenue",
    "results",
    "official",
    "company",
    "investor",
    "relations",
    "technical",
    "evidence",
}


def _terms(text: str) -> list[str]:
    raw = re.findall(r"[\w+\-]{2,}", text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "into",
        "using",
        "official",
        "company",
        "investor",
        "relations",
        "technical",
        "evidence",
    }
    return [t for t in raw if t not in stop]


def _term_weight(term: str) -> float:
    if term in _GENERIC_QUERY_TERMS:
        return 0.6
    if len(term) <= 4:
        return 2.0
    if "-" in term or any(ch.isdigit() for ch in term):
        return 1.4
    return 1.0


def _query_terms(text: str) -> Counter[str]:
    return Counter({term: _term_weight(term) for term in _terms(text)})


def _ranked_query_terms(text: str, limit: int = 8) -> list[str]:
    items = sorted(_query_terms(text).items(), key=lambda x: (-x[1], x[0]))
    return [term for term, _ in items[:limit]]


def _score(query_terms: Counter[str], payload: dict[str, Any]) -> float:
    title = str(payload.get("title", ""))
    text = str(payload.get("text", ""))
    tags = " ".join(str(x) for x in payload.get("tags", []) or [])
    hay = f"{title} {title} {tags} {text}".lower()
    score = 0.0
    for term, weight in query_terms.items():
        if term in hay:
            score += weight * (2.0 if term in title.lower() else 1.0)
    if not score:
        return 0.0
    length_penalty = math.log1p(max(100, len(text))) / 8.0
    return score / length_penalty


def search_lexical_payload(query: str, top_k: int = 10, limit: int = 5000) -> list[dict]:
    hits = _search_qdrant_payload(query, top_k=top_k, limit=limit)
    if os.getenv("RAG_LEXICAL_INCLUDE_POSTGRES", "1").strip().lower() in {"1", "true", "yes", "y"}:
        hits.extend(_search_postgres_chunks(query, top_k=max(top_k, 50), limit=limit))

    dedup: dict[str, dict[str, Any]] = {}
    for item in hits:
        key = str(item.get("doc_uid") or item.get("id"))
        prev = dedup.get(key)
        if prev is None or float(item.get("score", 0.0)) > float(prev.get("score", 0.0)):
            dedup[key] = item
    out = list(dedup.values())
    out.sort(key=lambda x: -float(x.get("score", 0.0)))
    return out[:top_k]


def _search_qdrant_payload(query: str, top_k: int = 10, limit: int = 5000) -> list[dict]:
    from app.db.qdrant import SEMI_KNOWLEDGE_COLLECTION, get_qdrant_client

    if os.getenv("RAG_QDRANT_DISABLED", "0").strip().lower() in {"1", "true", "yes", "y"}:
        return []
    client = get_qdrant_client()
    qterms = _query_terms(query)
    if not qterms:
        return []

    hits: list[dict[str, Any]] = []
    offset = None
    scanned = 0
    while scanned < limit:
        try:
            points, offset = client.scroll(
                collection_name=SEMI_KNOWLEDGE_COLLECTION,
                limit=min(256, limit - scanned),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning("Qdrant scroll failed, using Postgres fallback only: %s", e)
            return []
        if not points:
            break
        scanned += len(points)
        for p in points:
            payload = p.payload or {}
            score = _score(qterms, payload)
            if score <= 0:
                continue
            hits.append({"id": str(p.id), "score": score, **payload})
        if offset is None:
            break

    hits.sort(key=lambda x: -float(x.get("score", 0.0)))
    return hits[:top_k]


def _search_postgres_chunks(query: str, top_k: int = 10, limit: int = 5000) -> list[dict]:
    from app.db.postgres import get_pg_conn

    terms = _ranked_query_terms(query)
    if not terms:
        return []

    sql = """
        SELECT
            c.doc_uid,
            c.chunk_index,
            c.chunk_text,
            d.source,
            d.source_type,
            d.title,
            d.published_at,
            d.url,
            d.tags,
            d.extra
        FROM tech_document_chunks c
        JOIN tech_documents d ON d.doc_uid = c.doc_uid
        WHERE
    """
    params: list[Any] = []
    where_parts = []
    for term in terms:
        pattern = f"%{term}%"
        where_parts.append(
            "("
            "lower(coalesce(d.title, '')) LIKE %s OR "
            "lower(coalesce(d.summary, '')) LIKE %s OR "
            "lower(coalesce(c.chunk_text, '')) LIKE %s OR "
            "lower(coalesce(d.tags::text, '')) LIKE %s"
            ")"
        )
        params.extend([pattern, pattern, pattern, pattern])

    sql += " OR ".join(where_parts)
    sql += """
        ORDER BY COALESCE(d.published_at, d.collected_at) DESC, c.doc_uid, c.chunk_index
        LIMIT %s
    """
    params.append(limit)
    hits: list[dict[str, Any]] = []
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        for row in rows:
            rec = dict(zip(cols, row))
            extra = rec.get("extra") or {}
            tags = rec.get("tags") or []
            published_at = rec.get("published_at")
            text = str(rec.get("chunk_text") or "")
            payload = {
                "id": f"{rec.get('doc_uid')}:{rec.get('chunk_index')}",
                "doc_uid": rec.get("doc_uid"),
                "chunk_index": rec.get("chunk_index"),
                "source": rec.get("source") or "",
                "source_type": rec.get("source_type") or "",
                "title": rec.get("title") or "",
                "published_at": published_at.isoformat() if published_at else "",
                "url": rec.get("url"),
                "year": published_at.year if published_at else int(extra.get("year") or 0),
                "company": extra.get("company_code") or extra.get("company") or "",
                "domain": (extra.get("domain_hits") or [""])[0] if isinstance(extra.get("domain_hits"), list) else "",
                "tags": tags,
                "text": text,
            }
            score = _score(_query_terms(query), payload)
            if score <= 0:
                continue
            payload["score"] = score
            hits.append(payload)
    except Exception:
        return []

    # Also search document-level metadata so summary-only sources are not missed.
    doc_sql = """
        SELECT
            d.doc_uid,
            0 AS chunk_index,
            COALESCE(d.content, d.summary, d.title, '') AS chunk_text,
            d.source,
            d.source_type,
            d.title,
            d.published_at,
            d.url,
            d.tags,
            d.extra
        FROM tech_documents d
        WHERE
    """
    doc_params: list[Any] = []
    doc_where_parts = []
    for term in terms:
        pattern = f"%{term}%"
        doc_where_parts.append(
            "("
            "lower(coalesce(d.title, '')) LIKE %s OR "
            "lower(coalesce(d.summary, '')) LIKE %s OR "
            "lower(coalesce(d.content, '')) LIKE %s OR "
            "lower(coalesce(d.tags::text, '')) LIKE %s"
            ")"
        )
        doc_params.extend([pattern, pattern, pattern, pattern])
    doc_sql += " OR ".join(doc_where_parts)
    doc_sql += """
        ORDER BY COALESCE(d.published_at, d.collected_at) DESC, d.doc_uid
        LIMIT %s
    """
    doc_params.append(max(50, top_k * 4))
    try:
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(doc_sql, doc_params)
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
        for row in rows:
            rec = dict(zip(cols, row))
            extra = rec.get("extra") or {}
            tags = rec.get("tags") or []
            published_at = rec.get("published_at")
            text = str(rec.get("chunk_text") or "")
            payload = {
                "id": f"{rec.get('doc_uid')}:doc",
                "doc_uid": rec.get("doc_uid"),
                "chunk_index": 0,
                "source": rec.get("source") or "",
                "source_type": rec.get("source_type") or "",
                "title": rec.get("title") or "",
                "published_at": published_at.isoformat() if published_at else "",
                "url": rec.get("url"),
                "year": published_at.year if published_at else int(extra.get("year") or 0),
                "company": extra.get("company_code") or extra.get("company") or "",
                "domain": (extra.get("domain_hits") or [""])[0] if isinstance(extra.get("domain_hits"), list) else "",
                "tags": tags,
                "text": text,
            }
            score = _score(_query_terms(query), payload)
            if score <= 0:
                continue
            payload["score"] = score
            hits.append(payload)
    except Exception:
        pass

    hits.sort(key=lambda x: -float(x.get("score", 0.0)))
    return hits[:top_k]


def search_multi_query_lexical(queries: list[str], top_k: int = 10) -> list[dict]:
    combined: dict[str, dict[str, Any]] = {}
    scores: Counter[str] = Counter()
    for q in queries:
        for rank, item in enumerate(search_lexical_payload(q, top_k=max(20, top_k * 3)), start=1):
            pid = item["id"]
            scores[pid] += 1.0 / (20 + rank)
            combined.setdefault(pid, item)
    out = []
    for pid, score in scores.most_common(top_k):
        item = dict(combined[pid])
        item["score"] = float(score)
        out.append(item)
    return out
