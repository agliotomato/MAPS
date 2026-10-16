"""
BGE-M3 re-indexer for the `semi_knowledge` Qdrant collection.

Reads chunk records from Postgres, embeds them with BGE-M3 (dense+sparse),
and upserts them into the unified hybrid collection.

The overnight pipeline uses `index_all_chunks_safe()` so a temporary embedder
runtime failure does not stop collection, event building, or backtesting.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from datetime import timezone
from typing import Any

from app.db.postgres import get_pg_conn
from app.db.qdrant import ensure_semi_knowledge, upsert_chunks_batch
from app.rag.embedder import EmbedderUnavailableError, encode_batch

_log = logging.getLogger(__name__)

BATCH_SIZE = int(os.getenv("INDEXER_BATCH_SIZE", "16"))

_TAG_TO_DOMAIN: dict[str, str] = {
    "hbm": "hbm",
    "tc_bonding": "packaging",
    "hybrid_bonding": "packaging",
    "3d_nand": "nand",
    "euv": "litho",
    "gaa": "logic",
    "cowos": "packaging",
    "dram": "dram",
    "semiconductor": "general",
    "patent": "patent",
    "news": "news",
    "blog": "blog",
    "standard": "standard",
}


def _chunk_uid_to_point_id(doc_uid: str, chunk_index: int) -> str:
    raw = f"{doc_uid}:{chunk_index}"
    digest = hashlib.sha256(raw.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=digest))


def _infer_domain(tags: list[str], source_type: str) -> str:
    for tag in tags:
        tag_lower = tag.lower()
        for key, domain in _TAG_TO_DOMAIN.items():
            if key in tag_lower:
                return domain
    return source_type if source_type in ("patent", "news", "blog", "standard") else "general"


def _fetch_all_chunks(conn) -> list[dict[str, Any]]:
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
        ORDER BY c.doc_uid, c.chunk_index
    """
    try:
        from psycopg.rows import dict_row

        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql)
            return [dict(r) for r in cur.fetchall()]
    except Exception:
        pass

    import psycopg2.extras

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        return [dict(r) for r in cur.fetchall()]


def _build_payload(row: dict[str, Any]) -> dict[str, Any]:
    tags: list[str] = row.get("tags") or []
    extra: dict[str, Any] = row.get("extra") or {}
    published_at = row.get("published_at")
    year = published_at.astimezone(timezone.utc).year if published_at else 0
    company = extra.get("company_code") or extra.get("company") or ""
    domain = _infer_domain(tags, row["source_type"])
    return {
        "doc_uid": row["doc_uid"],
        "chunk_index": row["chunk_index"],
        "source": row["source"],
        "source_type": row["source_type"],
        "title": row["title"],
        "published_at": published_at.isoformat() if published_at else None,
        "url": row.get("url"),
        "year": year,
        "company": company,
        "domain": domain,
        "tags": tags,
        "text": row["chunk_text"],
    }


def index_all_chunks(force: bool = False) -> int:
    """
    Embed and upsert all chunks into `semi_knowledge`.

    Returns:
        total number of chunks indexed in this run.
    """
    del force  # reserved for future partial-reindex logic
    ensure_semi_knowledge()

    _log.info("Fetching all chunks from Postgres.")
    with get_pg_conn() as conn:
        rows = _fetch_all_chunks(conn)

    total = len(rows)
    _log.info("Total chunks to index: %d", total)

    indexed = 0
    for batch_start in range(0, total, BATCH_SIZE):
        batch = rows[batch_start : batch_start + BATCH_SIZE]
        texts = [row["chunk_text"] for row in batch]
        embeddings = encode_batch(texts)

        points = []
        for row, (dense, sparse) in zip(batch, embeddings, strict=False):
            points.append(
                {
                    "id": _chunk_uid_to_point_id(row["doc_uid"], row["chunk_index"]),
                    "dense": dense,
                    "sparse": sparse,
                    "payload": _build_payload(row),
                }
            )

        upsert_chunks_batch(points)
        indexed += len(batch)

        if indexed % 100 == 0 or indexed == total:
            _log.info("Indexed %d / %d chunks.", indexed, total)

    _log.info("Re-indexing complete. Total indexed: %d", indexed)
    return indexed


def index_all_chunks_safe() -> dict[str, Any]:
    """
    Safe wrapper for overnight automation.

    If the BGE-M3 runtime is unavailable, we keep the rest of the pipeline
    moving and explicitly report that the system stayed in lexical-only mode.
    """
    try:
        indexed = index_all_chunks()
        return {
            "mode": "hybrid",
            "indexed": indexed,
            "skipped": False,
            "reason": "",
        }
    except EmbedderUnavailableError as exc:
        _log.warning("Hybrid indexing unavailable; continuing with lexical-only retrieval | %s", exc)
        return {
            "mode": "lexical_only",
            "indexed": 0,
            "skipped": True,
            "reason": str(exc),
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = index_all_chunks_safe()
    print(result)
