"""
Qdrant client & collection management.

Collections
-----------
Legacy (per-source, dense only):
    irds_chunks, jedec_chunks, paper_chunks, tech_blog_chunks, news_chunks

Unified BGE-M3 hybrid (dense 1024 + sparse BM25):
    semi_knowledge   ← all chunks land here, filtered by payload
"""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

_log = logging.getLogger(__name__)

# ── dimension constants ──────────────────────────────────────────────────────
DENSE_DIM = 1024          # BGE-M3 dense output dimension
LEGACY_DIM = int(os.getenv("EMBED_DIM", "384"))   # old MiniLM collections

# Name used for vectors in semi_knowledge
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# The unified collection name
SEMI_KNOWLEDGE_COLLECTION = "semi_knowledge"


def get_qdrant_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key or None)
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


# ── legacy single-vector collections ────────────────────────────────────────

def ensure_collection(name: str, size: int | None = None) -> None:
    """Create a simple dense-only collection if it doesn't exist (legacy support)."""
    if name == SEMI_KNOWLEDGE_COLLECTION:
        ensure_semi_knowledge()
        return

    if size is None:
        env_key = f"QDRANT_{name.upper().replace('-', '_')}_DIM"
        size = int(os.getenv(env_key, str(LEGACY_DIM)))

    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if name in existing:
        return

    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE),
    )
    _log.info("Created legacy collection '%s' dim=%d", name, size)


# ── unified BGE-M3 hybrid collection ────────────────────────────────────────

def ensure_semi_knowledge() -> None:
    """
    Create the `semi_knowledge` collection with named vectors:
      - dense:  1024-dim COSINE
      - sparse: Qdrant sparse (SPLADE / BGE-M3 lexical weights)

    Payload schema (all indexed for fast filtering):
      source_type: str   → 'paper' | 'news' | 'patent' | 'blog' | 'standard' | 'report'
      source:      str   → origin identifier (e.g. 'arxiv', 'openalex', 'dart')
      company:     str   → company_code or '' for market-wide
      domain:      str   → e.g. 'hbm', 'packaging', 'euv', 'dram', 'general'
      year:        int   → publication year (0 if unknown)
      doc_uid:     str   → FK to tech_documents.doc_uid
      chunk_index: int
      title:       str
      tags:        list[str]
    """
    client = get_qdrant_client()
    existing = {c.name for c in client.get_collections().collections}

    if SEMI_KNOWLEDGE_COLLECTION in existing:
        _log.info("Collection '%s' already exists — skipping.", SEMI_KNOWLEDGE_COLLECTION)
        return

    client.create_collection(
        collection_name=SEMI_KNOWLEDGE_COLLECTION,
        vectors_config={
            DENSE_VECTOR_NAME: models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            SPARSE_VECTOR_NAME: models.SparseVectorParams(
                index=models.SparseIndexParams(
                    on_disk=False,
                )
            ),
        },
    )

    # Create payload indexes for fast filtered search
    payload_indexes = [
        ("source_type", models.PayloadSchemaType.KEYWORD),
        ("source",      models.PayloadSchemaType.KEYWORD),
        ("company",     models.PayloadSchemaType.KEYWORD),
        ("domain",      models.PayloadSchemaType.KEYWORD),
        ("year",        models.PayloadSchemaType.INTEGER),
        ("doc_uid",     models.PayloadSchemaType.KEYWORD),
    ]
    for field, schema in payload_indexes:
        client.create_payload_index(
            collection_name=SEMI_KNOWLEDGE_COLLECTION,
            field_name=field,
            field_schema=schema,
        )

    _log.info(
        "Created '%s' collection: dense=%d + sparse, with %d payload indexes.",
        SEMI_KNOWLEDGE_COLLECTION, DENSE_DIM, len(payload_indexes),
    )


def upsert_chunk(
    *,
    point_id: str,
    dense: list[float],
    sparse: dict[int, float],
    payload: dict,
) -> None:
    """Upsert a single chunk into semi_knowledge."""
    client = get_qdrant_client()
    client.upsert(
        collection_name=SEMI_KNOWLEDGE_COLLECTION,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense,
                    SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=list(sparse.keys()),
                        values=list(sparse.values()),
                    ),
                },
                payload=payload,
            )
        ],
    )


def upsert_chunks_batch(
    points: list[dict],
) -> None:
    """
    Batch upsert into semi_knowledge.
    Each element in `points`:
        {
            "id":      str (UUID or deterministic hash),
            "dense":   list[float],
            "sparse":  dict[int, float],
            "payload": dict,
        }
    """
    client = get_qdrant_client()
    qdrant_points = [
        models.PointStruct(
            id=p["id"],
            vector={
                DENSE_VECTOR_NAME: p["dense"],
                SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=list(p["sparse"].keys()),
                    values=list(p["sparse"].values()),
                ),
            },
            payload=p["payload"],
        )
        for p in points
    ]
    client.upsert(collection_name=SEMI_KNOWLEDGE_COLLECTION, points=qdrant_points)
