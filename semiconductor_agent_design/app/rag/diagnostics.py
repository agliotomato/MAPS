"""
RAG diagnostics.

Use this before judging RAG quality. It separates infrastructure failures
from actual retrieval-quality failures.
"""

from __future__ import annotations

import json
import time
from typing import Any


def _timed(name: str, fn) -> dict[str, Any]:
    t0 = time.time()
    try:
        value = fn()
        return {"name": name, "ok": True, "elapsed_sec": round(time.time() - t0, 2), "value": value}
    except Exception as e:
        return {"name": name, "ok": False, "elapsed_sec": round(time.time() - t0, 2), "error": str(e)}


def run_rag_diagnostics(load_model: bool = False) -> dict[str, Any]:
    checks = []

    def qdrant_collections():
        from app.db.qdrant import get_qdrant_client
        client = get_qdrant_client()
        cols = client.get_collections().collections
        return [c.name for c in cols]

    def semi_collection():
        from app.db.qdrant import SEMI_KNOWLEDGE_COLLECTION, get_qdrant_client
        client = get_qdrant_client()
        info = client.get_collection(SEMI_KNOWLEDGE_COLLECTION)
        return {"points_count": info.points_count, "vectors_count": info.vectors_count}

    def postgres_chunk_count():
        from app.db.postgres import get_pg_conn
        with get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM tech_document_chunks")
                return int(cur.fetchone()[0])

    def embedder_import():
        import FlagEmbedding  # noqa: F401
        return "FlagEmbedding import ok"

    def embedder_load():
        from app.rag.embedder import encode_query
        dense, sparse = encode_query("HBM hybrid bonding")
        return {"dense_dim": len(dense), "sparse_terms": len(sparse)}

    checks.append(_timed("qdrant_collections", qdrant_collections))
    checks.append(_timed("qdrant_semi_knowledge", semi_collection))
    checks.append(_timed("postgres_chunk_count", postgres_chunk_count))
    checks.append(_timed("embedder_import", embedder_import))
    if load_model:
        checks.append(_timed("embedder_load", embedder_load))

    return {"checks": checks}


def print_rag_diagnostics(load_model: bool = False) -> None:
    print(json.dumps(run_rag_diagnostics(load_model=load_model), ensure_ascii=False, indent=2))
