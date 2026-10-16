"""
Hybrid Retriever (dense + sparse) for semi_knowledge
------------------------------------------------------
Uses BGE-M3 to encode the query, then performs:
  1. Dense ANN search (COSINE semantic similarity)
  2. Sparse search (BM25 lexical overlap via BGE-M3 lexical weights)
  3. Reciprocal Rank Fusion (RRF) to merge results

Supports payload filters for targeted retrieval:
    source_type, company, domain, year_min / year_max

Usage
-----
    from app.rag.retriever import search

    results = search(
        "HBM3 thermal interface resistance copper pillar",
        top_k=10,
        filter_source_type="paper",
        filter_domain="hbm",
    )
    for r in results:
        print(r["score"], r["title"], r["chunk_index"])
        print(r["text"][:300])
"""

from __future__ import annotations

import logging
import os
from typing import Any

from qdrant_client.http import models

from app.db.qdrant import (
    DENSE_VECTOR_NAME,
    SEMI_KNOWLEDGE_COLLECTION,
    SPARSE_VECTOR_NAME,
    get_qdrant_client,
)
from app.rag.lexical import search_lexical_payload

_log = logging.getLogger(__name__)

_PREFETCH_LIMIT = int(os.getenv("RAG_PREFETCH_LIMIT", "50"))
_DEFAULT_TOP_K  = int(os.getenv("RAG_TOP_K", "10"))
_RRF_K          = float(os.getenv("RAG_RRF_K", "60"))


def _build_filter(
    source_type: str | None = None,
    source: str | None = None,
    company: str | None = None,
    domain: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
) -> models.Filter | None:
    must: list[models.Condition] = []

    if source_type:
        must.append(models.FieldCondition(
            key="source_type", match=models.MatchValue(value=source_type)))
    if source:
        must.append(models.FieldCondition(
            key="source", match=models.MatchValue(value=source)))
    if company:
        must.append(models.FieldCondition(
            key="company", match=models.MatchValue(value=company)))
    if domain:
        must.append(models.FieldCondition(
            key="domain", match=models.MatchValue(value=domain)))
    if year_min is not None:
        must.append(models.FieldCondition(
            key="year", range=models.Range(gte=year_min)))
    if year_max is not None:
        must.append(models.FieldCondition(
            key="year", range=models.Range(lte=year_max)))

    return models.Filter(must=must) if must else None


def _rrf_merge(
    dense_hits: list[Any],
    sparse_hits: list[Any],
    k: float = _RRF_K,
) -> list[dict]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        payloads[pid] = hit.payload or {}

    for rank, hit in enumerate(sparse_hits, start=1):
        pid = str(hit.id)
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
        if pid not in payloads:
            payloads[pid] = hit.payload or {}

    return [
        {"id": pid, "score": sc, **payloads.get(pid, {})}
        for pid, sc in sorted(scores.items(), key=lambda x: -x[1])
    ]


def _apply_local_filters(results: list[dict], **filter_kwargs) -> list[dict]:
    def _match(item: dict) -> bool:
        if filter_kwargs.get("filter_source_type") and str(item.get("source_type", "")) != str(filter_kwargs["filter_source_type"]):
            return False
        if filter_kwargs.get("filter_source") and str(item.get("source", "")) != str(filter_kwargs["filter_source"]):
            return False
        if filter_kwargs.get("filter_company") and str(item.get("company", "")) != str(filter_kwargs["filter_company"]):
            return False
        if filter_kwargs.get("filter_domain") and str(item.get("domain", "")) != str(filter_kwargs["filter_domain"]):
            return False
        year = item.get("year")
        if filter_kwargs.get("filter_year_min") is not None and (year is None or int(year) < int(filter_kwargs["filter_year_min"])):
            return False
        if filter_kwargs.get("filter_year_max") is not None and (year is None or int(year) > int(filter_kwargs["filter_year_max"])):
            return False
        return True

    return [item for item in results if _match(item)]


def search(
    query: str,
    top_k: int = _DEFAULT_TOP_K,
    *,
    filter_source_type: str | None = None,
    filter_source: str | None = None,
    filter_company: str | None = None,
    filter_domain: str | None = None,
    filter_year_min: int | None = None,
    filter_year_max: int | None = None,
) -> list[dict]:
    """
    Hybrid search: dense + sparse → RRF merge.
    Returns list of payload dicts sorted by RRF score.
    Each dict includes: id, score, doc_uid, chunk_index, title, text,
                        source_type, source, company, domain, year, tags
    """
    dense_search_enabled = os.getenv("RAG_QDRANT_DISABLED", "0").strip().lower() not in {"1", "true", "yes", "y"}
    dense, sparse = None, None
    if dense_search_enabled:
        try:
            from app.rag.embedder import encode_query  # type: ignore
            dense, sparse = encode_query(query)
        except Exception as e:
            _log.warning("dense/sparse encode unavailable, using lexical fallback: %s", e)
            dense_search_enabled = False

    local_filters = {
        "filter_source_type": filter_source_type,
        "filter_source": filter_source,
        "filter_company": filter_company,
        "filter_domain": filter_domain,
        "filter_year_min": filter_year_min,
        "filter_year_max": filter_year_max,
    }

    qfilter = _build_filter(
        source_type=filter_source_type,
        source=filter_source,
        company=filter_company,
        domain=filter_domain,
        year_min=filter_year_min,
        year_max=filter_year_max,
    )

    if not dense_search_enabled:
        return _apply_local_filters(search_lexical_payload(query, top_k=max(top_k, _PREFETCH_LIMIT)), **local_filters)[:top_k]

    try:
        client = get_qdrant_client()

        # qdrant-client v2 API (query_points)
        dense_hits = client.query_points(
            collection_name=SEMI_KNOWLEDGE_COLLECTION,
            query=dense,
            using=DENSE_VECTOR_NAME,
            limit=_PREFETCH_LIMIT,
            query_filter=qfilter,
            with_payload=True,
        ).points

        sparse_hits = client.query_points(
            collection_name=SEMI_KNOWLEDGE_COLLECTION,
            query=models.SparseVector(
                indices=list(sparse.keys()),
                values=list(sparse.values()),
            ),
            using=SPARSE_VECTOR_NAME,
            limit=_PREFETCH_LIMIT,
            query_filter=qfilter,
            with_payload=True,
        ).points

        merged = _rrf_merge(dense_hits, sparse_hits)
        if merged:
            return merged[:top_k]
    except Exception as e:
        _log.warning("hybrid search failed, using lexical fallback: %s", e)

    return _apply_local_filters(search_lexical_payload(query, top_k=max(top_k, _PREFETCH_LIMIT)), **local_filters)[:top_k]


def search_multi_query(
    queries: list[str],
    top_k: int = _DEFAULT_TOP_K,
    **filter_kwargs,
) -> list[dict]:
    """여러 쿼리를 RRF로 합산. 다각도 검색에 사용."""
    from collections import defaultdict

    all_scored: dict[str, float] = defaultdict(float)
    all_payloads: dict[str, dict] = {}

    for q in queries:
        results = search(q, top_k=_PREFETCH_LIMIT, **filter_kwargs)
        for r_rank, r in enumerate(results, start=1):
            pid = r["id"]
            all_scored[pid] += 1.0 / (_RRF_K + r_rank)
            if pid not in all_payloads:
                all_payloads[pid] = {k: v for k, v in r.items()
                                     if k not in ("id", "score")}

    return [
        {"id": pid, "score": sc, **all_payloads.get(pid, {})}
        for pid, sc in sorted(all_scored.items(), key=lambda x: -x[1])[:top_k]
    ]


def get_context_for_llm(
    query: str,
    top_k: int = 8,
    max_chars: int = 6000,
    **filter_kwargs,
) -> str:
    """LLM 프롬프트용 컨텍스트 문자열 생성."""
    results = search(query, top_k=top_k, **filter_kwargs)
    parts: list[str] = []
    total = 0
    for r in results:
        source_type = r.get("source_type", "")
        src   = r.get("source", "")
        company = r.get("company", "")
        domain = r.get("domain", "")
        published_at = r.get("published_at", "")
        year  = r.get("year", "")
        title = r.get("title", "")
        text  = r.get("text", "")
        block = f"[{source_type} | {src} | {published_at or year} | company={company} | domain={domain}] {title}\n{text}\n"
        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                parts.append(block[:remaining])
            break
        parts.append(block)
        total += len(block)

    return "\n---\n".join(parts)
