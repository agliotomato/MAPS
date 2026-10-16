"""
Vector DB Repository — 정성 데이터 RAG 파이프라인

핵심 함수:
    ingest_text_to_vector_db() — 텍스트 청킹 → 임베딩 → ChromaDB 저장
    query_vector_db()          — 유사도 검색으로 관련 컨텍스트 반환

추가 클래스:
    ChromaReportStore — news_tool.ReportStoreProtocol 구현체
                        set_report_store()로 InMemoryReportStore를 교체할 때 사용
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from macro_agent.db.vector.client import get_collection
from macro_agent.db.vector.smart_chunker import smart_chunk
from macro_agent.tools.news_tool import ReportStoreProtocol

logger = logging.getLogger(__name__)


# ── 리포트용 청킹 설정 (뉴스는 smart_chunk 사용) ──────────────────────
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", "。", " ", ""],
    length_function=len,
)


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────

def _content_id(content: str, metadata: dict) -> str:
    """
    (content + source + published_date) SHA-256 해시를 청크 ID로 사용.
    동일 문서를 재적재해도 중복 문서가 생성되지 않습니다.
    """
    key = f"{content}::{metadata.get('source_name', '')}::{metadata.get('published_date', '')}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _build_chunk_metadata(
    base_meta: dict,
    chunk_index: int,
    total_chunks: int,
    content_hash: str,
) -> dict:
    """
    ChromaDB 메타데이터는 str/int/float/bool만 허용.
    list·dict는 직렬화하여 저장합니다.
    """
    tags_raw = base_meta.get("tags", [])
    tags_str = ",".join(tags_raw) if isinstance(tags_raw, list) else str(tags_raw)

    return {
        "source_type": str(base_meta.get("source_type", "unknown")),
        "source_name": str(base_meta.get("source_name", "")),
        "url": str(base_meta.get("url", "")),
        "published_date": str(base_meta.get("published_date", "")),
        "date_source": str(base_meta.get("date_source") or "unknown"),
        "text_source": str(base_meta.get("text_source") or "unknown"),
        "title": str(base_meta.get("title") or ""),
        "retrieved_at": str(base_meta.get("retrieved_at") or ""),
        "raw_sha256": str(base_meta.get("raw_sha256") or ""),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "tags": tags_str,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "content_hash": content_hash,
    }


# ── 핵심 공개 함수 ────────────────────────────────────────────────────

def ingest_text_to_vector_db(
    text: str,
    metadata: dict[str, Any],
    source_type: str = "report",
    lang: str = "en",
) -> int:
    """
    텍스트를 청킹·임베딩하여 ChromaDB에 저장합니다.

    처리 흐름:
        1. RecursiveCharacterTextSplitter로 chunk_size=800 청크 분할
        2. 청크별 SHA-256 ID 생성 (중복 ingestion 방지)
        3. ChromaDB 컬렉션에 배치 upsert (add → 중복 ID는 update)

    Args:
        text:        저장할 원문 텍스트 (최소 20자 이상)
        metadata:    문서 메타데이터 dict
                     권장 키: source_name, published_date, url, tags
        source_type: 'report' (분석 리포트) | 'news' (뉴스 기사)

    Returns:
        실제 저장된 청크 수

    Raises:
        ValueError: 텍스트가 너무 짧거나 비어있는 경우
    """
    text = text.strip()
    if len(text) < 20:
        raise ValueError(f"텍스트가 너무 짧습니다 (현재 {len(text)}자, 최소 20자 이상 필요).")

    enriched_meta = {**metadata, "source_type": source_type}
    content_hash = _content_id(text, enriched_meta)

    # 1. 청킹 — 뉴스는 스마트 청킹, 리포트는 고정 분할
    if source_type == "news":
        chunks = smart_chunk(text, lang=lang)
    else:
        chunks = _TEXT_SPLITTER.split_text(text)
    if not chunks:
        logger.warning("청킹 결과가 비어있습니다. 원문 길이: %d자", len(text))
        return 0

    total = len(chunks)

    # 2. ID 및 메타데이터 생성
    ids = [f"{content_hash}-{i:04d}" for i in range(total)]
    metadatas = [
        _build_chunk_metadata(enriched_meta, i, total, content_hash)
        for i in range(total)
    ]

    # 3. ChromaDB upsert (기존 ID는 업데이트, 신규는 삽입)
    collection = get_collection()
    collection.upsert(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info(
        "Vector DB 적재 완료 — source_type=%s, source=%s, 청크=%d개 (hash=%s…)",
        source_type,
        enriched_meta.get("source_name", ""),
        total,
        content_hash[:8],
    )
    return total


def query_vector_db(
    query: str,
    n_results: int = 5,
    filter_source_type: str | None = None,
    min_content_length: int = 30,
    published_after: str | None = None,
) -> list[dict[str, Any]]:
    """
    질의 텍스트와 유사도가 높은 청크를 ChromaDB에서 검색합니다.

    filter_source_type 지정 시 ChromaDB where 필터 대신 Python 레벨 필터를 사용합니다.
    (ChromaDB 1.5.x에서 where 필터 호환성 문제가 있어 충분한 pool을 먼저 가져온 후 필터링)

    Args:
        query:              검색할 자연어 질의
        n_results:          반환할 최대 결과 수 (기본값: 5)
        filter_source_type: 'report' | 'news' | None (전체)
        min_content_length: 최소 콘텐츠 길이 필터 (너무 짧은 청크 제거)
        published_after:    "YYYY-MM-DD" — 이 날짜 이후 발행된 문서만 반환.
                            None이면 날짜 필터 없음.
                            발행일 미기록 문서(date_source="unknown")는 포함.

    Returns:
        [{"content", "metadata", "distance", "relevance_score"}, ...]
        metadata에 "published_date", "date_source" 필드가 포함됩니다.
    """
    if not query.strip():
        return []

    collection = get_collection()
    total = collection.count()
    if total == 0 or n_results <= 0:
        return []

    # 필터 적용 시 충분한 pool 확보.
    # - source_type 필터: 뉴스(3436청크)와 리포트(1564청크)의 임베딩 분포가 달라
    #   유사도 상위에 한 타입이 몰릴 수 있으므로 pool을 크게 잡아야 함.
    #   실측 기준 top200 중 리포트 ~8개 → 리포트 10건 보장하려면 300+ pool 필요.
    # - published_after 날짜 필터: 기간 내 비율에 따라 30배 이상이 필요할 수 있음.
    has_filter = bool(filter_source_type or published_after)
    if has_filter:
        fetch_n = min(max(n_results * 30, 300), total)
    else:
        fetch_n = min(n_results, total)

    raw = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        include=["documents", "metadatas", "distances"],
    )

    docs  = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    results: list[dict[str, Any]] = []
    for doc, meta, dist in zip(docs, metas, dists):
        # ── 길이 필터 ─────────────────────────────────────────────────
        if len(doc) < min_content_length:
            continue

        # ── source_type 필터 ──────────────────────────────────────────
        if filter_source_type and meta.get("source_type") != filter_source_type:
            continue

        # ── 날짜 필터 ─────────────────────────────────────────────────
        # published_after 지정 시:
        #   - published_date가 있으면 비교 (이후 날짜만 통과)
        #   - published_date가 없거나 "" 이면 포함 (날짜 미기록 문서는 제외하지 않음)
        if published_after:
            doc_date = meta.get("published_date", "")
            if doc_date and doc_date < published_after:
                continue

        results.append({
            "content":         doc,
            "metadata":        meta,
            "distance":        round(float(dist), 4),
            "relevance_score": round(1.0 - float(dist), 4),
        })
        if len(results) >= n_results:
            break

    logger.info(
        "Vector DB 쿼리 완료 — query=%r, 반환=%d건 (source=%s, after=%s)",
        query[:40], len(results), filter_source_type or "all", published_after or "none",
    )
    return results


# ── news_tool.ReportStoreProtocol 구현체 ──────────────────────────────

class ChromaReportStore:
    """
    ChromaDB를 백엔드로 사용하는 ReportStoreProtocol 구현체.

    news_tool.set_report_store(ChromaReportStore())로 교체하면
    scraped_news_and_reports()의 RAG 검색이 ChromaDB를 사용합니다.

    Usage (애플리케이션 초기화 시 1회):
        from macro_agent.db.vector.repository import ChromaReportStore
        from macro_agent.tools.news_tool import set_report_store

        set_report_store(ChromaReportStore())
    """

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        """
        news_tool.upload_report_to_rag()에서 호출됩니다.
        문서 content + metadata를 Vector DB에 저장합니다.
        """
        for doc in documents:
            content = doc.get("content", "")
            meta = doc.get("metadata", {})
            ingest_text_to_vector_db(
                text=content,
                metadata={
                    "source_name": meta.get("source", "user_upload"),
                    "published_date": meta.get("report_date", ""),
                    "tags": meta.get("tags", []),
                    "url": "",
                },
                source_type="report",
            )

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        """
        news_tool.scraped_news_and_reports()의 RAG 검색에서 호출됩니다.
        ReportStoreProtocol이 기대하는 형식으로 변환합니다.
        """
        raw_results = query_vector_db(query, n_results=k)
        return [
            {
                "content": r["content"],
                "metadata": r["metadata"],
                "added_at": r["metadata"].get("ingested_at", ""),
            }
            for r in raw_results
        ]

    def document_count(self) -> int:
        """컬렉션의 전체 청크 수를 반환합니다."""
        try:
            return get_collection().count()
        except Exception:
            return 0
