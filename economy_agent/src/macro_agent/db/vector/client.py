"""
ChromaDB 클라이언트 싱글톤 관리

필요 환경변수:
    CHROMA_PERSIST_DIR : 영구 저장 경로 (미설정 시 인메모리 EphemeralClient 사용)
    OPENAI_API_KEY     : OpenAI 임베딩 API 키 (text-embedding-3-small 사용)
    EMBEDDING_MODEL    : 임베딩 모델명 (기본값: text-embedding-3-small)

컬렉션 구조:
    macro_context — 뉴스, 리포트를 단일 컬렉션으로 관리 (source_type 메타데이터로 구분)
"""

from __future__ import annotations

import logging
import os
from threading import Lock

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "macro_context"

_client: chromadb.ClientAPI | None = None
_client_lock = Lock()


# ── 클라이언트 싱글톤 ──────────────────────────────────────────────────

def get_chroma_client() -> chromadb.ClientAPI:
    """
    ChromaDB 클라이언트 싱글톤을 반환합니다.

    - CHROMA_PERSIST_DIR 설정 시: 디스크 영구 저장 (PersistentClient)
    - 미설정 시: 인메모리 휘발성 저장 (EphemeralClient, 테스트용)
    """
    global _client
    if _client is not None:
        return _client

    with _client_lock:
        if _client is not None:
            return _client

        persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "")
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            _client = chromadb.PersistentClient(path=persist_dir)
            logger.info("ChromaDB PersistentClient 초기화 (path=%s)", persist_dir)
        else:
            _client = chromadb.EphemeralClient()
            logger.warning(
                "ChromaDB EphemeralClient 사용 중 — 데이터가 메모리에만 저장됩니다. "
                "프로덕션에서는 CHROMA_PERSIST_DIR을 설정하세요."
            )

    return _client


def get_embedding_function() -> OpenAIEmbeddingFunction:
    """
    OpenAI text-embedding-3-small 임베딩 함수를 반환합니다.
    OPENAI_API_KEY 환경변수가 필요합니다.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
            "임베딩 생성에 OpenAI API 키가 필요합니다."
        )

    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddingFunction(api_key=api_key, model_name=model)


def get_collection() -> chromadb.Collection:
    """
    macro_context 컬렉션을 반환합니다.
    존재하지 않으면 cosine 유사도 기준으로 자동 생성합니다.
    """
    client = get_chroma_client()
    embedding_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},  # cosine 유사도 (텍스트 검색 표준)
    )
    return collection


def reset_collection() -> None:
    """컬렉션 전체 삭제 후 재생성 (테스트·초기화 용도)"""
    client = get_chroma_client()
    try:
        client.delete_collection(_COLLECTION_NAME)
        logger.warning("ChromaDB 컬렉션 '%s' 삭제됨", _COLLECTION_NAME)
    except Exception:
        pass
    get_collection()
    logger.info("ChromaDB 컬렉션 '%s' 재생성 완료", _COLLECTION_NAME)


def healthcheck() -> dict[str, str]:
    """ChromaDB 연결 상태를 반환합니다."""
    try:
        collection = get_collection()
        count = collection.count()
        return {
            "status": "ok",
            "collection": _COLLECTION_NAME,
            "document_count": str(count),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
