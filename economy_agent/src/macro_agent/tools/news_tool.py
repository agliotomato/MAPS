"""
News & RAG Tool
구글 뉴스 RSS 피드와 사용자 업로드 리포트(RAG 인터페이스)를 통해
반도체 거시경제·지정학 관련 컨텍스트 수집

설계 원칙:
    - ReportStoreProtocol: 벡터 DB 구현체(Chroma, Pinecone 등)로 교체 가능한 Protocol
    - InMemoryReportStore: 개발/테스트용 인메모리 폴백 구현체
    - set_report_store(): DI(의존성 주입)로 런타임에 스토어 교체
"""

from __future__ import annotations

import logging
import urllib.parse
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

import feedparser
import requests
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"
_REQUEST_HEADERS = {"User-Agent": "MacroAnalysisAgent/1.0 (+https://github.com/your-org/macro-agent)"}

# 반도체 도메인 기본 검색 쿼리 (한/영 혼합으로 커버리지 확대)
_DEFAULT_QUERIES: list[str] = [
    "semiconductor geopolitical risk supply chain",
    "US China chip export control BIS EAR",
    "TSMC Samsung Intel foundry capacity",
    "반도체 수출 규제 미중 갈등",
    "반도체 경기 사이클 재고 조정",
]


# ── RAG 인터페이스 (Protocol) ──────────────────────────────────────────

@runtime_checkable
class ReportStoreProtocol(Protocol):
    """
    업로드된 리포트를 저장·검색하는 벡터 스토어 인터페이스.

    프로덕션 교체 예시:
        from langchain_community.vectorstores import Chroma
        from langchain_openai import OpenAIEmbeddings

        chroma_store = ChromaReportStore(
            Chroma(embedding_function=OpenAIEmbeddings(), persist_directory="./chroma_db")
        )
        set_report_store(chroma_store)
    """

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]: ...
    def add_documents(self, documents: list[dict[str, Any]]) -> None: ...
    def document_count(self) -> int: ...


class InMemoryReportStore:
    """
    개발·테스트용 인메모리 스토어. 키워드 교집합 기반 유사도 검색.
    프로덕션에서는 LangChain VectorStore 래퍼 구현체로 교체하세요.
    """

    def __init__(self) -> None:
        self._docs: list[dict[str, Any]] = []

    def add_documents(self, documents: list[dict[str, Any]]) -> None:
        for doc in documents:
            content = doc.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 10:
                raise ValueError(f"문서 content가 너무 짧거나 유효하지 않습니다: {repr(content)[:50]}")
            self._docs.append({
                "content": content,
                "metadata": doc.get("metadata", {}),
                "added_at": datetime.utcnow().isoformat() + "Z",
            })

    def similarity_search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        query_terms = set(query.lower().split())
        scored: list[tuple[int, dict]] = []
        for doc in self._docs:
            content_words = set(doc["content"].lower().split())
            score = len(query_terms & content_words)
            if score > 0:
                scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[:k]]

    def document_count(self) -> int:
        return len(self._docs)


# 모듈 수준 싱글톤 스토어 (애플리케이션 수명 동안 공유)
_report_store: ReportStoreProtocol = InMemoryReportStore()


def set_report_store(store: ReportStoreProtocol) -> None:
    """
    프로덕션 벡터 스토어로 교체하는 의존성 주입 함수.
    애플리케이션 초기화 시점(main.py 등)에서 1회 호출하세요.

    Args:
        store: ReportStoreProtocol을 구현한 벡터 스토어 인스턴스
    """
    if not isinstance(store, ReportStoreProtocol):
        raise TypeError(
            f"{type(store).__name__}은 ReportStoreProtocol을 구현하지 않습니다. "
            "similarity_search, add_documents, document_count 메서드를 확인하세요."
        )
    global _report_store
    _report_store = store
    logger.info("ReportStore가 %s로 교체되었습니다.", type(store).__name__)


# ── 뉴스 수집 내부 함수 ───────────────────────────────────────────────

@dataclass(frozen=True)
class NewsItem:
    title: str
    link: str
    published: str
    summary: str
    source: str
    query: str


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    wait=wait_exponential(multiplier=1, min=1, max=6),
    stop=stop_after_attempt(2),
    reraise=True,
)
def _fetch_rss_feed(query: str, lang: str, country: str, max_items: int) -> list[NewsItem]:
    """구글 뉴스 RSS 피드 파싱 및 NewsItem 변환"""
    params = {
        "q": query,
        "hl": lang,
        "gl": country,
        "ceid": f"{country}:{lang}",
    }
    url = f"{_GOOGLE_NEWS_RSS_BASE}?{urllib.parse.urlencode(params)}"

    # feedparser 직접 사용 시 타임아웃 제어 불가 → requests로 raw 바이트 먼저 수신
    resp = requests.get(url, timeout=10, headers=_REQUEST_HEADERS)
    resp.raise_for_status()

    feed = feedparser.parse(resp.content)

    # bozo=True 이지만 entries가 있으면 부분 파싱 성공으로 처리
    if feed.bozo and not feed.entries:
        raise ValueError(f"RSS 파싱 실패: {feed.bozo_exception}")

    items: list[NewsItem] = []
    for entry in feed.entries[:max_items]:
        source_info = entry.get("source", {})
        source_name = (
            source_info.get("title", "Google News")
            if isinstance(source_info, dict)
            else "Google News"
        )
        items.append(NewsItem(
            title=entry.get("title", "").strip(),
            link=entry.get("link", ""),
            published=entry.get("published", ""),
            summary=entry.get("summary", "")[:600],  # LLM 컨텍스트 절약
            source=source_name,
            query=query,
        ))
    return items


# ── Public Tools ───────────────────────────────────────────────────────

@tool
def scraped_news_and_reports(
    custom_queries: list[str] | None = None,
    include_rag: bool = True,
    max_news_per_query: int = 5,
    news_lang: str = "ko",
    news_country: str = "KR",
) -> dict[str, Any]:
    """
    구글 뉴스 RSS와 업로드된 리포트(RAG)를 통합하여 반도체 거시경제·지정학 관련
    최신 컨텍스트를 수집합니다. LLM 분석 노드에 배경 정보를 공급하는 데 사용됩니다.

    Args:
        custom_queries: 사용자 정의 검색 쿼리 목록 (None이면 기본 5개 쿼리 사용)
        include_rag: 업로드된 리포트(RAG 스토어) 포함 여부
        max_news_per_query: 쿼리당 최대 뉴스 기사 수 (기본값: 5)
        news_lang: 뉴스 언어 코드 (기본값: "ko")
        news_country: 뉴스 국가 코드 (기본값: "KR")

    Returns:
        Dict with keys: news_articles, news_article_count,
        rag_documents, rag_document_count, queries_used, errors, fetched_at
    """
    queries = custom_queries if custom_queries else _DEFAULT_QUERIES
    result: dict[str, Any] = {
        "fetched_at": datetime.utcnow().isoformat() + "Z",
        "queries_used": queries,
        "news_articles": [],
        "rag_documents": [],
        "errors": [],
    }

    # ── 1. 구글 뉴스 RSS 수집 ─────────────────────────────────────────
    seen_links: set[str] = set()  # 중복 기사 제거

    for query in queries:
        try:
            items = _fetch_rss_feed(
                query=query,
                lang=news_lang,
                country=news_country,
                max_items=max_news_per_query,
            )
            new_count = 0
            for item in items:
                if item.link not in seen_links:
                    seen_links.add(item.link)
                    new_count += 1
                    result["news_articles"].append({
                        "title": item.title,
                        "link": item.link,
                        "published": item.published,
                        "summary": item.summary,
                        "source": item.source,
                        "query": item.query,
                    })
            logger.info("뉴스 수집 완료 (query=%r, 신규=%d건)", query, new_count)

        except requests.exceptions.Timeout:
            msg = f"뉴스 RSS 타임아웃 (query={query!r})"
            logger.warning(msg)
            result["errors"].append({"type": "timeout", "query": query, "message": msg})
        except requests.exceptions.HTTPError as exc:
            msg = f"뉴스 RSS HTTP {exc.response.status_code} (query={query!r})"
            logger.error(msg)
            result["errors"].append({"type": "http_error", "query": query, "message": msg})
        except ValueError as exc:
            msg = f"뉴스 RSS 파싱 실패 (query={query!r}): {exc}"
            logger.error(msg)
            result["errors"].append({"type": "parse_error", "query": query, "message": msg})
        except Exception as exc:
            msg = f"뉴스 수집 알 수 없는 오류 (query={query!r}): {exc}"
            logger.error(msg, exc_info=True)
            result["errors"].append({"type": "unknown", "query": query, "message": msg})

    result["news_article_count"] = len(result["news_articles"])

    # ── 2. RAG 리포트 검색 ────────────────────────────────────────────
    if include_rag:
        rag_query = " ".join(queries[:3])  # 상위 3개 쿼리 조합으로 검색
        try:
            total_docs = _report_store.document_count()
            if total_docs == 0:
                logger.info("RAG 스토어가 비어있습니다. upload_report_to_rag로 문서를 추가하세요.")
                result["rag_documents"] = []
                result["rag_document_count"] = 0
                result["rag_store_status"] = "empty"
            else:
                docs = _report_store.similarity_search(rag_query, k=5)
                result["rag_documents"] = [
                    {
                        "content": doc["content"][:1500],  # LLM 컨텍스트 절약
                        "metadata": doc.get("metadata", {}),
                        "added_at": doc.get("added_at", ""),
                    }
                    for doc in docs
                ]
                result["rag_document_count"] = len(docs)
                result["rag_store_status"] = f"active ({total_docs} docs in store)"
                logger.info("RAG 검색 완료: %d건 반환 (전체 %d건)", len(docs), total_docs)
        except Exception as exc:
            msg = f"RAG 검색 실패: {exc}"
            logger.error(msg, exc_info=True)
            result["rag_documents"] = []
            result["rag_document_count"] = 0
            result["rag_store_status"] = "error"
            result["errors"].append({"type": "rag_error", "message": msg})
    else:
        result["rag_document_count"] = 0
        result["rag_store_status"] = "skipped"

    return result


@tool
def upload_report_to_rag(
    content: str,
    source: str = "user_upload",
    report_date: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """
    환율 리포트, 분석 문서, 뉴스 요약 등을 RAG 스토어에 저장합니다.
    저장된 문서는 이후 scraped_news_and_reports 호출 시 컨텍스트로 활용됩니다.

    Args:
        content: 저장할 문서 텍스트 (최소 50자)
        source: 문서 출처 (예: "KDB리포트", "삼성증권", "user_upload")
        report_date: 리포트 작성일 (YYYY-MM-DD, 생략 시 현재 시각)
        tags: 태그 목록 (예: ["반도체", "환율", "Fed"])

    Returns:
        Dict with keys: success, preview, char_count, metadata
    """
    content = content.strip()
    if len(content) < 50:
        return {
            "success": False,
            "error": f"문서가 너무 짧습니다 (현재 {len(content)}자, 최소 50자 이상 필요).",
        }

    metadata: dict[str, Any] = {
        "source": source,
        "report_date": report_date or datetime.utcnow().strftime("%Y-%m-%d"),
        "uploaded_at": datetime.utcnow().isoformat() + "Z",
        "tags": tags or [],
        "char_count": len(content),
    }

    try:
        _report_store.add_documents([{"content": content, "metadata": metadata}])
        logger.info(
            "RAG 스토어에 문서 저장 완료 (source=%r, %d자)", source, len(content)
        )
        return {
            "success": True,
            "preview": content[:300] + ("…" if len(content) > 300 else ""),
            "char_count": len(content),
            "metadata": metadata,
            "store_total": _report_store.document_count(),
        }
    except ValueError as exc:
        logger.error("RAG 문서 유효성 검사 실패: %s", exc)
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.error("RAG 문서 저장 실패: %s", exc, exc_info=True)
        return {"success": False, "error": f"저장 중 오류 발생: {exc}"}
