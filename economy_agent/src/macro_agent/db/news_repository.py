"""
뉴스 기사 PostgreSQL CRUD — news_articles 테이블

핵심 함수:
    upsert_article()           — 기사 저장/업데이트, article_id 반환
    get_unanalyzed_articles()  — FinBERT 미분석 기사 조회
    update_analysis()          — 감성/NER 결과 저장
    get_not_vector_ingested()  — ChromaDB 미적재 기사 조회
    mark_vector_ingested()     — Vector DB 적재 완료 표시
    url_exists()               — URL 중복 체크
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from macro_agent.db.timescale.client import get_connection

logger = logging.getLogger(__name__)


@dataclass
class ArticleRow:
    url: str
    title: str
    body: str | None = None
    summary: str | None = None
    publisher: str | None = None
    author: str | None = None
    published_date: date | None = None
    lang: str = "en"
    query_tag: str | None = None


def upsert_article(article: ArticleRow) -> int | None:
    """
    기사를 삽입하거나 제목/본문/요약을 업데이트합니다.

    Returns:
        삽입/업데이트된 행의 id (None이면 오류)
    """
    sql = """
        INSERT INTO news_articles
            (url, title, body, summary, publisher, author, published_date, lang, query_tag)
        VALUES
            (%(url)s, %(title)s, %(body)s, %(summary)s,
             %(publisher)s, %(author)s, %(published_date)s, %(lang)s, %(query_tag)s)
        ON CONFLICT (url) DO UPDATE SET
            title          = EXCLUDED.title,
            body           = COALESCE(EXCLUDED.body, news_articles.body),
            summary        = COALESCE(EXCLUDED.summary, news_articles.summary),
            publisher      = COALESCE(EXCLUDED.publisher, news_articles.publisher),
            author         = COALESCE(EXCLUDED.author, news_articles.author),
            published_date = COALESCE(EXCLUDED.published_date, news_articles.published_date),
            lang           = EXCLUDED.lang,
            query_tag      = COALESCE(EXCLUDED.query_tag, news_articles.query_tag)
        RETURNING id
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "url":            article.url,
                    "title":          article.title,
                    "body":           article.body,
                    "summary":        article.summary,
                    "publisher":      article.publisher,
                    "author":         article.author,
                    "published_date": article.published_date,
                    "lang":           article.lang,
                    "query_tag":      article.query_tag,
                })
                row = cur.fetchone()
                return row["id"] if row else None
    except Exception as exc:
        logger.error("upsert_article 실패 (url=%r): %s", article.url[:60], exc)
        return None


def url_exists(url: str) -> bool:
    """URL이 이미 DB에 있는지 확인합니다."""
    sql = "SELECT 1 FROM news_articles WHERE url = %s LIMIT 1"
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (url,))
                return cur.fetchone() is not None
    except Exception:
        return False


def get_unanalyzed_articles(limit: int = 50) -> list[dict[str, Any]]:
    """
    FinBERT 감성 분석이 완료되지 않은 기사를 조회합니다.
    body 또는 summary가 있는 기사만 반환합니다.
    """
    sql = """
        SELECT id, title, body, summary, lang
        FROM news_articles
        WHERE analyzed_at IS NULL
          AND (body IS NOT NULL OR summary IS NOT NULL)
        ORDER BY crawled_at DESC
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                return [dict(row) for row in cur.fetchall()]
    except Exception as exc:
        logger.error("get_unanalyzed_articles 실패: %s", exc)
        return []


def update_analysis(
    article_id: int,
    sentiment_pos: float,
    sentiment_neg: float,
    sentiment_neu: float,
    entities: dict[str, list[str]],
    keywords: list[str],
) -> None:
    """FinBERT 감성 점수와 NER 엔티티를 news_articles에 저장합니다."""
    sql = """
        UPDATE news_articles SET
            sentiment_pos = %(pos)s,
            sentiment_neg = %(neg)s,
            sentiment_neu = %(neu)s,
            entities      = %(entities)s,
            keywords      = %(keywords)s,
            analyzed_at   = NOW()
        WHERE id = %(id)s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {
                    "id":       article_id,
                    "pos":      round(sentiment_pos, 4),
                    "neg":      round(sentiment_neg, 4),
                    "neu":      round(sentiment_neu, 4),
                    "entities": json.dumps(entities, ensure_ascii=False),
                    "keywords": keywords,
                })
    except Exception as exc:
        logger.error("update_analysis 실패 (id=%d): %s", article_id, exc)


def get_not_vector_ingested(limit: int = 100) -> list[dict[str, Any]]:
    """
    Vector DB(ChromaDB)에 아직 적재되지 않은 기사를 조회합니다.
    body가 있는 기사를 우선 반환합니다.
    """
    sql = """
        SELECT id, url, title, body, summary, publisher,
               published_date, lang, query_tag,
               sentiment_pos, sentiment_neg, sentiment_neu,
               entities, keywords
        FROM news_articles
        WHERE vector_ingested = FALSE
          AND (body IS NOT NULL OR summary IS NOT NULL)
        ORDER BY (body IS NOT NULL) DESC, published_date DESC NULLS LAST
        LIMIT %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (limit,))
                rows = []
                for row in cur.fetchall():
                    r = dict(row)
                    if r.get("entities") and isinstance(r["entities"], str):
                        try:
                            r["entities"] = json.loads(r["entities"])
                        except Exception:
                            r["entities"] = {}
                    rows.append(r)
                return rows
    except Exception as exc:
        logger.error("get_not_vector_ingested 실패: %s", exc)
        return []


def mark_vector_ingested(article_id: int, chunk_count: int) -> None:
    """Vector DB 적재 완료 표시."""
    sql = """
        UPDATE news_articles
        SET vector_ingested = TRUE, vector_chunks = %s
        WHERE id = %s
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (chunk_count, article_id))
    except Exception as exc:
        logger.error("mark_vector_ingested 실패 (id=%d): %s", article_id, exc)


def get_article_stats() -> dict[str, int]:
    """news_articles 테이블의 현황을 반환합니다."""
    sql = """
        SELECT
            COUNT(*)                                        AS total,
            COUNT(*) FILTER (WHERE body IS NOT NULL)        AS has_body,
            COUNT(*) FILTER (WHERE analyzed_at IS NOT NULL) AS analyzed,
            COUNT(*) FILTER (WHERE vector_ingested = TRUE)  AS vector_ingested
        FROM news_articles
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
                return {
                    "total":           row["total"] if row else 0,
                    "has_body":        row["has_body"] if row else 0,
                    "analyzed":        row["analyzed"] if row else 0,
                    "vector_ingested": row["vector_ingested"] if row else 0,
                }
    except Exception as exc:
        logger.error("get_article_stats 실패: %s", exc)
        return {"total": 0, "has_body": 0, "analyzed": 0, "vector_ingested": 0}
