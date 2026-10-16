"""
데이터 수집 → DB 적재 통합 파이프라인

Data Flow:
    [ECOS API]       ──→ TimescaleDB (macro_metrics hypertable)
    [FRED API]       ──→ TimescaleDB (macro_metrics hypertable)
    [News RSS]       ──→ Vector DB   (macro_context collection)
    [은행 FX 리포트] ──→ Vector DB   (macro_context collection)
    [리포트 업로드]  ──→ Vector DB   (macro_context collection)

LangGraph data_collector 노드에서 run_ingestion_pipeline()을 호출하거나,
스케줄러(Airflow, APScheduler 등)에서 주기적으로 실행합니다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from macro_agent.db.timescale.repository import (
    UpsertStats,
    insert_macro_data,
    normalize_customs_to_records,
    normalize_ecos_to_records,
    normalize_fred_to_records,
)
from macro_agent.db.vector.repository import ingest_text_to_vector_db
from macro_agent.tools.ecos_tool import fetch_ecos_data
from macro_agent.tools.firecrawl_fetcher import (
    ArticleFetchResult,
    fetch_articles_bodies,
    is_enabled as firecrawl_enabled,
)
from macro_agent.tools.fred_tool import fetch_fred_data
from macro_agent.tools.fx_report_crawler import (
    CrawlStats,
    crawl_and_ingest_fx_reports,
    crawl_stats_summary,
)
from macro_agent.tools.customs_tool import (
    fetch_customs_20day_data,
    fetch_customs_monthly_data,
    is_enabled as customs_enabled,
)
from macro_agent.tools.news_tool import scraped_news_and_reports

logger = logging.getLogger(__name__)


# ── 파이프라인 결과 모델 ──────────────────────────────────────────────

@dataclass
class IngestionResult:
    """run_ingestion_pipeline() 실행 결과 요약"""
    run_at: datetime
    ecos_stats: UpsertStats | None = None
    fred_stats: UpsertStats | None = None
    customs_stats: UpsertStats | None = None          # 관세청 수출입 통계
    news_chunks_ingested: int = 0
    fx_report_stats: list[CrawlStats] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def total_ts_rows(self) -> int:
        ecos    = self.ecos_stats.changed    if self.ecos_stats    else 0
        fred    = self.fred_stats.changed    if self.fred_stats    else 0
        customs = self.customs_stats.changed if self.customs_stats else 0
        return ecos + fred + customs

    @property
    def fx_report_chunks(self) -> int:
        return sum(s.vector_chunks for s in self.fx_report_stats)

    @property
    def is_healthy(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> dict[str, Any]:
        return {
            "run_at": self.run_at.isoformat(),
            "timescale_rows_changed": self.total_ts_rows,
            "vector_chunks_ingested": self.news_chunks_ingested + self.fx_report_chunks,
            "ecos": {
                "submitted": self.ecos_stats.submitted if self.ecos_stats else 0,
                "changed":   self.ecos_stats.changed   if self.ecos_stats else 0,
            },
            "fred": {
                "submitted": self.fred_stats.submitted if self.fred_stats else 0,
                "changed":   self.fred_stats.changed   if self.fred_stats else 0,
            },
            "customs": {
                "submitted": self.customs_stats.submitted if self.customs_stats else 0,
                "changed":   self.customs_stats.changed   if self.customs_stats else 0,
                "enabled":   customs_enabled(),
            },
            "fx_reports": crawl_stats_summary(self.fx_report_stats),
            "errors": self.errors,
            "healthy": self.is_healthy,
        }


# ── 뉴스 → Vector DB 적재 헬퍼 ───────────────────────────────────────

def _parse_rss_date(raw: str) -> str:
    """RSS RFC 2822 날짜 → YYYY-MM-DD. 파싱 실패 시 빈 문자열 반환."""
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).strftime("%Y-%m-%d")
    except Exception:
        # 이미 YYYY-MM-DD 형식이면 그대로 반환
        if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
            return raw[:10]
        return ""


def _ingest_news_articles(articles: list[dict]) -> int:
    """
    뉴스 기사 목록을 Vector DB에 청킹·임베딩·저장합니다.

    FIRECRAWL_API_KEY가 설정된 경우:
        Firecrawl로 기사 전문(최대 8000자) + 발행일을 함께 수집합니다.
        - 본문: title + body (Firecrawl 마크다운)
        - 발행일: Firecrawl metadata.publishedTime 우선, 없으면 RSS pubDate 폴백
    미설정 시:
        RSS 요약(최대 600자)과 RSS pubDate로 폴백합니다.

    날짜 우선순위:
        1. Firecrawl metadata.publishedTime   (기사 HTML 메타태그, 가장 정확)
        2. RSS feed pubDate                   (RSS 발행 시각 기준)
        3. 빈 문자열 (둘 다 없으면)
    """
    total_chunks = 0

    # ── Firecrawl 활성화 시 일괄 본문 + 발행일 수집 ──────────────────
    fetch_map: dict[str, ArticleFetchResult] = {}
    if firecrawl_enabled():
        logger.info("Firecrawl 활성화 — %d건 기사 본문·발행일 수집 시작", len(articles))
        fetch_map = fetch_articles_bodies(articles, url_key="link")
    else:
        logger.debug("Firecrawl 비활성화 — RSS 요약·발행일로 Vector DB 저장")

    # ── 기사별 Vector DB 적재 ─────────────────────────────────────────
    for article in articles:
        title   = article.get("title", "").strip()
        summary = article.get("summary", "").strip()
        url     = article.get("link", "")

        # ── 본문: Firecrawl 우선, 없으면 RSS 요약 폴백 ───────────────
        fc_result   = fetch_map.get(url, {})
        body        = fc_result.get("body")       if fc_result else None
        fc_date     = fc_result.get("published_date") if fc_result else None

        if body:
            text        = f"{title}\n\n{body}".strip()
            text_source = "body"
        else:
            text        = f"{title}\n\n{summary}".strip()
            text_source = "summary"

        if len(text) < 30:
            continue

        # ── 발행일: Firecrawl 날짜 우선, RSS pubDate 폴백 ────────────
        rss_date       = _parse_rss_date(article.get("published", ""))
        published_date = fc_date or rss_date

        date_source = (
            "firecrawl" if fc_date
            else ("rss" if rss_date else "unknown")
        )
        logger.debug(
            "날짜 확정 — %s (%s) url=%r",
            published_date or "없음", date_source, url[:60],
        )

        try:
            chunks = ingest_text_to_vector_db(
                text=text,
                metadata={
                    "source_name":    article.get("source", "Google News"),
                    "published_date": published_date,   # "YYYY-MM-DD" | ""
                    "date_source":    date_source,       # "firecrawl" | "rss" | "unknown"
                    "url":            url,
                    "tags":           ["뉴스", article.get("query", "")],
                    "text_source":    text_source,       # "body" | "summary"
                },
                source_type="news",
            )
            total_chunks += chunks
        except Exception as exc:
            logger.warning("뉴스 기사 Vector DB 적재 실패 (title=%r): %s", title[:40], exc)

    return total_chunks


# ── 메인 파이프라인 ───────────────────────────────────────────────────

def run_ingestion_pipeline(
    months_back: int = 6,
    fetch_news: bool = True,
    fetch_fx_reports: bool = True,
    fetch_customs: bool = True,
    custom_news_queries: list[str] | None = None,
    fx_months_back: int | None = None,
) -> IngestionResult:
    """
    전체 데이터 수집 → DB 적재 파이프라인을 실행합니다.

    실행 순서:
        1. ECOS API       → TimescaleDB (환율·반도체 수출·출하·재고지수)
        2. FRED API       → TimescaleDB (SOX·제조업IP·연방기금금리·GSCPI)
        3. 뉴스 RSS       → Vector DB  (청킹·임베딩 후 RAG용 저장)
        4. KCIF FX 리포트 → Vector DB  (PDF 크롤링·추출 후 RAG용 저장)
        5. 관세청 API     → TimescaleDB (20일 동향 + 월별 수출입 통계)
           ※ CUSTOMS_API_KEY 미설정 시 자동 스킵

    각 단계는 독립적으로 예외 처리되어, 한 단계 실패가 다음 단계를 막지 않습니다.

    Args:
        months_back:          API 조회 기간 (월 단위)
        fetch_news:           뉴스 RSS 수집 및 Vector DB 적재 여부
        fetch_fx_reports:     KCIF FX 리포트 PDF 크롤링 여부
        fetch_customs:        관세청 수출입 통계 수집 여부 (CUSTOMS_API_KEY 필요)
        custom_news_queries:  뉴스 검색 쿼리 (None이면 기본 5개 쿼리)
        fx_months_back:       FX 리포트 수집 기간 (None이면 months_back 사용)

    Returns:
        IngestionResult — 각 단계 결과와 오류 목록 포함
    """
    result = IngestionResult(run_at=datetime.now(timezone.utc))
    logger.info("=== 데이터 수집 파이프라인 시작 (months_back=%d) ===", months_back)

    # ── Step 1. ECOS → TimescaleDB ────────────────────────────────────
    try:
        logger.info("[Step 1/4] ECOS API 호출 중...")
        ecos_data: dict = fetch_ecos_data.invoke({"months_back": months_back})

        records = normalize_ecos_to_records(ecos_data)
        result.ecos_stats = insert_macro_data(records)

        logger.info(
            "[Step 1/4] ECOS 완료 — 제출: %d, 변경: %d건",
            result.ecos_stats.submitted,
            result.ecos_stats.changed,
        )
    except EnvironmentError as exc:
        msg = f"[Step 1/4] ECOS API 키 미설정: {exc}"
        logger.error(msg)
        result.errors.append(msg)
    except Exception as exc:
        msg = f"[Step 1/4] ECOS 적재 실패: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)

    # ── Step 2. FRED → TimescaleDB ────────────────────────────────────
    try:
        logger.info("[Step 2/4] FRED API 호출 중...")
        fred_data: dict = fetch_fred_data.invoke({"months_back": months_back})

        records = normalize_fred_to_records(fred_data)
        result.fred_stats = insert_macro_data(records)

        logger.info(
            "[Step 2/4] FRED 완료 — 제출: %d, 변경: %d건",
            result.fred_stats.submitted,
            result.fred_stats.changed,
        )
    except EnvironmentError as exc:
        msg = f"[Step 2/4] FRED API 키 미설정: {exc}"
        logger.error(msg)
        result.errors.append(msg)
    except Exception as exc:
        msg = f"[Step 2/4] FRED 적재 실패: {exc}"
        logger.error(msg, exc_info=True)
        result.errors.append(msg)

    # ── Step 3. 뉴스 RSS → Vector DB ─────────────────────────────────
    if fetch_news:
        try:
            logger.info("[Step 3/4] 뉴스 RSS 수집 중...")
            news_data: dict = scraped_news_and_reports.invoke({
                "custom_queries": custom_news_queries,
                "include_rag": False,   # 수집 단계에서는 RAG 검색 비활성화
                "max_news_per_query": 5,
            })

            articles = news_data.get("news_articles", [])
            result.news_chunks_ingested = _ingest_news_articles(articles)

            logger.info(
                "[Step 3/4] 뉴스 완료 — 기사: %d건, Vector DB 청크: %d개",
                len(articles),
                result.news_chunks_ingested,
            )

            # 뉴스 수집 중 발생한 소프트 오류도 기록
            for err in news_data.get("errors", []):
                result.errors.append(f"[Step 3/3 소프트 오류] {err.get('message', '')}")

        except Exception as exc:
            msg = f"[Step 3/4] 뉴스 Vector DB 적재 실패: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)
    else:
        logger.info("[Step 3/4] 뉴스 수집 스킵 (fetch_news=False)")

    # ── Step 4. 은행 FX 리포트 PDF → Vector DB ───────────────────────
    if fetch_fx_reports:
        try:
            report_months = fx_months_back if fx_months_back is not None else months_back
            logger.info("[Step 4/4] 은행 FX 리포트 크롤링 중 (months_back=%d)...", report_months)
            crawl_results = crawl_and_ingest_fx_reports(
                months_back=report_months,
            )
            result.fx_report_stats = crawl_results

            total_downloaded = sum(s.downloaded for s in crawl_results)
            total_chunks = sum(s.vector_chunks for s in crawl_results)
            logger.info(
                "[Step 4/4] FX 리포트 완료 — 다운로드: %d건, Vector DB 청크: %d개",
                total_downloaded,
                total_chunks,
            )

            # 크롤러별 에러를 파이프라인 에러에 병합
            for cs in crawl_results:
                for err in cs.errors:
                    result.errors.append(f"[Step 4/4 {cs.source}] {err}")

        except Exception as exc:
            msg = f"[Step 4/4] FX 리포트 크롤링 실패: {exc}"
            logger.error(msg, exc_info=True)
            result.errors.append(msg)
    else:
        logger.info("[Step 4/4] FX 리포트 크롤링 스킵 (fetch_fx_reports=False)")

    # ── Step 5. 관세청 수출입 통계 → TimescaleDB ─────────────────────
    if fetch_customs:
        if not customs_enabled():
            logger.info("[Step 5/5] 관세청 API 키 미설정 — 수출입 통계 수집 스킵")
        else:
            all_customs_records = []

            # 5-A. 당월 20일 동향 (잠정치)
            try:
                logger.info("[Step 5/5-A] 관세청 20일 수출입 동향 조회 중...")
                data_20d = fetch_customs_20day_data()
                if data_20d.get("error"):
                    raise RuntimeError(data_20d["error"])
                recs_20d = normalize_customs_to_records(data_20d)
                all_customs_records.extend(recs_20d)
                logger.info(
                    "[Step 5/5-A] 20일 동향 완료 — 기간: %s, 레코드: %d건",
                    data_20d.get("period", "N/A"),
                    len(recs_20d),
                )
            except Exception as exc:
                msg = f"[Step 5/5-A] 관세청 20일 동향 수집 실패: {exc}"
                logger.error(msg, exc_info=True)
                result.errors.append(msg)

            # 5-B. 월별 확정 통계 (months_back 개월)
            try:
                logger.info(
                    "[Step 5/5-B] 관세청 월별 수출입 통계 조회 중 (months_back=%d)...",
                    months_back,
                )
                data_monthly = fetch_customs_monthly_data(months_back=months_back)
                for err in data_monthly.get("errors", []):
                    result.errors.append(f"[Step 5/5-B 소프트 오류] {err}")
                recs_monthly = normalize_customs_to_records(data_monthly)
                all_customs_records.extend(recs_monthly)
                logger.info(
                    "[Step 5/5-B] 월별 통계 완료 — %d개월, 레코드: %d건",
                    months_back,
                    len(recs_monthly),
                )
            except Exception as exc:
                msg = f"[Step 5/5-B] 관세청 월별 통계 수집 실패: {exc}"
                logger.error(msg, exc_info=True)
                result.errors.append(msg)

            # 5-C. 일괄 Upsert
            if all_customs_records:
                try:
                    result.customs_stats = insert_macro_data(all_customs_records)
                    logger.info(
                        "[Step 5/5] 관세청 통계 DB 적재 완료 — 제출: %d, 변경: %d건",
                        result.customs_stats.submitted,
                        result.customs_stats.changed,
                    )
                except Exception as exc:
                    msg = f"[Step 5/5] 관세청 통계 DB 적재 실패: {exc}"
                    logger.error(msg, exc_info=True)
                    result.errors.append(msg)
    else:
        logger.info("[Step 5/5] 관세청 수출입 통계 수집 스킵 (fetch_customs=False)")

    # ── 완료 로그 ──────────────────────────────────────────────────────
    status = "정상" if result.is_healthy else f"오류 {len(result.errors)}건"
    logger.info(
        "=== 파이프라인 완료 [%s] — TS변경: %d행, Vector청크: %d개 (뉴스:%d + FX리포트:%d) ===",
        status,
        result.total_ts_rows,
        result.news_chunks_ingested + result.fx_report_chunks,
        result.news_chunks_ingested,
        result.fx_report_chunks,
    )
    return result


# ── 개별 리포트 수동 적재 ─────────────────────────────────────────────

def ingest_report(
    content: str,
    source_name: str,
    published_date: str,
    tags: list[str] | None = None,
    url: str = "",
) -> int:
    """
    사용자가 수동으로 업로드한 환율 리포트·분석 문서를 Vector DB에 적재합니다.

    Args:
        content:        리포트 전문 텍스트
        source_name:    출처명 (예: "KDB산업은행 반도체 산업 보고서")
        published_date: 발간일 (YYYY-MM-DD)
        tags:           태그 목록 (예: ["반도체", "환율", "지정학"])
        url:            원본 URL (선택)

    Returns:
        저장된 청크 수
    """
    chunks = ingest_text_to_vector_db(
        text=content,
        metadata={
            "source_name": source_name,
            "published_date": published_date,
            "url": url,
            "tags": tags or [],
        },
        source_type="report",
    )
    logger.info("리포트 수동 적재 완료 — source=%r, 청크=%d개", source_name, chunks)
    return chunks
