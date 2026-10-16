"""
Firecrawl 기사 본문 + 발행일 수집기

FIRECRAWL_API_KEY 환경변수가 설정된 경우 Firecrawl API로 기사 전문과
발행일을 함께 수집합니다.
미설정 시 fetch_article_body()가 None을 반환 → pipeline이 RSS summary(600자)로 자동 폴백.

발급: https://www.firecrawl.dev/
설치: pip install firecrawl-py

설계 원칙:
    - API 키 미설정 → 경고 없이 None 반환 (선택적 기능)
    - 패키지 미설치 → ImportError 대신 경고 로그 후 None 반환
    - 네트워크 오류·스크래핑 실패 → WARNING 로그 후 None 반환
    - 모듈 수준 싱글톤 클라이언트로 연결 재사용
    - 요청 간 최소 간격(RATE_LIMIT_DELAY)으로 서버 부하 방지

발행일 추출 우선순위:
    1. Firecrawl metadata.publishedTime  (기사 원문 HTML 메타 태그)
    2. Firecrawl metadata.ogPublishedTime
    3. Firecrawl metadata.article:published_time
    4. 없으면 None → pipeline에서 RSS pubDate로 폴백

반환 구조 (fetch_articles_bodies):
    {url: {"body": str | None, "published_date": str | None}}
    published_date 형식: "YYYY-MM-DD" (시각 정보는 날짜만 남김)
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# ── 설정 상수 ─────────────────────────────────────────────────────────
_RATE_LIMIT_DELAY: float = 0.5    # 연속 요청 최소 간격 (초)
_MIN_BODY_LENGTH: int = 200       # 본문으로 인정할 최소 글자 수
_MAX_BODY_LENGTH: int = 8000      # Vector DB 청킹 전 최대 길이 (토큰 절약)

# Firecrawl metadata에서 발행일이 담길 수 있는 키 목록 (우선순위 순)
_DATE_META_KEYS: tuple[str, ...] = (
    "publishedTime",           # Firecrawl 표준 키 (가장 흔함)
    "ogPublishedTime",         # Open Graph
    "article:published_time",  # Facebook OG
    "datePublished",           # JSON-LD schema.org
    "date",                    # 일부 미디어사이트
)

# ── 반환 타입 ─────────────────────────────────────────────────────────

class ArticleFetchResult(TypedDict):
    """fetch_article_result() 반환 타입."""
    body:           str | None   # 마크다운 본문 (최대 8000자), 실패 시 None
    published_date: str | None   # "YYYY-MM-DD", 추출 실패 시 None


# ── 모듈 수준 싱글톤 ──────────────────────────────────────────────────
_client: Any = None               # FirecrawlApp 인스턴스 또는 None
_client_initialized: bool = False  # 초기화 시도 여부 (재시도 방지)
_last_request_at: float = 0.0     # 마지막 요청 시각 (monotonic)


def _init_client() -> Any:
    """
    Firecrawl 클라이언트를 초기화합니다. 최초 호출 시 1회만 실행됩니다.

    Returns:
        FirecrawlApp 인스턴스, 또는 API 키 미설정·패키지 미설치 시 None
    """
    global _client, _client_initialized

    if _client_initialized:
        return _client

    _client_initialized = True

    api_key = os.environ.get("FIRECRAWL_API_KEY", "").strip()
    if not api_key:
        logger.debug("FIRECRAWL_API_KEY 미설정 — 뉴스 본문 크롤링 비활성화 (RSS 요약 사용)")
        return None

    try:
        from firecrawl import FirecrawlApp  # type: ignore[import]
        _client = FirecrawlApp(api_key=api_key)
        logger.info("Firecrawl 클라이언트 초기화 완료 (key=...%s)", api_key[-4:])
    except ImportError:
        logger.warning(
            "firecrawl-py 패키지가 설치되지 않았습니다. "
            "pip install firecrawl-py 후 재시작하세요. (RSS 요약으로 폴백)"
        )
        _client = None
    except Exception as exc:
        logger.warning("Firecrawl 클라이언트 초기화 실패: %s (RSS 요약으로 폴백)", exc)
        _client = None

    return _client


def is_enabled() -> bool:
    """Firecrawl이 활성화되어 있는지 (API 키 설정 + 패키지 설치) 확인합니다."""
    return _init_client() is not None


# ── 날짜 파싱 헬퍼 ────────────────────────────────────────────────────

def _extract_date_from_metadata(metadata: Any) -> str | None:
    """
    Firecrawl 응답의 metadata 객체·dict에서 발행일을 추출합니다.

    Args:
        metadata: dict 또는 객체 (firecrawl-py 버전마다 다름)

    Returns:
        "YYYY-MM-DD" 문자열, 또는 추출 실패 시 None
    """
    if metadata is None:
        return None

    # dict 또는 객체 두 가지 형태 모두 처리
    def _get(key: str) -> str | None:
        if isinstance(metadata, dict):
            return metadata.get(key)
        return getattr(metadata, key, None)

    for key in _DATE_META_KEYS:
        raw = _get(key)
        if not raw or not isinstance(raw, str):
            continue
        parsed = _parse_date_str(raw.strip())
        if parsed:
            return parsed

    return None


def _parse_date_str(raw: str) -> str | None:
    """
    다양한 날짜 포맷을 "YYYY-MM-DD"로 정규화합니다.

    처리 포맷:
        ISO 8601    : "2026-06-03T10:30:00Z", "2026-06-03T10:30:00+09:00"
        날짜만      : "2026-06-03"
        슬래시 구분 : "2026/06/03"
        점 구분     : "2026.06.03"
    """
    if not raw:
        return None

    # ISO 8601 또는 YYYY-MM-DD 패턴에서 날짜 부분만 추출
    m = re.match(r"(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", raw)
    if m:
        y, mo, d = m.group(1), m.group(2).zfill(2), m.group(3).zfill(2)
        # 날짜 범위 간단 검증
        if 1 <= int(mo) <= 12 and 1 <= int(d) <= 31:
            return f"{y}-{mo}-{d}"

    return None


# ── 핵심 수집 함수 ─────────────────────────────────────────────────────

def fetch_article_result(url: str) -> ArticleFetchResult:
    """
    Firecrawl API로 URL의 기사 본문과 발행일을 함께 가져옵니다.

    처리 흐름:
        1. API 키 확인 → 미설정이면 즉시 {"body": None, "published_date": None} 반환
        2. Rate limiting — 이전 요청으로부터 최소 0.5초 대기
        3. Firecrawl scrape_url() 호출 (formats=["markdown"])
        4. 본문(markdown) + 발행일(metadata.publishedTime 등) 동시 추출
        5. 실패 시 {"body": None, "published_date": None} 반환

    Args:
        url: 기사 원문 URL

    Returns:
        ArticleFetchResult {"body": str|None, "published_date": "YYYY-MM-DD"|None}
    """
    global _last_request_at

    _empty: ArticleFetchResult = {"body": None, "published_date": None}

    if not url or not url.startswith("http"):
        return _empty

    client = _init_client()
    if client is None:
        return _empty

    # ── Rate Limiting ─────────────────────────────────────────────────
    elapsed = time.monotonic() - _last_request_at
    if elapsed < _RATE_LIMIT_DELAY:
        time.sleep(_RATE_LIMIT_DELAY - elapsed)

    # ── Firecrawl 호출 ────────────────────────────────────────────────
    try:
        # firecrawl-py 1.x: scrape_url(url, formats=[...])
        # firecrawl-py 0.x: scrape_url(url, params={"formats": [...]})
        try:
            result = client.scrape_url(url, formats=["markdown"])
        except TypeError:
            result = client.scrape_url(url, params={"formats": ["markdown"]})

        _last_request_at = time.monotonic()

        # ── 본문 추출 ─────────────────────────────────────────────────
        if isinstance(result, dict):
            body_raw  = result.get("markdown") or result.get("content") or ""
            metadata  = result.get("metadata")
        else:
            body_raw  = getattr(result, "markdown", None) or getattr(result, "content", None) or ""
            metadata  = getattr(result, "metadata", None)

        body = str(body_raw).strip()

        if len(body) < _MIN_BODY_LENGTH:
            logger.debug(
                "Firecrawl 본문 너무 짧음 — url=%r, len=%d (RSS 요약 사용)",
                url[:80], len(body),
            )
            body = None
        elif len(body) > _MAX_BODY_LENGTH:
            body = body[:_MAX_BODY_LENGTH]
            logger.debug("Firecrawl 본문 %d자로 truncate (url=%r)", _MAX_BODY_LENGTH, url[:80])

        # ── 발행일 추출 ───────────────────────────────────────────────
        published_date = _extract_date_from_metadata(metadata)

        if body:
            logger.info(
                "Firecrawl 수집 완료 — 본문 %d자, 발행일 %s (url=%r)",
                len(body), published_date or "미추출", url[:80],
            )
        else:
            logger.debug("Firecrawl 본문 없음 (url=%r)", url[:80])

        return {"body": body, "published_date": published_date}

    except Exception as exc:
        _last_request_at = time.monotonic()  # 오류 시에도 rate limit 적용
        logger.warning("Firecrawl 수집 실패 (url=%r): %s", url[:80], exc)
        return _empty


def fetch_article_body(url: str) -> str | None:
    """
    하위 호환 래퍼 — 본문 문자열만 반환합니다.
    발행일이 필요하면 fetch_article_result()를 직접 사용하세요.
    """
    return fetch_article_result(url)["body"]


def fetch_articles_bodies(
    articles: list[dict],
    url_key: str = "link",
    max_failures: int = 5,
) -> dict[str, ArticleFetchResult]:
    """
    기사 목록의 URL로 본문 + 발행일을 일괄 수집합니다.

    연속 실패(max_failures)가 발생하면 나머지 기사는 스킵합니다
    (Firecrawl 서비스 이상 또는 rate limit 초과 시 전체 파이프라인 지연 방지).

    Args:
        articles:     기사 dict 목록 (url_key 필드에 URL 포함)
        url_key:      URL이 저장된 필드명 (기본값: "link")
        max_failures: 연속 실패 허용 횟수 (초과 시 조기 종료)

    Returns:
        {url: {"body": str|None, "published_date": "YYYY-MM-DD"|None}} 매핑 dict
    """
    if not is_enabled():
        return {}

    results: dict[str, ArticleFetchResult] = {}
    consecutive_failures = 0

    for article in articles:
        url = article.get(url_key, "")
        if not url:
            continue

        if consecutive_failures >= max_failures:
            remaining = sum(
                1 for a in articles
                if a.get(url_key) and a.get(url_key) not in results
            )
            logger.warning(
                "Firecrawl 연속 %d회 실패 — 나머지 %d건 스킵 (RSS 요약 사용)",
                max_failures, remaining,
            )
            break

        fetch_result = fetch_article_result(url)
        results[url] = fetch_result

        if fetch_result["body"] is None:
            consecutive_failures += 1
        else:
            consecutive_failures = 0  # 성공 시 카운터 초기화

    fetched_body  = sum(1 for v in results.values() if v["body"] is not None)
    fetched_date  = sum(1 for v in results.values() if v["published_date"] is not None)
    logger.info(
        "Firecrawl 일괄 수집 완료 — 본문: %d/%d건, 발행일 추출: %d건",
        fetched_body, len(results), fetched_date,
    )
    return results
