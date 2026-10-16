"""
환율·국제금융 리포트 PDF 자동 크롤러

지원 소스:
    국제금융센터 (KCIF) — https://www.kcif.or.kr
    - 검증 완료: requests 기반으로 로그인 없이 PDF 다운로드 가능
    - 처리 흐름: 목록 수집 → 상세 페이지에서 fno 추출 → AuthCheck → PDF 다운로드

크롤링 → PDF 다운로드 → 텍스트 추출 → Vector DB 적재까지 자동화
중복 적재 방지: data/ingested_report_urls.json 으로 URL 캐시 관리
"""

from __future__ import annotations

import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── 상수 ──────────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parents[3] / "data"
_URL_CACHE_PATH = _DATA_DIR / "ingested_report_urls.json"

_BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

_REQUEST_TIMEOUT = 20  # seconds
_CRAWL_DELAY = 1.5     # 과도한 크롤링 방지용 딜레이 (초)


# ── 데이터 모델 ────────────────────────────────────────────────────────────

@dataclass
class ReportItem:
    """크롤링된 개별 리포트"""
    url: str
    source_name: str
    title: str
    published_date: str   # YYYY-MM-DD
    tags: list[str] = field(default_factory=list)


@dataclass
class CrawlStats:
    """크롤링 실행 결과 요약"""
    source: str
    found: int = 0
    downloaded: int = 0
    skipped_cache: int = 0
    failed: int = 0
    vector_chunks: int = 0
    errors: list[str] = field(default_factory=list)


# ── URL 캐시 (중복 적재 방지) ──────────────────────────────────────────────

def _load_url_cache() -> set[str]:
    if _URL_CACHE_PATH.exists():
        try:
            data = json.loads(_URL_CACHE_PATH.read_text(encoding="utf-8"))
            return set(data.get("urls", []))
        except Exception:
            return set()
    return set()


def _save_url_cache(urls: set[str]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _URL_CACHE_PATH.write_text(
        json.dumps(
            {"urls": sorted(urls), "updated_at": datetime.utcnow().isoformat()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ── PDF 텍스트 추출 ────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """PyMuPDF로 PDF 바이트에서 텍스트 추출"""
    try:
        doc = fitz.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
        pages_text = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages_text).strip()
    except Exception as exc:
        logger.warning("PDF 텍스트 추출 실패: %s", exc)
        return ""


# ── 국제금융센터 (KCIF) 크롤러 ────────────────────────────────────────────

class KCIFCrawler:
    """
    국제금융센터 (Korea Center for International Finance) 리포트 크롤러

    처리 흐름:
        1. /annual/reportList?pg={n}&pp=20 에서 rpt_no 목록 수집
        2. /annual/reportView?rpt_no={n} 에서 날짜·fno 추출
        3. POST /comm/AuthCheck → GET /common/file/userDownload?atch_no={fno}
    """

    source_name = "국제금융센터(KCIF)"
    default_tags = ["환율", "FX", "국제금융", "KCIF", "거시경제"]

    _LIST_URL = "https://www.kcif.or.kr/annual/reportList"
    _DETAIL_URL = "https://www.kcif.or.kr/annual/reportView"
    _AUTH_URL = "https://www.kcif.or.kr/comm/AuthCheck"
    _DOWNLOAD_URL = "https://www.kcif.or.kr/common/file/userDownload"

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(_BROWSER_HEADERS)

    def fetch_items_with_fno(
        self, months_back: int, max_pages: int = 30
    ) -> tuple[list[ReportItem], dict[str, str]]:
        """
        리포트 목록과 fno 맵을 함께 반환합니다.

        Returns:
            (items, fno_map)
            fno_map: detail_url → fno (다운로드 키)
        """
        items: list[ReportItem] = []
        fno_map: dict[str, str] = {}
        seen_rpt_nos: set[str] = set()
        cutoff = datetime.now(timezone.utc) - timedelta(days=30 * months_back)
        page = 1

        while page <= max_pages:
            try:
                time.sleep(_CRAWL_DELAY)
                resp = self.session.get(
                    self._LIST_URL,
                    params={"pg": page, "pp": 20},
                    timeout=_REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
            except requests.RequestException as exc:
                logger.warning("KCIF 목록 p.%d 접근 실패: %s", page, exc)
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            report_links = soup.select("a[href*='reportView']")
            if not report_links:
                break

            # 중복 제거
            unique = []
            for lnk in report_links:
                m = re.search(r"rpt_no=(\d+)", lnk.get("href", ""))
                if m and m.group(1) not in seen_rpt_nos:
                    seen_rpt_nos.add(m.group(1))
                    unique.append((m.group(1), lnk.get_text(strip=True)))

            if not unique:
                break

            stop_paging = False
            for rpt_no, link_title in unique:
                try:
                    time.sleep(_CRAWL_DELAY)
                    detail = self._fetch_detail(rpt_no, link_title)
                    if detail is None:
                        continue

                    try:
                        dt = datetime.strptime(detail["date"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
                        if dt < cutoff:
                            stop_paging = True
                            break
                    except ValueError:
                        pass

                    item = detail["item"]
                    items.append(item)
                    fno_map[item.url] = detail["fno"]
                except Exception as exc:
                    logger.debug("KCIF rpt_no=%s 파싱 실패: %s", rpt_no, exc)

            if stop_paging:
                break
            page += 1

        logger.info("KCIF: %d개 리포트+fno 수집 완료", len(items))
        return items, fno_map

    def _fetch_detail(self, rpt_no: str, fallback_title: str) -> dict | None:
        resp = self.session.get(
            self._DETAIL_URL,
            params={"rpt_no": rpt_no},
            timeout=_REQUEST_TIMEOUT,
        )
        soup = BeautifulSoup(resp.text, "html.parser")

        # 날짜: 첫 번째 YYYY.MM.DD 패턴
        date_matches = re.findall(r"(\d{4}\.\d{2}\.\d{2})", resp.text)
        date = date_matches[0].replace(".", "-") if date_matches else datetime.now().strftime("%Y-%m-%d")

        # fno (파일 키)
        fno_matches = re.findall(r"reportdownload\([\"'](.*?)[\"']\)", resp.text)
        if not fno_matches:
            logger.debug("KCIF rpt_no=%s: fno 없음 (PDF 없는 보고서)", rpt_no)
            return None
        fno = fno_matches[0]

        # 제목
        title_el = soup.select_one("title")
        title = title_el.get_text(strip=True).split(" - ")[0] if title_el else fallback_title

        detail_url = f"{self._DETAIL_URL}?rpt_no={rpt_no}"
        return {
            "date": date,
            "fno": fno,
            "item": ReportItem(
                url=detail_url,
                source_name=self.source_name,
                title=title,
                published_date=date,
                tags=self.default_tags,
            ),
        }

    def download_pdf(self, fno: str, rpt_no: str) -> bytes | None:
        """AuthCheck → PDF 다운로드"""
        try:
            auth_resp = self.session.post(
                self._AUTH_URL,
                data={"type": "FILE", "fno": fno, "logType": "D", "lang": "KR"},
                headers={"Referer": f"{self._DETAIL_URL}?rpt_no={rpt_no}"},
                timeout=_REQUEST_TIMEOUT,
            )
            auth_data = auth_resp.json()
            if auth_data.get("auth_yn") in ("N", "N1"):
                logger.warning("KCIF: 파일 접근 거부 (auth_yn=%s)", auth_data.get("auth_yn"))
                return None

            time.sleep(_CRAWL_DELAY)
            # fno는 이미 URL-인코딩된 상태 → params= 사용 시 이중 인코딩되므로 직접 구성
            dl_resp = self.session.get(
                f"{self._DOWNLOAD_URL}?atch_no={fno}",
                timeout=30,
            )
            dl_resp.raise_for_status()
            return dl_resp.content
        except Exception as exc:
            logger.warning("KCIF PDF 다운로드 실패 (fno=%s…): %s", fno[:20], exc)
            return None


# ── 메인 크롤링 함수 ──────────────────────────────────────────────────────

def crawl_and_ingest_fx_reports(
    months_back: int = 3,
    max_pages: int = 30,
) -> list[CrawlStats]:
    """
    KCIF 리포트를 크롤링해 Vector DB에 적재합니다.

    처리 흐름:
        1. 목록 수집 + fno 추출
        2. URL 캐시 확인 → 이미 적재된 URL 스킵
        3. PDF 다운로드 → PyMuPDF 텍스트 추출
        4. Vector DB 적재 (ingest_text_to_vector_db)
        5. URL 캐시 업데이트

    Args:
        months_back: 수집할 과거 기간 (월 단위, 기본값: 3)
        max_pages:   목록 페이지 최대 수 (기본값: 30 = 약 600건)
                     과거 데이터 수집 시 60 이상으로 늘릴 것

    Returns:
        CrawlStats 목록 (현재 KCIF 단일 소스)
    """
    from macro_agent.db.vector.repository import ingest_text_to_vector_db

    crawler = KCIFCrawler()
    stats = CrawlStats(source=crawler.source_name)
    url_cache = _load_url_cache()

    logger.info("=== [%s] 크롤링 시작 (months_back=%d, max_pages=%d) ===",
                crawler.source_name, months_back, max_pages)

    try:
        items, fno_map = crawler.fetch_items_with_fno(months_back, max_pages=max_pages)
    except Exception as exc:
        msg = f"리포트 목록 수집 실패: {exc}"
        logger.error(msg, exc_info=True)
        stats.errors.append(msg)
        return [stats]

    stats.found = len(items)

    for item in items:
        if item.url in url_cache:
            stats.skipped_cache += 1
            logger.debug("캐시 히트, 스킵: %s", item.url)
            continue

        fno = fno_map.get(item.url, "")
        if not fno:
            stats.failed += 1
            continue

        rpt_no_m = re.search(r"rpt_no=(\d+)", item.url)
        rpt_no = rpt_no_m.group(1) if rpt_no_m else ""

        pdf_bytes = crawler.download_pdf(fno, rpt_no)
        if pdf_bytes is None:
            stats.failed += 1
            continue

        text = extract_text_from_pdf(pdf_bytes)
        if len(text.strip()) < 100:
            logger.warning("텍스트 너무 짧음 (%d자), 스킵: %s", len(text), item.url)
            stats.failed += 1
            url_cache.add(item.url)  # 재시도 방지
            continue

        try:
            chunks = ingest_text_to_vector_db(
                text=text,
                metadata={
                    "source_name": item.source_name,
                    "published_date": item.published_date,
                    "url": item.url,
                    "tags": item.tags,
                    "title": item.title,
                },
                source_type="report",
            )
            stats.downloaded += 1
            stats.vector_chunks += chunks
            url_cache.add(item.url)
            logger.info("[%s] 적재 완료: %s → %d 청크", crawler.source_name, item.title[:40], chunks)
        except Exception as exc:
            msg = f"Vector DB 적재 실패 ({item.url}): {exc}"
            logger.error(msg)
            stats.errors.append(msg)
            stats.failed += 1

    _save_url_cache(url_cache)

    logger.info(
        "=== [%s] 완료 — 발견:%d, 적재:%d, 캐시스킵:%d, 실패:%d ===",
        crawler.source_name, stats.found, stats.downloaded,
        stats.skipped_cache, stats.failed,
    )
    return [stats]


def crawl_stats_summary(all_stats: list[CrawlStats]) -> dict[str, Any]:
    """CrawlStats 목록을 요약 dict로 변환"""
    return {
        "sources": [
            {
                "source": s.source,
                "found": s.found,
                "downloaded": s.downloaded,
                "skipped_cache": s.skipped_cache,
                "failed": s.failed,
                "vector_chunks": s.vector_chunks,
                "errors": s.errors,
            }
            for s in all_stats
        ],
        "total_downloaded": sum(s.downloaded for s in all_stats),
        "total_vector_chunks": sum(s.vector_chunks for s in all_stats),
        "total_errors": sum(len(s.errors) for s in all_stats),
    }
