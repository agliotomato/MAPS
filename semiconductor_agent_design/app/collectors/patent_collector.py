"""
patent_collector.py
===================
KIPRIS(특허정보넷) REST API를 통해 반도체 관련 특허를 수집합니다.

수집 대상
  ① 출원인 기반 : 한미반도체, 삼성전자, SK하이닉스
  ② IPC 코드 기반 : H01L21/60(TC 본딩), H01L25/065(3D 스택),
                    B23K20/02(열압착 접합), H01L21/683(웨이퍼 본딩),
                    H01L25/18(HBM 패키지)

저장
  - tech_documents  : 개별 특허 레코드 (source_type='patent')
  - metric_observations : 월별 출원 건수 집계 (domain='patent')

환경변수
  KIPRIS_API_KEY        : KIPRIS 서비스키 (필수)
  KIPRIS_LOOKBACK_DAYS  : 수집 기간 (기본 90일)
  KIPRIS_MAX_PAGES      : 페이지 상한 (기본 5)
  HTTP_TIMEOUT_SEC      : HTTP 타임아웃 (기본 20)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from typing import Iterator

import httpx

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

# ── KIPRIS REST API ──────────────────────────────────────────────────
KIPRIS_BASE = "https://plus.kipris.or.kr/openapi/rest"
APPLICANT_SEARCH_EP = f"{KIPRIS_BASE}/patUtiModInfoSearchSevice/applicantNameSearchInfo"
IPC_SEARCH_EP       = f"{KIPRIS_BASE}/patUtiModInfoSearchSevice/ipcCpcSearch"

# ── 수집 대상 ────────────────────────────────────────────────────────
APPLICANT_TARGETS: list[dict] = [
    {"company_code": "042700", "query": "한미반도체", "tag": "hana_microdisplay"},
    {"company_code": "005930", "query": "삼성전자",  "tag": "samsung_elec"},
    {"company_code": "000660", "query": "에스케이하이닉스", "tag": "sk_hynix"},
]

# 반도체 패키징/공정 IPC 코드
IPC_TARGETS: list[dict] = [
    {"ipc": "H01L21/60",  "label": "TC본딩_다이접합",  "tags": ["tc_bonding", "packaging"]},
    {"ipc": "H01L25/065", "label": "3D_집적회로스택", "tags": ["3d_stack", "packaging"]},
    {"ipc": "B23K20/02",  "label": "열압착접합",       "tags": ["thermocompression", "bonding"]},
    {"ipc": "H01L21/683", "label": "웨이퍼본딩",       "tags": ["wafer_bonding", "hybrid_bonding"]},
    {"ipc": "H01L25/18",  "label": "HBM패키지",        "tags": ["hbm", "packaging"]},
]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _api_key() -> str | None:
    return os.getenv("KIPRIS_API_KEY", "").strip() or None


def _lookback_days() -> int:
    return int(os.getenv("KIPRIS_LOOKBACK_DAYS", "90"))


def _max_pages() -> int:
    return int(os.getenv("KIPRIS_MAX_PAGES", "5"))


def _http_timeout() -> float:
    return float(os.getenv("HTTP_TIMEOUT_SEC", "20"))


def _date_range() -> tuple[str, str]:
    """(YYYYMMDD start, YYYYMMDD end)"""
    end = date.today()
    start = end - timedelta(days=_lookback_days())
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# ── XML 파싱 유틸 ────────────────────────────────────────────────────

def _text(el: ET.Element | None, tag: str, ns: str = "") -> str:
    if el is None:
        return ""
    child = el.find(f"{ns}{tag}")
    return (child.text or "").strip() if child is not None else ""


def _parse_application_date(date_str: str) -> datetime | None:
    """YYYYMMDD → datetime UTC"""
    s = date_str.strip().replace("-", "")
    if len(s) < 8:
        return None
    try:
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), tzinfo=timezone.utc)
    except Exception:
        return None


def _patent_uid(app_no: str) -> str:
    return hashlib.sha1(f"kipris|{app_no}".encode()).hexdigest()


def _sanitize(text: str | None) -> str:
    if text is None:
        return ""
    return text.replace("\x00", "")


# ── DB 저장 ──────────────────────────────────────────────────────────

def _upsert_patent_doc(
    app_no: str,
    title: str,
    applicant: str,
    ipc: str,
    app_date: datetime | None,
    open_date: datetime | None,
    reg_date: datetime | None,
    status: str,
    tags: list[str],
    extra: dict,
) -> str:
    uid = _patent_uid(app_no)
    url = f"https://doi.kipris.or.kr/patil/kipriscall?apno={app_no}"
    summary = (
        f"출원번호: {app_no} | 출원인: {applicant} | IPC: {ipc} | "
        f"상태: {status} | 출원일: {(app_date.date() if app_date else 'N/A')}"
    )

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tech_documents(
                  doc_uid, source, source_type, title, url,
                  published_at, collected_at, summary, content,
                  tags, confidence, extra
                ) VALUES (
                  %s, 'kipris', 'patent', %s, %s,
                  %s, %s, %s, NULL,
                  %s, 0.90, %s::jsonb
                )
                ON CONFLICT (doc_uid) DO UPDATE SET
                  title       = EXCLUDED.title,
                  summary     = EXCLUDED.summary,
                  tags        = EXCLUDED.tags,
                  extra       = EXCLUDED.extra,
                  collected_at = EXCLUDED.collected_at
                """,
                (
                    uid,
                    _sanitize(title) or f"특허 {app_no}",
                    url,
                    app_date,
                    _now_utc(),
                    _sanitize(summary),
                    tags,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()
    return uid


def _record_monthly_patent_metric(
    company_code: str,
    year_month: str,   # "YYYY-MM"
    count: int,
    extra: dict,
) -> None:
    """월별 특허 출원 건수 → metric_observations"""
    try:
        y, m = int(year_month[:4]), int(year_month[5:7])
        period_dt = datetime(y, m, 1, tzinfo=timezone.utc)
    except Exception:
        return

    prov_entity = f"kipris:{company_code}:patent_count:{year_month}"

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO metric_observations(
                  company_code, domain, metric_name, metric_value, unit,
                  is_proxy, observed_at, published_at, valid_from, valid_to,
                  source_tier, confidence, prov_entity, source_url, extra
                ) VALUES (
                  %s, 'patent', 'patent_applications_monthly', %s, 'count',
                  FALSE, %s, %s, %s, NULL,
                  2, 0.90, %s,
                  'https://plus.kipris.or.kr/',
                  %s::jsonb
                )
                ON CONFLICT DO NOTHING
                """,
                (
                    company_code, count,
                    _now_utc(), period_dt, period_dt,
                    prov_entity,
                    json.dumps({**extra, "year_month": year_month}, ensure_ascii=False),
                ),
            )
        conn.commit()


# ── KIPRIS API 호출 ──────────────────────────────────────────────────

def _fetch_applicant_page(
    applicant: str,
    start_date: str,
    end_date: str,
    page: int,
    docs_count: int = 20,
) -> list[dict]:
    """출원인명 검색 — 단일 페이지 반환."""
    api_key = _api_key()
    if not api_key:
        return []

    params = {
        "ServiceKey": api_key,
        "applicant": applicant,
        "applicationDate": f"{start_date}~{end_date}",
        "docsStart": (page - 1) * docs_count + 1,
        "docsCount": docs_count,
        "patent": "Y",
        "utility": "N",
    }

    try:
        with httpx.Client(timeout=_http_timeout(), follow_redirects=True) as client:
            resp = client.get(APPLICANT_SEARCH_EP, params=params)
            resp.raise_for_status()
            return _parse_patent_xml(resp.text)
    except Exception as e:
        logger.warning("KIPRIS applicant page fetch failed | applicant=%s page=%d error=%s", applicant, page, e)
        return []


def _fetch_ipc_page(
    ipc: str,
    start_date: str,
    end_date: str,
    page: int,
    docs_count: int = 20,
) -> list[dict]:
    """IPC 코드 검색 — 단일 페이지 반환."""
    api_key = _api_key()
    if not api_key:
        return []

    params = {
        "ServiceKey": api_key,
        "ipcNumber": ipc,
        "applicationDate": f"{start_date}~{end_date}",
        "docsStart": (page - 1) * docs_count + 1,
        "docsCount": docs_count,
        "patent": "Y",
        "utility": "N",
    }

    try:
        with httpx.Client(timeout=_http_timeout(), follow_redirects=True) as client:
            resp = client.get(IPC_SEARCH_EP, params=params)
            resp.raise_for_status()
            return _parse_patent_xml(resp.text)
    except Exception as e:
        logger.warning("KIPRIS IPC page fetch failed | ipc=%s page=%d error=%s", ipc, page, e)
        return []


def _parse_patent_xml(xml_text: str) -> list[dict]:
    """KIPRIS XML 응답 → list of patent dicts."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("KIPRIS XML parse error: %s", e)
        return []

    # 결과 코드 체크
    result_code = _text(root.find(".//header"), "resultCode")
    if result_code not in ("", "00", "0000"):
        result_msg = _text(root.find(".//header"), "resultMsg")
        logger.warning("KIPRIS API error | code=%s msg=%s", result_code, result_msg)
        return []

    patents = []
    for item in root.findall(".//item"):
        app_no    = _text(item, "applicationNumber")
        app_date  = _text(item, "applicationDate")
        title     = _text(item, "inventionTitle")
        applicant = _text(item, "applicantName")
        ipc       = _text(item, "ipcNumber")
        status    = _text(item, "registerStatus")
        open_no   = _text(item, "openNumber")
        open_date = _text(item, "openDate")
        reg_no    = _text(item, "registerNumber")
        reg_date  = _text(item, "registerDate")
        abstract  = _text(item, "astrtCont")  # 요약 (일부 API 버전에서 제공)

        if not app_no:
            continue

        patents.append({
            "app_no":     app_no,
            "app_date":   app_date,
            "title":      title,
            "applicant":  applicant,
            "ipc":        ipc,
            "status":     status,
            "open_no":    open_no,
            "open_date":  open_date,
            "reg_no":     reg_no,
            "reg_date":   reg_date,
            "abstract":   abstract,
        })

    return patents


def _paginated_fetch(
    fetch_fn,
    *args,
    max_pages: int,
    sleep_sec: float = 0.5,
) -> Iterator[dict]:
    """여러 페이지를 순회하며 특허를 가져오는 제너레이터."""
    for page in range(1, max_pages + 1):
        items = fetch_fn(*args, page)
        if not items:
            break
        yield from items
        if len(items) < 20:
            break  # 마지막 페이지
        time.sleep(sleep_sec)


# ── 핵심 수집 함수 ───────────────────────────────────────────────────

def collect_kipris_patents() -> None:
    """KIPRIS 특허 수집 (출원인 + IPC 코드 기반 — 월간 실행 권장)."""
    logger.info("=== collect_kipris_patents START ===")

    api_key = _api_key()
    if not api_key:
        logger.warning(
            "KIPRIS_API_KEY 미설정. KIPRIS 특허 수집을 건너뜁니다. "
            "https://plus.kipris.or.kr 에서 무료 API 키를 발급받으세요."
        )
        return

    start_date, end_date = _date_range()
    max_pages = _max_pages()
    total_inserted = 0

    # ① 출원인 기반 수집
    for target in APPLICANT_TARGETS:
        company_code = target["company_code"]
        query        = target["query"]
        tag          = target["tag"]
        logger.info("출원인 수집: %s (%s)", query, company_code)

        monthly_counts: dict[str, int] = {}

        for pat in _paginated_fetch(
            _fetch_applicant_page, query, start_date, end_date,
            max_pages=max_pages
        ):
            app_date_dt  = _parse_application_date(pat["app_date"])
            open_date_dt = _parse_application_date(pat["open_date"])
            reg_date_dt  = _parse_application_date(pat["reg_date"])

            tags = ["patent", "kipris", tag]
            if app_date_dt:
                ym = f"{app_date_dt.year}-{app_date_dt.month:02d}"
                monthly_counts[ym] = monthly_counts.get(ym, 0) + 1

            extra = {
                "source": "kipris_applicant",
                "app_no": pat["app_no"],
                "applicant": pat["applicant"],
                "ipc": pat["ipc"],
                "open_no": pat["open_no"],
                "reg_no": pat["reg_no"],
                "status": pat["status"],
                "company_code": company_code,
            }

            try:
                _upsert_patent_doc(
                    app_no    = pat["app_no"],
                    title     = pat["title"],
                    applicant = pat["applicant"],
                    ipc       = pat["ipc"],
                    app_date  = app_date_dt,
                    open_date = open_date_dt,
                    reg_date  = reg_date_dt,
                    status    = pat["status"],
                    tags      = tags,
                    extra     = extra,
                )
                total_inserted += 1
            except Exception as e:
                logger.warning("patent upsert failed | app_no=%s error=%s", pat["app_no"], e)

        # 월별 집계 저장
        for ym, cnt in monthly_counts.items():
            try:
                _record_monthly_patent_metric(
                    company_code = company_code,
                    year_month   = ym,
                    count        = cnt,
                    extra        = {"applicant": query, "tag": tag},
                )
            except Exception as e:
                logger.warning("patent metric insert failed | %s %s: %s", company_code, ym, e)

        logger.info("출원인 수집 완료: %s → %d 건", query, sum(monthly_counts.values()))
        time.sleep(1.0)  # API 쿼터 보호

    # ② IPC 코드 기반 수집
    for ipc_target in IPC_TARGETS:
        ipc   = ipc_target["ipc"]
        label = ipc_target["label"]
        tags  = ["patent", "kipris", "ipc_search"] + ipc_target["tags"]
        logger.info("IPC 코드 수집: %s (%s)", ipc, label)

        ipc_count = 0
        monthly_counts_ipc: dict[str, int] = {}

        for pat in _paginated_fetch(
            _fetch_ipc_page, ipc, start_date, end_date,
            max_pages=max_pages
        ):
            app_date_dt  = _parse_application_date(pat["app_date"])
            open_date_dt = _parse_application_date(pat["open_date"])
            reg_date_dt  = _parse_application_date(pat["reg_date"])

            # IPC 수집은 company_code를 applicant명으로 추정 (KRX 종목 매핑)
            applicant_lower = (pat["applicant"] or "").lower()
            if "삼성" in applicant_lower or "samsung" in applicant_lower:
                c_code = "005930"
            elif "하이닉스" in applicant_lower or "hynix" in applicant_lower:
                c_code = "000660"
            elif "한미" in applicant_lower:
                c_code = "042700"
            else:
                c_code = "SEMI_ETC"  # 기타 반도체 업체

            if app_date_dt:
                ym = f"{app_date_dt.year}-{app_date_dt.month:02d}"
                monthly_counts_ipc[ym] = monthly_counts_ipc.get(ym, 0) + 1

            extra = {
                "source": "kipris_ipc",
                "ipc_query": ipc,
                "ipc_label": label,
                "app_no": pat["app_no"],
                "applicant": pat["applicant"],
                "ipc": pat["ipc"],
                "status": pat["status"],
                "company_code": c_code,
            }

            try:
                _upsert_patent_doc(
                    app_no    = pat["app_no"],
                    title     = pat["title"],
                    applicant = pat["applicant"],
                    ipc       = pat["ipc"],
                    app_date  = app_date_dt,
                    open_date = open_date_dt,
                    reg_date  = reg_date_dt,
                    status    = pat["status"],
                    tags      = tags,
                    extra     = extra,
                )
                ipc_count += 1
                total_inserted += 1
            except Exception as e:
                logger.warning("IPC patent upsert failed | app_no=%s error=%s", pat["app_no"], e)

        # IPC 기반 집계는 'SEMI_IPC' 집합 단위로 저장
        for ym, cnt in monthly_counts_ipc.items():
            try:
                _record_monthly_patent_metric(
                    company_code = "SEMI_IPC",
                    year_month   = ym,
                    count        = cnt,
                    extra        = {"ipc": ipc, "label": label},
                )
            except Exception as e:
                logger.warning("IPC metric insert failed | %s %s: %s", ipc, ym, e)

        logger.info("IPC 수집 완료: %s → %d 건", ipc, ipc_count)
        time.sleep(1.0)

    logger.info("=== collect_kipris_patents DONE | total_inserted=%d ===", total_inserted)
