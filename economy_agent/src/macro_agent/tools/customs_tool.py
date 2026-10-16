"""
customs_tool.py — 관세청 품목별 수출입 통계 수집 도구

데이터 소스:
    공공데이터포털(data.go.kr) 관세청_품목별 수출입실적 API
    ─ EndPoint: https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList

수집 지표:
    ① 반도체(HS 8542) 수출입 월별 통계
        customs.semiconductor_export_usd    반도체 수출금액 (USD)
        customs.semiconductor_import_usd    반도체 수입금액 (USD)
        customs.semiconductor_balance_usd   반도체 무역수지 (USD)

    ② 대중국 수출입 월별 통계
        customs.china_export_usd            대중국 수출금액 (USD)
        customs.china_import_usd            대중국 수입금액 (USD)
        customs.china_balance_usd           대중국 무역수지 (USD)

    ③ 대미국 수출입 월별 통계
        customs.us_export_usd               대미국 수출금액 (USD)
        customs.us_import_usd               대미국 수입금액 (USD)
        customs.us_balance_usd              대미국 무역수지 (USD)

필요 환경변수:
    CUSTOMS_API_KEY : 공공데이터포털 인증키 (data.go.kr 가입 후 발급)
                      구독 API: 관세청_품목별 수출입실적
                      발급처: https://www.data.go.kr/data/15100475/openapi.do

    미설정 시: 빈 dict 반환 (파이프라인 중단 없음)

API 규격:
    GET https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList
    Query Params (필수):
        serviceKey  : 인증키
        strtYymm    : 시작년월 (YYYYMM, 예: "202601")
        endYymm     : 종료년월 (YYYYMM, 예: "202605")
        hsSgn       : HS 품목코드 (hsSgn 또는 cntyCd 중 1개 이상 필수)
        cntyCd      : 국가코드   (ISO2 코드, 예: CN / US)
    Query Params (선택):
        pageNo      : 페이지 번호 (기본: 1)
        numOfRows   : 페이지당 건수 (기본: 9999)

    Response XML 필드:
        expDlr          수출금액 (USD)
        impDlr          수입금액 (USD)
        balPayments     무역수지 (USD) = expDlr - impDlr
        expWgt          수출중량 (kg)
        impWgt          수입중량 (kg)
        hsCd            HS코드 (6자리)
        statCd          국가코드 (ISO2)
        statCdCntnKor1  국가명 (한국어)
        statKor         품목명 (한국어)
        year            기간 (예: "2026.01" 또는 "총계")
"""

from __future__ import annotations

import logging
import os
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ── API 상수 ──────────────────────────────────────────────────────────

_API_URL = "https://apis.data.go.kr/1220000/nitemtrade/getNitemtradeList"

# HS코드: 반도체
_HS_SEMICONDUCTOR = "8542"   # 전자집적회로(IC) — HS 8542

# 국가코드 (ISO 2자리)
_CNTY_CHINA = "CN"
_CNTY_USA   = "US"
_CNTY_JAPAN = "JP"
_CNTY_EU    = "EU"


# ── 환경변수 ──────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    return os.environ.get("CUSTOMS_API_KEY", "").strip() or None


def is_enabled() -> bool:
    return bool(_get_api_key())


# ── 날짜 헬퍼 ──────────────────────────────────────────────────────────

def _date_range(months_back: int) -> tuple[str, str]:
    """(strtYymm, endYymm) — 현재 월에서 months_back 이전 ~ 현재."""
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    start_month = month - months_back
    start_year  = year
    while start_month <= 0:
        start_month += 12
        start_year  -= 1
    return f"{start_year:04d}{start_month:02d}", f"{year:04d}{month:02d}"


# ── HTTP 헬퍼 ──────────────────────────────────────────────────────────

@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_api(params: dict[str, str]) -> ET.Element:
    """
    API 단일 호출 → XML ElementTree 반환.
    HTTP 오류 또는 API 에러코드 시 예외 발생.
    """
    resp = requests.get(_API_URL, params=params, timeout=20)
    resp.raise_for_status()

    ct = resp.headers.get("Content-Type", "")
    if "text/html" in ct or resp.text.lstrip().startswith("<!"):
        raise ValueError(
            f"API가 HTML을 반환했습니다 (Content-Type: {ct}). "
            "serviceKey 또는 엔드포인트를 확인하세요."
        )

    root = ET.fromstring(resp.text)
    result_code = root.findtext(".//resultCode", "")
    result_msg  = root.findtext(".//resultMsg",  "")
    if result_code and result_code != "00":
        raise ValueError(f"API 오류 [{result_code}]: {result_msg}")

    return root


def _safe_int(v: str | None) -> int | None:
    try:
        return int(v) if v and v not in ("-", "") else None
    except (ValueError, TypeError):
        return None


# ── 응답 파서 ──────────────────────────────────────────────────────────

def _parse_items(root: ET.Element) -> list[dict[str, Any]]:
    """XML에서 item 목록을 파싱합니다."""
    items = []
    for item in root.findall(".//item"):
        row: dict[str, Any] = {c.tag: c.text for c in item}
        # 숫자 필드 변환
        for field in ("expDlr", "impDlr", "balPayments", "expWgt", "impWgt"):
            row[field] = _safe_int(row.get(field))
        items.append(row)
    return items


def _extract_total(items: list[dict]) -> dict[str, Any]:
    """year='총계' 행 추출 (없으면 전체 합산)."""
    totals = [r for r in items if r.get("year") == "총계"]
    if totals:
        t = totals[0]
        return {
            "export_usd":  t.get("expDlr"),
            "import_usd":  t.get("impDlr"),
            "balance_usd": t.get("balPayments"),
        }
    # 합산 폴백
    exp = sum(r["expDlr"] or 0 for r in items if r.get("year") != "총계")
    imp = sum(r["impDlr"] or 0 for r in items if r.get("year") != "총계")
    return {"export_usd": exp or None, "import_usd": imp or None, "balance_usd": (exp - imp) if exp and imp else None}


def _group_by_year(items: list[dict]) -> dict[str, dict[str, Any]]:
    """year 기준으로 그룹화된 월별 통계 반환 (총계 행 제외)."""
    monthly: dict[str, dict] = {}
    for row in items:
        y = row.get("year", "")
        if not y or y == "총계":
            continue
        if y not in monthly:
            monthly[y] = {"export_usd": 0, "import_usd": 0, "balance_usd": 0}
        monthly[y]["export_usd"]  += row.get("expDlr")  or 0
        monthly[y]["import_usd"]  += row.get("impDlr")  or 0
        monthly[y]["balance_usd"] += row.get("balPayments") or 0
    return monthly


# ── Public API ────────────────────────────────────────────────────────

def fetch_semiconductor_trade(months_back: int = 6) -> dict[str, Any]:
    """
    반도체(HS 8542, 전자집적회로) 수출입 통계를 조회합니다.

    Args:
        months_back: 조회할 과거 개월 수 (기본값: 6)

    Returns:
        {
          "hs_code"    : "8542",
          "period"     : "202601 ~ 202606",
          "total"      : {"export_usd": ..., "import_usd": ..., "balance_usd": ...},
          "monthly"    : {"2026.01": {...}, "2026.02": {...}, ...},
          "fetched_at" : ISO8601,
          "error"      : null | <str>,
          "status"     : "ok" | "api_key_missing" | "error",
        }
    """
    if not is_enabled():
        logger.info("CUSTOMS_API_KEY 미설정 — 반도체 수출입 데이터 수집 스킵")
        return {"status": "api_key_missing", "error": "CUSTOMS_API_KEY not set",
                "hs_code": _HS_SEMICONDUCTOR, "total": {}, "monthly": {},
                "fetched_at": datetime.now(timezone.utc).isoformat()}

    strt, end_ = _date_range(months_back)
    result: dict[str, Any] = {
        "hs_code":    _HS_SEMICONDUCTOR,
        "period":     f"{strt} ~ {end_}",
        "total":      {},
        "monthly":    {},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "error":      None,
        "status":     "ok",
    }

    try:
        root  = _call_api({
            "serviceKey": _get_api_key(),
            "strtYymm":   strt,
            "endYymm":    end_,
            "hsSgn":      _HS_SEMICONDUCTOR,
            "pageNo":     "1",
            "numOfRows":  "9999",
        })
        items = _parse_items(root)
        result["total"]   = _extract_total(items)
        result["monthly"] = _group_by_year(items)
        logger.info(
            "반도체 수출입 수집 완료 — 기간: %s~%s, 수출: %s USD",
            strt, end_, result["total"].get("export_usd"),
        )
    except requests.exceptions.Timeout:
        result["error"]  = "요청 타임아웃"
        result["status"] = "error"
        logger.error("관세청 반도체 통계 타임아웃")
    except Exception as exc:
        result["error"]  = str(exc)
        result["status"] = "error"
        logger.error("관세청 반도체 통계 수집 실패: %s", exc, exc_info=True)

    return result


def fetch_country_trade(
    cnty_cd: str,
    cnty_name: str,
    months_back: int = 6,
) -> dict[str, Any]:
    """
    특정 국가와의 수출입 통계를 조회합니다.

    Args:
        cnty_cd:    국가코드 (예: "CN", "US", "JP")
        cnty_name:  국가명 (로깅·결과용, 예: "중국", "미국")
        months_back: 조회 개월 수

    Returns:
        {
          "country_code": "CN",
          "country_name": "중국",
          "period": "202601 ~ 202606",
          "total": {"export_usd": ..., "import_usd": ..., "balance_usd": ...},
          "monthly": {...},
          "fetched_at": ISO8601,
          "error": null | <str>,
          "status": "ok" | "api_key_missing" | "error",
        }
    """
    if not is_enabled():
        return {"status": "api_key_missing", "error": "CUSTOMS_API_KEY not set",
                "country_code": cnty_cd, "country_name": cnty_name,
                "total": {}, "monthly": {},
                "fetched_at": datetime.now(timezone.utc).isoformat()}

    strt, end_ = _date_range(months_back)
    result: dict[str, Any] = {
        "country_code": cnty_cd,
        "country_name": cnty_name,
        "period":       f"{strt} ~ {end_}",
        "total":        {},
        "monthly":      {},
        "fetched_at":   datetime.now(timezone.utc).isoformat(),
        "error":        None,
        "status":       "ok",
    }

    try:
        root  = _call_api({
            "serviceKey": _get_api_key(),
            "strtYymm":   strt,
            "endYymm":    end_,
            "cntyCd":     cnty_cd,
            "pageNo":     "1",
            "numOfRows":  "9999",
        })
        items = _parse_items(root)
        result["total"]   = _extract_total(items)
        result["monthly"] = _group_by_year(items)
        logger.info(
            "대%s 수출입 수집 완료 — 기간: %s~%s, 수출: %s USD",
            cnty_name, strt, end_, result["total"].get("export_usd"),
        )
    except requests.exceptions.Timeout:
        result["error"]  = "요청 타임아웃"
        result["status"] = "error"
        logger.error("관세청 대%s 통계 타임아웃", cnty_name)
    except Exception as exc:
        result["error"]  = str(exc)
        result["status"] = "error"
        logger.error("관세청 대%s 통계 수집 실패: %s", cnty_name, exc, exc_info=True)

    return result


def fetch_customs_trade(months_back: int = 6) -> dict[str, Any]:
    """
    반도체 수출입 + 주요 국가별 수출입을 일괄 조회합니다.
    pipeline.py에서 호출하는 주 함수입니다.

    Args:
        months_back: 조회 개월 수 (기본값: 6)

    Returns:
        {
          "semiconductor": fetch_semiconductor_trade() 결과,
          "china":         fetch_country_trade("CN") 결과,
          "usa":           fetch_country_trade("US") 결과,
          "fetched_at":    ISO8601,
        }
    """
    return {
        "semiconductor": fetch_semiconductor_trade(months_back=months_back),
        "china":         fetch_country_trade(_CNTY_CHINA, "중국",  months_back=months_back),
        "usa":           fetch_country_trade(_CNTY_USA,   "미국",  months_back=months_back),
        "fetched_at":    datetime.now(timezone.utc).isoformat(),
    }


# ── 이전 코드 호환용 별칭 (pipeline.py에서 사용) ─────────────────────

def fetch_customs_20day_data(yyyyMm: str | None = None, **_kwargs) -> dict[str, Any]:
    """
    구 API 호환 래퍼 — 새 API는 20일 동향을 지원하지 않으므로
    반도체 월별 데이터로 대체합니다.
    """
    logger.info("fetch_customs_20day_data → fetch_semiconductor_trade()로 대체됨")
    result = fetch_semiconductor_trade(months_back=2)
    return {
        "period":     result.get("period"),
        "data_type":  "semiconductor_monthly",
        "fetched_at": result.get("fetched_at"),
        "items":      [],   # pipeline normalize 호환용 빈 리스트
        "semiconductor_total": result.get("total"),
        "error":      result.get("error"),
        "status":     result.get("status"),
    }


def fetch_customs_monthly_data(months_back: int = 6, **_kwargs) -> dict[str, Any]:
    """
    구 API 호환 래퍼 — fetch_customs_trade()로 대체.
    """
    logger.info("fetch_customs_monthly_data → fetch_customs_trade()로 대체됨")
    result = fetch_customs_trade(months_back=months_back)
    return {
        "data_type":  "customs_trade",
        "fetched_at": result.get("fetched_at"),
        "items":      [],   # pipeline normalize 호환용 빈 리스트
        "semiconductor": result.get("semiconductor"),
        "china":      result.get("china"),
        "usa":        result.get("usa"),
        "errors":     [],
    }
