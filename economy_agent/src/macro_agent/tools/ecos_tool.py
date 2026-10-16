"""
ECOS (Economic Statistics System) API Tool
한국은행 경제통계시스템 API를 통해 환율 및 반도체 경제지표 수집

필요 환경변수:
    ECOS_API_KEY: 한국은행 ECOS 오픈API 인증키 (https://ecos.bok.or.kr/api/#/DevGuide/TokenKey)

ECOS StatisticSearch 엔드포인트 형식:
    GET /StatisticSearch/{KEY}/json/kr/{start_row}/{end_row}/{stat_code}/{cycle}/{start}/{end}/{item_code}
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import requests
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_ECOS_BASE_URL = "https://ecos.bok.or.kr/api"

# ECOS 통계표 코드 (한국은행 ECOS API 기준)
# 실제 코드는 https://ecos.bok.or.kr/api/#/StatisticalStatList 에서 확인
_STAT_CODES: dict[str, str] = {
    "exchange_rate_daily": "731Y001",    # 외국환 거래 기준환율 (일별)
    "exchange_rate_monthly": "731Y004",  # 주요국 통화의 대원화환율 (월평균, 전체 통화 참조용)
    "semiconductor_export": "402Y016",   # 수출물가지수 — 품목별
    "production_shipment_inventory": "901Y032",  # 산업별 생산·출하·재고지수
}

# ECOS 통계항목 코드
_ITEM_CODES: dict[str, str] = {
    "usd": "0000001",           # 미국 달러(USD)
    "jpy": "0000002",           # 일본 엔(JPY)
    "eur": "0000003",           # 유로(EUR)
    "cny": "0000053",           # 중국 위안(CNY)
    "electronics_mfg": "I11ACQ",  # 전자부품·컴퓨터·영상·음향·통신장비 제조업
}


# ── 내부 헬퍼 함수 ─────────────────────────────────────────────────────

def _get_api_key() -> str:
    key = os.environ.get("ECOS_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "ECOS_API_KEY 환경변수가 설정되지 않았습니다. "
            ".env 파일에 ECOS_API_KEY=<your_key>를 추가하세요."
        )
    return key


def _build_ecos_url(
    stat_code: str,
    cycle: str,
    start_date: str,
    end_date: str,
    item_code: str = "",
    row_limit: int = 200,
) -> str:
    """ECOS StatisticSearch URL 조립"""
    key = _get_api_key()
    base = f"{_ECOS_BASE_URL}/StatisticSearch/{key}/json/kr/1/{row_limit}"
    path = f"/{stat_code}/{cycle}/{start_date}/{end_date}"
    item = f"/{item_code}" if item_code else ""
    return base + path + item


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _request_ecos(url: str, timeout: int = 15) -> dict:
    """ECOS API HTTP GET 요청 (네트워크 오류 시 최대 3회 재시도)"""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _extract_rows(data: dict, service: str = "StatisticSearch") -> list[dict]:
    """응답 JSON에서 row 리스트 추출, API 오류 코드를 예외로 변환"""
    if service not in data:
        error = data.get("RESULT", {})
        code = error.get("CODE", "UNKNOWN")
        msg = error.get("MESSAGE", "알 수 없는 오류")
        raise ValueError(f"ECOS API 오류 [{code}]: {msg}")
    return data[service].get("row", [])


def _monthly_date_range(months_back: int) -> tuple[str, str]:
    """(start_YYYYMM, end_YYYYMM) 반환"""
    now = datetime.now()
    start = now - timedelta(days=30 * months_back)
    return start.strftime("%Y%m"), now.strftime("%Y%m")


def _daily_date_range(months_back: int) -> tuple[str, str]:
    """(start_YYYYMMDD, end_YYYYMMDD) 반환"""
    now = datetime.now()
    start = now - timedelta(days=30 * months_back)
    return start.strftime("%Y%m%d"), now.strftime("%Y%m%d")


def _safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "", ".") else None
    except (ValueError, TypeError):
        return None


# ── Public Tool ────────────────────────────────────────────────────────

@tool
def fetch_ecos_data(months_back: int = 6) -> dict[str, Any]:
    """
    한국은행 ECOS API로 다음 4가지 지표를 일괄 조회합니다:
      1) 원/달러(USD) 일별 기준환율 (731Y001, 일별)
      2) 주요국 전체 환율 월평균 (731Y004, 월별, 참조용)
      3) 전자부품 제조업 생산·출하·재고지수 (반도체 포함, 월별)
      4) 수출물가지수 — 품목별 (월별)

    반도체 도메인 거시경제 분석의 한국 측 데이터 수집에 사용됩니다.

    Args:
        months_back: 조회할 과거 기간 (월 단위, 기본값: 6)

    Returns:
        Dict with keys: usd_krw, all_exchange_rates, production_shipment_inventory,
        export_price_index, query_period, fetched_at, source
    """
    start_daily, end_daily = _daily_date_range(months_back)
    start_monthly, end_monthly = _monthly_date_range(months_back)
    result: dict[str, Any] = {
        "query_period": f"{start_daily} ~ {end_daily}",
        "source": "ECOS BOK (https://ecos.bok.or.kr)",
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }

    # ── 1. 원/달러 일별 기준환율 (731Y001) ───────────────────────────
    try:
        url = _build_ecos_url(
            stat_code=_STAT_CODES["exchange_rate_daily"],
            cycle="D",
            start_date=start_daily,
            end_date=end_daily,
            item_code=_ITEM_CODES["usd"],
            row_limit=2000,
        )
        rows = _extract_rows(_request_ecos(url))
        usd_series = [
            {"date": r["TIME"], "krw_per_usd": _safe_float(r.get("DATA_VALUE"))}
            for r in rows
            if _safe_float(r.get("DATA_VALUE")) is not None
        ]
        result["usd_krw"] = {
            "series": usd_series,
            "latest": usd_series[-1] if usd_series else None,
            "unit": "KRW per 1 USD (일별 기준환율)",
        }
        logger.info("원/달러 일별 환율 %d건 수집 완료", len(usd_series))
    except EnvironmentError:
        raise
    except requests.exceptions.Timeout:
        logger.error("ECOS 원/달러 환율 조회 타임아웃")
        result["usd_krw"] = {"error": "Request timeout"}
    except requests.exceptions.HTTPError as exc:
        logger.error("ECOS HTTP 오류 (원/달러): %s", exc)
        result["usd_krw"] = {"error": f"HTTP {exc.response.status_code}"}
    except (ValueError, KeyError) as exc:
        logger.error("ECOS 원/달러 데이터 파싱 실패: %s", exc)
        result["usd_krw"] = {"error": str(exc)}

    # ── 2. 주요국 전체 환율 (월평균, 참조용) ─────────────────────────
    try:
        url = _build_ecos_url(
            stat_code=_STAT_CODES["exchange_rate_monthly"],
            cycle="M",
            start_date=start_monthly,
            end_date=end_monthly,
            row_limit=1000,
        )
        rows = _extract_rows(_request_ecos(url))
        all_rates = [
            {
                "currency_name": r.get("ITEM_NAME1", ""),
                "item_code": r.get("ITEM_CODE1", ""),
                "date": r["TIME"],
                "rate": _safe_float(r.get("DATA_VALUE")),
                "unit": r.get("UNIT_NAME", ""),
            }
            for r in rows
            if _safe_float(r.get("DATA_VALUE")) is not None
        ]
        unique_currencies = len({r["currency_name"] for r in all_rates})
        result["all_exchange_rates"] = {
            "series": all_rates,
            "currency_count": unique_currencies,
        }
        logger.info("전체 환율 %d건 (%d개 통화) 수집 완료", len(all_rates), unique_currencies)
    except Exception as exc:
        logger.error("ECOS 전체 환율 조회 실패: %s", exc, exc_info=True)
        result["all_exchange_rates"] = {"error": str(exc)}

    # ── 3. 전자부품 제조업 생산·출하·재고지수 (901Y032/I11ACQ) ────────
    # 발표 지연 약 2개월 → months_back=1 이면 미발표 구간만 조회돼 INFO-200 반환
    # → 최소 3개월 lookback 으로 보정
    psi_start, psi_end = _monthly_date_range(max(months_back, 3))
    try:
        url = _build_ecos_url(
            stat_code=_STAT_CODES["production_shipment_inventory"],
            cycle="M",
            start_date=psi_start,
            end_date=psi_end,
            item_code=_ITEM_CODES["electronics_mfg"],
            row_limit=500,
        )
        rows = _extract_rows(_request_ecos(url))

        production, shipment, inventory = [], [], []
        for r in rows:
            name2 = r.get("ITEM_NAME2", "")
            val = _safe_float(r.get("DATA_VALUE"))
            if val is None:
                continue
            entry = {"date": r["TIME"], "index": val, "type": name2}
            if "생산지수(원지수)" in name2:
                production.append(entry)
            elif "출하지수(원지수)" in name2:
                shipment.append(entry)
            elif "재고지수(원지수)" in name2:
                inventory.append(entry)

        result["production_shipment_inventory"] = {
            "description": "전자부품·컴퓨터·영상·음향·통신장비 제조업 (기준년도=100)",
            "production": {"series": production, "latest": production[-1] if production else None},
            "shipment": {"series": shipment, "latest": shipment[-1] if shipment else None},
            "inventory": {"series": inventory, "latest": inventory[-1] if inventory else None},
        }
        logger.info("생산·출하·재고지수 각 %d/%d/%d건 수집 완료",
                    len(production), len(shipment), len(inventory))
    except Exception as exc:
        logger.error("ECOS 생산·출하·재고지수 조회 실패: %s", exc, exc_info=True)
        result["production_shipment_inventory"] = {"error": str(exc)}

    # ── 4. 수출물가지수 — 품목별 (403Y016) ───────────────────────────
    # 발표 지연 약 1개월 → 최소 3개월 lookback 으로 보정
    epi_start, epi_end = _monthly_date_range(max(months_back, 3))
    try:
        url = _build_ecos_url(
            stat_code=_STAT_CODES["semiconductor_export"],
            cycle="M",
            start_date=epi_start,
            end_date=epi_end,
            row_limit=200,
        )
        rows = _extract_rows(_request_ecos(url))
        export_prices = [
            {
                "date": r["TIME"],
                "item": r.get("ITEM_NAME1", ""),
                "index": _safe_float(r.get("DATA_VALUE")),
                "unit": r.get("UNIT_NAME", ""),
            }
            for r in rows
            if _safe_float(r.get("DATA_VALUE")) is not None
        ]
        result["export_price_index"] = {
            "series": export_prices,
            "description": "수출물가지수 — 품목별 (기준년도=100)",
            "latest_count": len({r["item"] for r in export_prices}),
        }
        logger.info("수출물가지수 %d건 수집 완료", len(export_prices))
    except Exception as exc:
        logger.error("ECOS 수출물가지수 조회 실패: %s", exc, exc_info=True)
        result["export_price_index"] = {"error": str(exc)}

    return result
