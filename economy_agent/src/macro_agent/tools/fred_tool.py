"""
FRED & yfinance Tool
미국 거시경제 지표 수집: 연방기금금리, 제조업 산업생산, 필라델피아 반도체지수(SOX),
글로벌 공급망 압력 지수(GSCPI)

필요 환경변수:
    FRED_API_KEY: St. Louis Federal Reserve FRED API 키
                  발급: https://fred.stlouisfed.org/docs/api/api_key.html

데이터 소스:
    - FRED API: FEDFUNDS (연방기금금리), IPMAN (제조업 산업생산), GSCPI (글로벌 공급망 압력)
    - yfinance: 필라델피아 반도체지수 ^SOX (FRED에 없는 시장 지수)

GSCPI (Global Supply Chain Pressure Index) 개요:
    - 발표 기관 : 뉴욕 연방준비은행 (NY Fed)
    - FRED ID   : GSCPI
    - 발표 주기 : 월 1회 (매달 첫째 주 금요일)
    - 단위      : 표준편차 (역사적 평균 = 0)
    - 해석      :
        < -1.0  : 공급망 여건 매우 완화 (반도체 투입재 비용 하락 국면)
        -1.0~0  : 정상 이하 — 공급망 압력 낮음
         0~+1.0 : 정상 수준 — 역사적 평균
        +1.0~2.0: 압력 상승 — 납기 지연·운임 상승 주의
        > +2.0  : 심각한 공급망 교란 (COVID 피크: ~4.3, 2021 Q4)
    - 활용      : 반도체 소부장 납기·조달 비용 리스크 선행 지표
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any

import requests
import yfinance as yf
from langchain_core.tools import tool
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# FRED 시계열 ID (https://fred.stlouisfed.org/series/{ID} 에서 확인)
# GSCPI: 2025년경 FRED에서 시리즈 삭제됨 — NY Fed 직접 제공으로 이전
#         대안: https://www.newyorkfed.org/research/policy/gscpi
_FRED_SERIES: dict[str, str] = {
    "fed_funds_rate":   "FEDFUNDS",  # 연방기금 실효금리 (월평균, %)
    "manufacturing_ip": "IPMAN",     # 제조업 산업생산지수
}

# yfinance 티커
_SOX_TICKER = "^SOX"


# ── 내부 헬퍼 함수 ─────────────────────────────────────────────────────

def _get_fred_api_key() -> str:
    key = os.environ.get("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY 환경변수가 설정되지 않았습니다. "
            ".env 파일에 FRED_API_KEY=<your_key>를 추가하세요."
        )
    return key


@retry(
    retry=retry_if_exception_type(requests.exceptions.ConnectionError),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _request_fred_observations(
    series_id: str,
    observation_start: str,
    observation_end: str | None = None,
    frequency: str = "m",  # m=monthly, q=quarterly, a=annual
) -> list[dict[str, str]]:
    """
    FRED series/observations 엔드포인트 호출.
    결측치(".")는 자동 제거됩니다.
    """
    key = _get_fred_api_key()
    params: dict[str, str] = {
        "series_id": series_id,
        "api_key": key,
        "file_type": "json",
        "observation_start": observation_start,
        "frequency": frequency,
        "sort_order": "asc",
    }
    if observation_end:
        params["observation_end"] = observation_end

    resp = requests.get(
        f"{_FRED_BASE_URL}/series/observations",
        params=params,
        timeout=15,
    )
    resp.raise_for_status()

    data = resp.json()
    if "observations" not in data:
        raise ValueError(f"FRED 응답에 observations 키가 없습니다: {data}")

    return [
        {"date": obs["date"], "value": obs["value"]}
        for obs in data["observations"]
        if obs.get("value") not in (".", "", None)
    ]


def _fetch_sox_via_yfinance(months_back: int) -> dict[str, Any]:
    """yfinance로 SOX 월봉 OHLCV 데이터 조회"""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=30 * months_back)

    ticker = yf.Ticker(_SOX_TICKER)
    hist = ticker.history(
        start=start_dt.strftime("%Y-%m-%d"),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
    )

    if hist.empty:
        raise ValueError(
            f"yfinance에서 {_SOX_TICKER} 데이터를 반환하지 않았습니다. "
            "네트워크 상태 또는 티커명을 확인하세요."
        )

    series = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "close": round(float(row["Close"]), 2),
            "high": round(float(row["High"]), 2),
            "low": round(float(row["Low"]), 2),
            "volume": int(row["Volume"]),
        }
        for idx, row in hist.iterrows()
    ]

    latest = series[-1] if series else None
    first_close = series[0]["close"] if series else None
    period_return_pct = (
        round((latest["close"] / first_close - 1) * 100, 2)
        if latest and first_close and first_close != 0
        else None
    )

    return {
        "ticker": _SOX_TICKER,
        "series": series,
        "latest": latest,
        "period_return_pct": period_return_pct,
        "description": "Philadelphia Semiconductor Index (SOX) — 일봉 종가 기준",
        "source": "Yahoo Finance (yfinance)",
    }


def _interpret_pmi(value: float) -> str:
    """PMI 수치를 경기 국면 문자열로 해석"""
    if value >= 55:
        return "강한 확장 (Strong Expansion, ≥55)"
    if value >= 50:
        return "확장 (Expansion, 50–55)"
    if value >= 45:
        return "약한 수축 (Mild Contraction, 45–50)"
    return "강한 수축 (Strong Contraction, <45)"




def _compute_rate_trend(series: list[dict[str, str]], window: int = 3) -> str | None:
    """최근 n개월 금리 방향성 (상승/하락/보합) 판단"""
    if len(series) < 2:
        return None
    try:
        recent = [float(s["value"]) for s in series[-window:]]
        delta = recent[-1] - recent[0]
        if delta > 0.1:
            return f"상승 추세 (+{delta:.2f}%p, 최근 {window}개월)"
        if delta < -0.1:
            return f"하락 추세 ({delta:.2f}%p, 최근 {window}개월)"
        return f"보합 ({delta:+.2f}%p, 최근 {window}개월)"
    except (ValueError, IndexError):
        return None


# ── Public Tool ────────────────────────────────────────────────────────

@tool
def fetch_fred_data(months_back: int = 12) -> dict[str, Any]:
    """
    FRED API와 yfinance를 통해 미국 반도체·거시경제 핵심 지표 3종을 조회합니다:
      1) 필라델피아 반도체지수 (SOX)  — yfinance
      2) 제조업 산업생산지수           — FRED (IPMAN, 발표 지연 ~6주, 최소 3개월 lookback)
      3) 미국 연방기금 실효금리        — FRED (FEDFUNDS)

    ※ GSCPI(글로벌 공급망 압력 지수)는 FRED에서 시리즈 삭제됨 (2025년경).
       NY Fed 직접 제공: https://www.newyorkfed.org/research/policy/gscpi

    반도체 업황 사이클과 미국 통화정책 환경 분석에 사용됩니다.

    Args:
        months_back: 조회할 과거 기간 (월 단위, 기본값: 12)

    Returns:
        Dict with keys: sox_index, manufacturing_ip, federal_funds_rate,
        query_period_start, fetched_at
    """
    start_date_str = (datetime.now() - timedelta(days=30 * months_back)).strftime("%Y-%m-%d")
    result: dict[str, Any] = {
        "query_period_start": start_date_str,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }

    # ── 1. 필라델피아 반도체지수 (SOX) — yfinance ─────────────────────
    try:
        result["sox_index"] = _fetch_sox_via_yfinance(months_back)
        logger.info(
            "SOX 데이터 %d건 수집 완료 (최신: %s)",
            len(result["sox_index"]["series"]),
            result["sox_index"].get("latest", {}).get("date"),
        )
    except Exception as exc:
        logger.error("SOX 데이터 조회 실패: %s", exc, exc_info=True)
        result["sox_index"] = {
            "error": str(exc),
            "ticker": _SOX_TICKER,
            "source": "Yahoo Finance (yfinance)",
        }

    # ── 2. 제조업 산업생산지수 — FRED (IPMAN) ────────────────────────
    # 발표 지연 약 6주 → months_back=1 이면 데이터가 없을 수 있어 최소 3개월 확보
    ip_start = (datetime.now() - timedelta(days=30 * max(months_back, 3))).strftime("%Y-%m-%d")
    try:
        ip_series = _request_fred_observations(
            series_id=_FRED_SERIES["manufacturing_ip"],
            observation_start=ip_start,
        )
        latest_ip = ip_series[-1] if ip_series else None
        result["manufacturing_ip"] = {
            "series": ip_series,
            "latest": latest_ip,
            "series_id": _FRED_SERIES["manufacturing_ip"],
            "description": "Industrial Production: Manufacturing (NAICS) — 기준년도=100",
            "source": "FRED — Federal Reserve Bank of St. Louis",
        }
        logger.info(
            "제조업 산업생산지수 %d건 수집 완료 (최신: %s = %s)",
            len(ip_series),
            latest_ip["date"] if latest_ip else "N/A",
            latest_ip["value"] if latest_ip else "N/A",
        )
    except requests.exceptions.Timeout:
        logger.error("FRED 제조업 산업생산지수 요청 타임아웃")
        result["manufacturing_ip"] = {"error": "Request timeout", "source": "FRED"}
    except requests.exceptions.HTTPError as exc:
        logger.error("FRED HTTP 오류 (제조업 산업생산지수): %s", exc)
        result["manufacturing_ip"] = {
            "error": f"HTTP {exc.response.status_code}",
            "source": "FRED",
        }
    except EnvironmentError:
        raise
    except Exception as exc:
        logger.error("FRED 제조업 산업생산지수 파싱 실패: %s", exc, exc_info=True)
        result["manufacturing_ip"] = {"error": str(exc), "source": "FRED"}

    # ── 3. 연방기금 실효금리 — FRED ───────────────────────────────────
    try:
        rate_series = _request_fred_observations(
            series_id=_FRED_SERIES["fed_funds_rate"],
            observation_start=start_date_str,
        )
        latest_rate = rate_series[-1] if rate_series else None
        trend = _compute_rate_trend(rate_series)

        result["federal_funds_rate"] = {
            "series": rate_series,
            "latest": latest_rate,
            "trend": trend,
            "series_id": _FRED_SERIES["fed_funds_rate"],
            "description": "Federal Funds Effective Rate — 월평균 (%)",
            "source": "FRED — Federal Reserve Bank of St. Louis",
        }
        logger.info(
            "연방기금금리 %d건 수집 완료 (최신: %s = %s%%)",
            len(rate_series),
            latest_rate["date"] if latest_rate else "N/A",
            latest_rate["value"] if latest_rate else "N/A",
        )
    except requests.exceptions.Timeout:
        logger.error("FRED 연방기금금리 요청 타임아웃")
        result["federal_funds_rate"] = {"error": "Request timeout", "source": "FRED"}
    except requests.exceptions.HTTPError as exc:
        logger.error("FRED HTTP 오류 (연방기금금리): %s", exc)
        result["federal_funds_rate"] = {
            "error": f"HTTP {exc.response.status_code}",
            "source": "FRED",
        }
    except EnvironmentError:
        raise
    except Exception as exc:
        logger.error("FRED 연방기금금리 파싱 실패: %s", exc, exc_info=True)
        result["federal_funds_rate"] = {"error": str(exc), "source": "FRED"}

    # ── GSCPI: FRED에서 시리즈 삭제됨 ───────────────────────────────
    # NY Fed 직접 제공으로 이전: https://www.newyorkfed.org/research/policy/gscpi
    result["gscpi"] = {
        "status": "series_retired",
        "description": "GSCPI 시리즈가 FRED에서 삭제됨. NY Fed 직접 제공으로 이전.",
        "source": "NY Federal Reserve Bank",
        "url": "https://www.newyorkfed.org/research/policy/gscpi",
    }
    logger.warning("GSCPI: FRED 시리즈 삭제됨 — NY Fed 직접 URL 참조 필요")

    return result
