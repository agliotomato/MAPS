"""
market_collector.py
===================
① KRX 일별 주가  (pykrx — 무료, 로그인 불필요)
   타겟: 삼성전자 005930 / SK하이닉스 000660 / 한미반도체 042700
② 글로벌 반도체 기업 주가  (yfinance)
   TSMC TSM / Micron MU / NVIDIA NVDA / ASML ASML / Lam LRCX /
   Applied Materials AMAT / KLA KLAC / Intel INTC / AMD AMD
③ 글로벌 기업 분기 실적  (yfinance quarterly financials)

DB: price_daily (KRX + 글로벌)
    metric_observations (EPS, Revenue, Gross Margin 등)
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from typing import Any

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

# ── 대상 종목 ───────────────────────────────────────────────────────
KRX_TARGETS: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "042700": "한미반도체",
}

GLOBAL_TARGETS: dict[str, str] = {
    "TSM":  "TSMC",
    "MU":   "Micron Technology",
    "NVDA": "NVIDIA",
    "ASML": "ASML Holding",
    "LRCX": "Lam Research",
    "AMAT": "Applied Materials",
    "KLAC": "KLA Corporation",
    "INTC": "Intel",
    "AMD":  "Advanced Micro Devices",
}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_companies() -> None:
    """companies 테이블에 종목 등록 (KRX + Global)."""
    rows: list[tuple] = [("MARKET", "Market-wide indicators", "MACRO", "GLOBAL")]
    for code, name in KRX_TARGETS.items():
        rows.append((code, name, "KOSPI", "KR"))
    for ticker, name in GLOBAL_TARGETS.items():
        rows.append((ticker, name, "NYSE/NASDAQ", "US"))

    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            for code, name, market, country in rows:
                cur.execute(
                    """
                    INSERT INTO companies(company_code, company_name, market, country)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (company_code) DO UPDATE
                      SET company_name = EXCLUDED.company_name,
                          market = EXCLUDED.market
                    """,
                    (code, name, market, country),
                )
        conn.commit()


def _insert_metric_once(
    *,
    company_code: str,
    domain: str,
    metric_name: str,
    metric_value: float,
    unit: str,
    observed_at: datetime,
    published_at: datetime,
    valid_from: datetime,
    source_tier: int,
    confidence: float,
    prov_entity: str,
    source_url: str,
    extra: dict[str, Any],
    is_proxy: bool = False,
) -> bool:
    """Insert a metric row once by provenance key."""
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM metric_observations WHERE prov_entity=%s LIMIT 1",
                (prov_entity,),
            )
            if cur.fetchone():
                return False
            cur.execute(
                """
                INSERT INTO metric_observations(
                  company_code, domain, metric_name, metric_value, unit,
                  is_proxy, observed_at, published_at, valid_from, valid_to,
                  source_tier, confidence, prov_entity, source_url, extra
                ) VALUES (
                  %s,%s,%s,%s,%s,
                  %s,%s,%s,%s,NULL,
                  %s,%s,%s,%s,%s::jsonb
                )
                """,
                (
                    company_code, domain, metric_name, metric_value, unit,
                    is_proxy, observed_at, published_at, valid_from,
                    source_tier, confidence, prov_entity, source_url,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()
    return True


# ════════════════════════════════════════════════════════════════════
# ① KRX 주가
# ════════════════════════════════════════════════════════════════════

def _market_history_lookback_days() -> int:
    return int(os.getenv("MARKET_HISTORY_LOOKBACK_DAYS", "1825"))  # 기본 5년


def _krx_lookback_days() -> int:
    return int(os.getenv("KRX_LOOKBACK_DAYS", str(_market_history_lookback_days())))


def collect_krx_daily() -> None:
    """KRX OHLCV 일별 데이터 수집 (pykrx 사용)."""
    logger.info("=== KRX: collect_krx_daily START ===")
    try:
        from pykrx import stock  # type: ignore
    except ImportError:
        logger.error("pykrx not installed. pip install pykrx --break-system-packages")
        return

    _ensure_companies()
    lookback = _krx_lookback_days()
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    total_inserted = 0
    for code, name in KRX_TARGETS.items():
        try:
            df = stock.get_market_ohlcv(start_str, end_str, code)
            if df is None or df.empty:
                logger.warning("KRX no data: %s %s", code, name)
                continue

            rows = []
            for idx, row in df.iterrows():
                trade_date = idx.date() if hasattr(idx, "date") else idx
                # pykrx 컬럼명: 시가/고가/저가/종가/거래량
                open_p  = float(row.get("시가", row.get("Open", 0)) or 0)
                high_p  = float(row.get("고가", row.get("High", 0)) or 0)
                low_p   = float(row.get("저가", row.get("Low", 0)) or 0)
                close_p = float(row.get("종가", row.get("Close", 0)) or 0)
                volume  = int(row.get("거래량", row.get("Volume", 0)) or 0)
                if close_p == 0:
                    continue
                rows.append((code, trade_date, open_p, high_p, low_p, close_p, volume))

            with get_pg_conn() as conn:
                with conn.cursor() as cur:
                    for r in rows:
                        cur.execute(
                            """
                            INSERT INTO price_daily(
                              company_code, trade_date, open, high, low, close, volume
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (company_code, trade_date) DO UPDATE
                              SET open=EXCLUDED.open, high=EXCLUDED.high,
                                  low=EXCLUDED.low, close=EXCLUDED.close,
                                  volume=EXCLUDED.volume
                            """,
                            r,
                        )
                conn.commit()

            logger.info("KRX inserted: %s(%s) %d rows", name, code, len(rows))
            total_inserted += len(rows)
            time.sleep(0.5)

        except Exception as e:
            logger.error("KRX collect failed: %s %s: %s", code, name, e)

    logger.info("=== KRX done: total_rows=%d ===", total_inserted)


def collect_krx_investor_flows() -> None:
    """KRX 투자자별 순매수 금액 수집."""
    logger.info("=== KRX investor flows: START ===")
    try:
        from pykrx import stock  # type: ignore
    except ImportError:
        logger.error("pykrx not installed. pip install pykrx")
        return

    _ensure_companies()
    lookback = int(
        os.getenv(
            "KRX_FLOW_LOOKBACK_DAYS",
            os.getenv("KRX_LOOKBACK_DAYS", str(_market_history_lookback_days())),
        )
    )
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")

    column_map = {
        "개인": "retail_net_buy_krw",
        "외국인합계": "foreign_net_buy_krw",
        "외국인": "foreign_net_buy_krw",
        "기관합계": "institution_net_buy_krw",
        "기관": "institution_net_buy_krw",
        "기타법인": "other_corp_net_buy_krw",
    }

    inserted = 0
    for code, name in KRX_TARGETS.items():
        try:
            df = stock.get_market_trading_value_by_date(start_str, end_str, code)
            if df is None or df.empty:
                logger.warning("KRX investor flow no data: %s %s", code, name)
                continue

            for idx, row in df.iterrows():
                trade_date = idx.date() if hasattr(idx, "date") else idx
                observed = datetime.combine(trade_date, datetime.min.time(), tzinfo=timezone.utc)
                for col, metric_name in column_map.items():
                    if col not in row:
                        continue
                    try:
                        value = float(row[col])
                    except Exception:
                        continue
                    prov = f"krx_flow:{code}:{metric_name}:{trade_date}"
                    if _insert_metric_once(
                        company_code=code,
                        domain="market_flow",
                        metric_name=metric_name,
                        metric_value=value,
                        unit="KRW",
                        observed_at=observed,
                        published_at=observed,
                        valid_from=observed,
                        source_tier=2,
                        confidence=0.90,
                        prov_entity=prov,
                        source_url="https://data.krx.co.kr/",
                        extra={"source": "pykrx", "company_name": name, "raw_column": col},
                    ):
                        inserted += 1

            logger.info("KRX investor flow inserted: %s(%s)", name, code)
            time.sleep(0.5)
        except Exception as e:
            logger.error("KRX investor flow failed: %s %s: %s", code, name, e)

    logger.info("=== KRX investor flows done: total_metrics=%d ===", inserted)


# ════════════════════════════════════════════════════════════════════
# ② 글로벌 반도체 기업 주가  (yfinance)
# ════════════════════════════════════════════════════════════════════

def collect_global_stock_prices() -> None:
    """TSMC/Micron/NVIDIA 등 글로벌 반도체 기업 주가 수집."""
    logger.info("=== Global stocks: START ===")
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.error("yfinance not installed")
        return

    _ensure_companies()
    lookback = int(os.getenv("GLOBAL_STOCK_LOOKBACK_DAYS", str(_market_history_lookback_days())))
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)

    total = 0
    for ticker, name in GLOBAL_TARGETS.items():
        try:
            df = yf.download(
                ticker,
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                auto_adjust=True,
                progress=False,
            )
            if df is None or df.empty:
                logger.warning("yfinance no data: %s", ticker)
                continue

            rows = []
            for idx, row in df.iterrows():
                trade_date = idx.date() if hasattr(idx, "date") else idx
                # auto_adjust=True → Open/High/Low/Close are adjusted
                def safe_float(col):
                    v = row.get(col)
                    if v is None:
                        return 0.0
                    # yfinance sometimes returns Series for multi-level columns
                    if hasattr(v, 'iloc'):
                        v = v.iloc[0]
                    try:
                        return float(v)
                    except Exception:
                        return 0.0

                close_p = safe_float("Close")
                if close_p == 0:
                    continue
                rows.append((
                    ticker, trade_date,
                    safe_float("Open"), safe_float("High"),
                    safe_float("Low"), close_p,
                    int(safe_float("Volume")),
                ))

            with get_pg_conn() as conn:
                with conn.cursor() as cur:
                    for r in rows:
                        cur.execute(
                            """
                            INSERT INTO price_daily(
                              company_code, trade_date, open, high, low, close, volume
                            ) VALUES (%s,%s,%s,%s,%s,%s,%s)
                            ON CONFLICT (company_code, trade_date) DO UPDATE
                              SET open=EXCLUDED.open, high=EXCLUDED.high,
                                  low=EXCLUDED.low, close=EXCLUDED.close,
                                  volume=EXCLUDED.volume
                            """,
                            r,
                        )
                conn.commit()

            logger.info("Global stock inserted: %s(%s) %d rows", name, ticker, len(rows))
            total += len(rows)
            time.sleep(0.3)

        except Exception as e:
            logger.error("Global stock failed: %s %s: %s", ticker, name, e)

    logger.info("=== Global stocks done: total=%d ===", total)


# ════════════════════════════════════════════════════════════════════
# ③ 글로벌 기업 분기 실적  (yfinance)
# ════════════════════════════════════════════════════════════════════

# 실적 지표 → metric_name 매핑
EARNINGS_METRICS = {
    "Total Revenue":      ("revenue_usd",        "USD", False),
    "Gross Profit":       ("gross_profit_usd",    "USD", False),
    "Net Income":         ("net_income_usd",      "USD", False),
    "Operating Income":   ("operating_income_usd","USD", False),
    "Research And Development": ("rd_expense_usd","USD", False),
}


def collect_global_earnings() -> None:
    """TSMC/Micron/NVIDIA 등 분기 실적 (EPS, Revenue, Gross Margin) 수집."""
    logger.info("=== Global earnings: START ===")
    try:
        import yfinance as yf  # type: ignore
    except ImportError:
        logger.error("yfinance not installed")
        return

    _ensure_companies()
    total = 0

    for ticker, name in GLOBAL_TARGETS.items():
        try:
            tkr = yf.Ticker(ticker)
            # 분기 재무제표
            qf = tkr.quarterly_financials  # DataFrame: rows=metrics, cols=quarters
            if qf is None or qf.empty:
                logger.warning("No quarterly financials: %s", ticker)
                continue

            for metric_label, (metric_name, unit, is_proxy) in EARNINGS_METRICS.items():
                if metric_label not in qf.index:
                    continue
                series = qf.loc[metric_label]
                for period_end, value in series.items():
                    if value is None:
                        continue
                    try:
                        val = float(value)
                    except Exception:
                        continue
                    if val == 0:
                        continue

                    # period_end → published_at (실적 발표는 보통 분기 종료 후 4~6주)
                    if hasattr(period_end, "to_pydatetime"):
                        period_dt = period_end.to_pydatetime().replace(tzinfo=timezone.utc)
                    else:
                        period_dt = datetime(
                            getattr(period_end, "year", 2000),
                            getattr(period_end, "month", 1),
                            getattr(period_end, "day", 1),
                            tzinfo=timezone.utc,
                        )

                    prov_entity = f"yfinance:{ticker}:{metric_name}:{period_dt.date()}"

                    with get_pg_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO metric_observations(
                                  company_code, domain, metric_name, metric_value, unit,
                                  is_proxy, observed_at, published_at, valid_from, valid_to,
                                  source_tier, confidence, prov_entity, source_url, extra
                                ) VALUES (
                                  %s,'financials',%s,%s,%s,
                                  %s,%s,%s,%s,NULL,
                                  2,0.85,%s,%s,%s::jsonb
                                )
                                ON CONFLICT DO NOTHING
                                """,
                                (
                                    ticker, metric_name, val, unit,
                                    is_proxy,
                                    _now_utc(), period_dt, period_dt,
                                    prov_entity,
                                    f"https://finance.yahoo.com/quote/{ticker}/financials",
                                    json.dumps({"source": "yfinance", "period_end": str(period_dt.date())}),
                                ),
                            )
                        conn.commit()
                    total += 1

            # EPS (별도 처리)
            eps_df = tkr.quarterly_earnings
            if eps_df is not None and not eps_df.empty and "EPS" in eps_df.columns:
                for period_idx, row in eps_df.iterrows():
                    eps_val = row.get("EPS")
                    if eps_val is None:
                        continue
                    try:
                        eps_val = float(eps_val)
                    except Exception:
                        continue
                    if hasattr(period_idx, "to_pydatetime"):
                        period_dt = period_idx.to_pydatetime().replace(tzinfo=timezone.utc)
                    else:
                        period_dt = _now_utc()
                    prov_entity = f"yfinance:{ticker}:eps:{period_dt.date()}"
                    with get_pg_conn() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                """
                                INSERT INTO metric_observations(
                                  company_code, domain, metric_name, metric_value, unit,
                                  is_proxy, observed_at, published_at, valid_from, valid_to,
                                  source_tier, confidence, prov_entity, source_url, extra
                                ) VALUES (
                                  %s,'financials','eps_actual',%s,'USD',
                                  FALSE,%s,%s,%s,NULL,
                                  2,0.85,%s,%s,%s::jsonb
                                )
                                ON CONFLICT DO NOTHING
                                """,
                                (
                                    ticker, eps_val,
                                    _now_utc(), period_dt, period_dt,
                                    prov_entity,
                                    f"https://finance.yahoo.com/quote/{ticker}/earnings",
                                    json.dumps({"source": "yfinance_eps"}),
                                ),
                            )
                        conn.commit()
                    total += 1

            logger.info("Earnings collected: %s(%s)", name, ticker)
            time.sleep(0.5)

        except Exception as e:
            logger.error("Earnings failed: %s %s: %s", ticker, name, e)

    logger.info("=== Global earnings done: total_metrics=%d ===", total)


def collect_ecos_exchange_rate() -> None:
    """한국은행 ECOS 원/달러 환율 수집."""
    api_key = os.getenv("ECOS_API_KEY", "").strip()
    if not api_key:
        logger.warning("ECOS_API_KEY is empty. Skip collect_ecos_exchange_rate.")
        return

    _ensure_companies()
    stat_code = os.getenv("ECOS_USD_KRW_STAT_CODE", "731Y001")
    item_code = os.getenv("ECOS_USD_KRW_ITEM_CODE", "0000001")
    cycle = os.getenv("ECOS_USD_KRW_CYCLE", "DD")
    lookback = int(os.getenv("ECOS_EXCHANGE_LOOKBACK_DAYS", "365"))
    end_dt = date.today()
    start_dt = end_dt - timedelta(days=lookback)
    start = start_dt.strftime("%Y%m%d")
    end = end_dt.strftime("%Y%m%d")

    url = (
        f"https://ecos.bok.or.kr/api/StatisticSearch/{api_key}/json/kr/1/10000/"
        f"{stat_code}/{cycle}/{start}/{end}/{item_code}"
    )

    logger.info("=== ECOS USD/KRW: START stat=%s item=%s range=%s-%s ===", stat_code, item_code, start, end)
    try:
        import httpx
        with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20")), follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.error("ECOS exchange fetch failed: %s", e)
        return

    rows = (data.get("StatisticSearch") or {}).get("row") or []
    if not rows:
        logger.warning("ECOS exchange returned no rows. Check ECOS_USD_KRW_STAT_CODE/ITEM_CODE.")
        return

    inserted = 0
    for r in rows:
        try:
            time_value = str(r.get("TIME", "")).strip()
            raw_value = str(r.get("DATA_VALUE", "")).replace(",", "").strip()
            if not time_value or not raw_value:
                continue
            if len(time_value) == 8:
                obs = datetime.strptime(time_value, "%Y%m%d").replace(tzinfo=timezone.utc)
            elif len(time_value) == 6:
                obs = datetime.strptime(time_value, "%Y%m").replace(tzinfo=timezone.utc)
            else:
                continue
            value = float(raw_value)
        except Exception:
            continue

        if _insert_metric_once(
            company_code="MARKET",
            domain="macro_fx",
            metric_name="usd_krw",
            metric_value=value,
            unit="KRW/USD",
            observed_at=obs,
            published_at=obs,
            valid_from=obs,
            source_tier=1,
            confidence=0.98,
            prov_entity=f"ecos:{stat_code}:{item_code}:{time_value}",
            source_url="https://ecos.bok.or.kr/",
            extra={
                "source": "ecos",
                "stat_code": stat_code,
                "item_code": item_code,
                "item_name": r.get("ITEM_NAME1", ""),
            },
        ):
            inserted += 1

    logger.info("=== ECOS USD/KRW done: rows=%d inserted=%d ===", len(rows), inserted)
