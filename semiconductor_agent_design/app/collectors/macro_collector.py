"""
Macro collectors for semiconductor-cycle features.

Targets:
- KOSIS monthly production/shipment/inventory indices when table mappings are set.
- Korea Customs public-data HS import/export statistics for semiconductor HS codes.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

DEFAULT_HS_CODES = ["8542", "8541", "8486"]
_KOSIS_SERIES_FILE = Path(__file__).resolve().parents[2] / "data" / "kosis_series.json"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_market_company() -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO companies(company_code, company_name, market, country)
                VALUES ('MARKET', 'Market-wide indicators', 'MACRO', 'GLOBAL')
                ON CONFLICT (company_code) DO NOTHING
                """
            )
        conn.commit()


def _insert_metric_once(
    *,
    domain: str,
    metric_name: str,
    metric_value: float,
    unit: str,
    observed_at: datetime,
    source_tier: int,
    confidence: float,
    prov_entity: str,
    source_url: str,
    extra: dict[str, Any],
    is_proxy: bool = False,
) -> bool:
    _ensure_market_company()
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM metric_observations WHERE prov_entity=%s LIMIT 1", (prov_entity,))
            if cur.fetchone():
                return False
            cur.execute(
                """
                INSERT INTO metric_observations(
                  company_code, domain, metric_name, metric_value, unit,
                  is_proxy, observed_at, published_at, valid_from, valid_to,
                  source_tier, confidence, prov_entity, source_url, extra
                ) VALUES (
                  'MARKET',%s,%s,%s,%s,
                  %s,%s,%s,%s,NULL,
                  %s,%s,%s,%s,%s::jsonb
                )
                """,
                (
                    domain, metric_name, metric_value, unit,
                    is_proxy, observed_at, observed_at, observed_at,
                    source_tier, confidence, prov_entity, source_url,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()
    return True


def _parse_period(period: str) -> datetime | None:
    p = str(period or "").strip().replace("-", "")
    try:
        if len(p) == 4:
            return datetime(int(p), 1, 1, tzinfo=timezone.utc)
        if len(p) == 6:
            return datetime(int(p[:4]), int(p[4:6]), 1, tzinfo=timezone.utc)
        if len(p) == 8:
            return datetime(int(p[:4]), int(p[4:6]), int(p[6:8]), tzinfo=timezone.utc)
    except Exception:
        return None
    return None


def _num(value: Any) -> float | None:
    if value is None:
        return None
    raw = str(value).replace(",", "").strip()
    if not raw or raw in {"-", "."}:
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _load_kosis_series() -> list[dict[str, Any]]:
    """
    Load KOSIS series mapping from env or data/kosis_series.json.

    Example:
    KOSIS_SERIES_JSON=[
      {
        "metric_name":"semiconductor_production_index",
        "orgId":"101",
        "tblId":"...",
        "itmId":"...",
        "objL1":"...",
        "prdSe":"M",
        "unit":"index"
      }
    ]
    """
    raw = os.getenv("KOSIS_SERIES_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Invalid KOSIS_SERIES_JSON: %s", e)
            return []

    if _KOSIS_SERIES_FILE.exists():
        try:
            data = json.loads(_KOSIS_SERIES_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("Invalid KOSIS series file %s: %s", _KOSIS_SERIES_FILE, e)
            return []

    return []


def _search_kosis_tables(api_key: str, search_nm: str, org_id: str | None = None) -> list[dict[str, Any]]:
    """Resolve a KOSIS search query into candidate table metadata."""
    params: dict[str, Any] = {
        "method": "getList",
        "apiKey": api_key,
        "searchNm": search_nm,
        "sort": "RANK",
        "startCount": "1",
        "resultCount": "10",
        "format": "json",
    }
    if org_id:
        params["orgId"] = org_id

    endpoint = "https://kosis.kr/openapi/statisticsSearch.do"
    try:
        with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20"))) as client:
            resp = client.get(endpoint, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except Exception as e:
        logger.error("KOSIS search failed searchNm=%s: %s", search_nm, e)
        return []

    if isinstance(payload, dict):
        for key in ("data", "rows", "list", "result"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _slugify_metric_component(value: Any) -> str:
    raw = str(value or "").strip().lower()
    raw = re.sub(r"[^0-9a-zA-Z가-힣]+", "_", raw)
    return raw.strip("_")


def _resolve_kosis_series(cfg: dict[str, Any], api_key: str) -> dict[str, Any] | None:
    """
    Resolve a series config either from explicit orgId/tblId or by KOSIS search.

    Supported config forms:
    - explicit: {"orgId":"101", "tblId":"DT_...", "itmId":"ALL"}
    - search-based: {"searchNm":"전자부품 컴퓨터 생산동향", "metric_name":"..."}
    """
    if cfg.get("orgId") and cfg.get("tblId"):
        return cfg

    search_nm = str(cfg.get("searchNm") or cfg.get("searchNmExact") or cfg.get("query") or "").strip()
    if not search_nm:
        return None

    candidates = _search_kosis_tables(api_key, search_nm, str(cfg.get("orgId") or "") or None)
    if not candidates:
        return None

    wanted_tbl = _normalize_text(cfg.get("tblNm") or cfg.get("tblName") or cfg.get("table_name") or "")
    wanted_query = _normalize_text(search_nm)
    wanted_stat = _normalize_text(cfg.get("statId") or cfg.get("stat_id") or "")

    def score(row: dict[str, Any]) -> tuple[int, int]:
        tbl_nm = _normalize_text(row.get("TBL_NM"))
        query = _normalize_text(row.get("QUERY"))
        stat_id = _normalize_text(row.get("STAT_ID"))
        exact = 0
        if wanted_tbl and tbl_nm == wanted_tbl:
            exact += 8
        if wanted_tbl and wanted_tbl in tbl_nm:
            exact += 4
        if wanted_query and (wanted_query in tbl_nm or wanted_query in query):
            exact += 3
        if wanted_stat and stat_id == wanted_stat:
            exact += 2
        if str(row.get("REC_TBL_SE") or "").upper() == "Y":
            exact += 1
        return exact, len(tbl_nm)

    best = sorted(candidates, key=score, reverse=True)[0]
    resolved = dict(cfg)
    resolved["orgId"] = best.get("ORG_ID") or cfg.get("orgId")
    resolved["tblId"] = best.get("TBL_ID") or cfg.get("tblId")
    resolved["tblNm"] = best.get("TBL_NM") or cfg.get("tblNm") or search_nm
    if best.get("STAT_ID"):
        resolved["statId"] = best.get("STAT_ID")
    if best.get("VW_CD"):
        resolved["vwCd"] = best.get("VW_CD")
    return resolved if resolved.get("orgId") and resolved.get("tblId") else None


def collect_kosis_stats() -> None:
    """KOSIS configured monthly macro/industry statistics."""
    api_key = os.getenv("KOSIS_API_KEY", "").strip()
    if not api_key:
        logger.warning("KOSIS_API_KEY is empty. Skip collect_kosis_stats.")
        return

    series = _load_kosis_series()
    if not series:
        logger.warning(
            "KOSIS_SERIES_JSON is empty. Add table mappings before collecting KOSIS stats."
        )
        return

    months = int(os.getenv("KOSIS_LOOKBACK_MONTHS", "60"))
    end = _now_utc()
    start = end - timedelta(days=months * 31)
    start_prd = start.strftime("%Y%m")
    end_prd = end.strftime("%Y%m")

    endpoint = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
    inserted = 0
    for cfg in series:
        resolved = _resolve_kosis_series(cfg, api_key)
        if not resolved:
            logger.warning("KOSIS series could not be resolved: %s", cfg.get("metric_name") or cfg.get("searchNm"))
            continue
        params = {
            "method": "getList",
            "apiKey": api_key,
            "format": "json",
            "jsonVD": "Y",
            "prdSe": resolved.get("prdSe", "M"),
            "startPrdDe": resolved.get("startPrdDe", start_prd),
            "endPrdDe": resolved.get("endPrdDe", end_prd),
            "orgId": resolved["orgId"],
            "tblId": resolved["tblId"],
            "itmId": resolved.get("itmId", "ALL"),
            "objL1": resolved.get("objL1", "ALL"),
            "objL2": resolved.get("objL2", ""),
            "objL3": resolved.get("objL3", ""),
            "objL4": resolved.get("objL4", ""),
            "objL5": resolved.get("objL5", ""),
            "objL6": resolved.get("objL6", ""),
            "objL7": resolved.get("objL7", ""),
            "objL8": resolved.get("objL8", ""),
        }
        try:
            with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20"))) as client:
                resp = client.get(endpoint, params=params)
                resp.raise_for_status()
                rows = resp.json()
        except Exception as e:
            logger.error(
                "KOSIS fetch failed metric=%s tbl=%s: %s",
                resolved.get("metric_name"),
                resolved.get("tblNm"),
                e,
            )
            continue

        if isinstance(rows, dict):
            rows = rows.get("data") or rows.get("rows") or []
        for row in rows if isinstance(rows, list) else []:
            obs = _parse_period(row.get("PRD_DE") or row.get("PRD_DE_NM"))
            value = _num(row.get("DT"))
            if obs is None or value is None:
                continue
            metric_name = resolved["metric_name"]
            item_id = str(row.get("ITM_ID") or row.get("ITEM_ID") or row.get("ITM_NM") or row.get("ITEM_NM") or "").strip()
            if item_id and item_id.upper() != "ALL":
                metric_name = f"{metric_name}_{_slugify_metric_component(item_id)}"
            prov = f"kosis:{resolved['orgId']}:{resolved['tblId']}:{metric_name}:{obs.strftime('%Y%m')}"
            if _insert_metric_once(
                domain=resolved.get("domain", "macro_industry"),
                metric_name=metric_name,
                metric_value=value,
                unit=resolved.get("unit", row.get("UNIT_NM", "")),
                observed_at=obs,
                source_tier=1,
                confidence=0.95,
                prov_entity=prov,
                source_url="https://kosis.kr/",
                extra={"source": "kosis", "config": {k: v for k, v in resolved.items() if k != "apiKey"}},
            ):
                inserted += 1

    logger.info("collect_kosis_stats done | inserted=%d", inserted)


def _customs_endpoint() -> str:
    return os.getenv(
        "CUSTOMS_TRADE_ENDPOINT",
        "https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList",
    )


def _extract_customs_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        body = ((payload.get("response") or {}).get("body") or {})
        items = body.get("items") or payload.get("items") or payload.get("item") or []
        if isinstance(items, dict):
            items = items.get("item") or []
        if isinstance(items, dict):
            return [items]
        if isinstance(items, list):
            return items
    return []


def collect_customs_trade() -> None:
    """Korea Customs HS import/export stats for semiconductor HS codes."""
    api_key = os.getenv("CUSTOMS_API_KEY", os.getenv("PUBLIC_DATA_API_KEY", "")).strip()
    if not api_key:
        logger.warning("CUSTOMS_API_KEY/PUBLIC_DATA_API_KEY is empty. Skip collect_customs_trade.")
        return

    hs_codes = [x.strip() for x in os.getenv("CUSTOMS_HS_CODES", ",".join(DEFAULT_HS_CODES)).split(",") if x.strip()]
    months = int(os.getenv("CUSTOMS_LOOKBACK_MONTHS", "60"))
    end = _now_utc()
    start = end - timedelta(days=months * 31)
    start_ym = start.strftime("%Y%m")
    end_ym = end.strftime("%Y%m")
    endpoint = _customs_endpoint()

    inserted = 0
    for hs_code in hs_codes:
        params = {
            "serviceKey": api_key,
            "pageNo": "1",
            "numOfRows": "1000",
            "type": "json",
            "strtYymm": start_ym,
            "endYymm": end_ym,
            "hsSgn": hs_code,
        }
        try:
            with httpx.Client(timeout=float(os.getenv("HTTP_TIMEOUT_SEC", "20")), follow_redirects=True) as client:
                resp = client.get(endpoint, params=params)
                resp.raise_for_status()
                try:
                    payload = resp.json()
                except Exception:
                    logger.warning("Customs endpoint did not return JSON for hs=%s", hs_code)
                    continue
        except Exception as e:
            logger.error("Customs fetch failed hs=%s endpoint=%s: %s", hs_code, endpoint, e)
            continue

        for row in _extract_customs_rows(payload):
            period = row.get("year") or row.get("yymm") or row.get("baseYymm") or row.get("balPaymentsYear")
            obs = _parse_period(str(period))
            if obs is None:
                continue

            export_value = _num(
                row.get("expDlr")
                or row.get("expUsdAmt")
                or row.get("exportAmount")
                or row.get("exptUsdAmt")
            )
            import_value = _num(
                row.get("impDlr")
                or row.get("impUsdAmt")
                or row.get("importAmount")
                or row.get("imptUsdAmt")
            )
            balance = _num(row.get("balPayments") or row.get("tradeBalance"))

            metrics = [
                (f"export_value_usd_hs{hs_code}", export_value),
                (f"import_value_usd_hs{hs_code}", import_value),
                (f"trade_balance_usd_hs{hs_code}", balance),
            ]
            if balance is None and export_value is not None and import_value is not None:
                metrics[2] = (metrics[2][0], export_value - import_value)

            for metric_name, value in metrics:
                if value is None:
                    continue
                prov = f"customs:{hs_code}:{metric_name}:{obs.strftime('%Y%m')}"
                if _insert_metric_once(
                    domain="customs_trade",
                    metric_name=metric_name,
                    metric_value=value,
                    unit="USD",
                    observed_at=obs,
                    source_tier=1,
                    confidence=0.92,
                    prov_entity=prov,
                    source_url="https://www.data.go.kr/",
                    extra={"source": "korea_customs_public_data", "hs_code": hs_code, "raw": row},
                ):
                    inserted += 1

    logger.info("collect_customs_trade done | hs_codes=%s inserted=%d", hs_codes, inserted)
