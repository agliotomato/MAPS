import json
import logging
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

import httpx

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

DART_BASE = "https://opendart.fss.or.kr/api"
REPORT_CODES = ["11013", "11012", "11014", "11011"]  # Q1, H1, Q3, Annual

METRIC_KEYWORDS = {
    "revenue": ["매출액", "매출", "수익(매출액)", "Revenue"],
    "operating_income": ["영업이익", "Operating income"],
    "gross_profit": ["매출총이익", "Gross profit"],
    "inventory": ["재고자산", "Inventories"],
    "r_and_d_expense": ["연구개발비", "R&D", "Research and development"],
    "capex_tangible": ["유형자산의 취득", "유형자산취득", "Acquisition of tangible assets"],
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _dart_key() -> str:
    return os.getenv("OPEN_DART_API_KEY", "").strip()


def _lookback_days() -> int:
    try:
        return int(os.getenv("DART_LOOKBACK_DAYS", "1"))
    except Exception:
        return 1


def _resolve_corp_codes(api_key: str, stock_codes: list[str]) -> dict[str, str]:
    timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))
    with httpx.Client(timeout=timeout) as client:
        res = client.get(f"{DART_BASE}/corpCode.xml", params={"crtfc_key": api_key})
        res.raise_for_status()
        blob = res.content

    xml_bytes: bytes | None = None
    try:
        with zipfile.ZipFile(BytesIO(blob)) as zf:
            xml_name = next((n for n in zf.namelist() if n.lower().endswith(".xml")), None)
            if xml_name:
                xml_bytes = zf.read(xml_name)
    except zipfile.BadZipFile:
        xml_bytes = blob

    if not xml_bytes:
        logger.warning("Open DART corpCode response has no XML payload.")
        return {}

    if b"<status>" in xml_bytes and b"<list>" not in xml_bytes:
        try:
            root_err = ET.fromstring(xml_bytes)
            status = (root_err.findtext(".//status") or "").strip()
            message = (root_err.findtext(".//message") or "").strip()
            logger.warning("Open DART corpCode status=%s message=%s", status, message)
        except Exception:
            logger.warning("Open DART corpCode returned non-list XML payload.")
        return {}

    root = ET.fromstring(xml_bytes)
    wanted = {x.strip() for x in stock_codes if x.strip()}
    found: dict[str, str] = {}
    for item in root.findall(".//list"):
        stock_code = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if stock_code in wanted and corp_code:
            found[stock_code] = corp_code
    return found


def _load_targets(api_key: str) -> dict[str, str]:
    raw = os.getenv("DART_TARGETS_JSON", "").strip()
    if raw:
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                return {}
            return {str(k): str(v) for k, v in data.items()}
        except Exception:
            return {}

    stock_codes_raw = os.getenv("DART_STOCK_CODES", "005930,000660,042700")
    stock_codes = [x.strip() for x in stock_codes_raw.split(",") if x.strip()]
    if not stock_codes:
        return {}
    return _resolve_corp_codes(api_key, stock_codes)


def _request_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    timeout = float(os.getenv("HTTP_TIMEOUT_SEC", "20"))
    with httpx.Client(timeout=timeout) as client:
        res = client.get(f"{DART_BASE}/{path}", params=params)
        res.raise_for_status()
        return res.json()


def _normalize_amount(raw: Any) -> float | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s in {"-", "null", "None"}:
        return None
    neg = s.startswith("(") and s.endswith(")")
    s = s.replace(",", "").replace("(", "").replace(")", "")
    s = re.sub(r"[^\d\.-]", "", s)
    if not s:
        return None
    try:
        v = float(s)
        return -v if neg else v
    except Exception:
        return None


def _parse_yyyymmdd(s: str) -> datetime:
    dt = datetime.strptime(s, "%Y%m%d")
    return dt.replace(tzinfo=timezone.utc)


def _upsert_companies(conn, targets: dict[str, str]) -> None:
    with conn.cursor() as cur:
        for stock_code in targets:
            cur.execute(
                """
                INSERT INTO companies(company_code, company_name, market, country)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (company_code) DO NOTHING
                """,
                (stock_code, stock_code, "KR", "KR"),
            )


def _upsert_disclosure(conn, stock_code: str, item: dict[str, Any]) -> None:
    rcept_no = str(item.get("rcept_no", "")).strip()
    rcept_dt = str(item.get("rcept_dt", "")).strip()
    if not rcept_no or not rcept_dt:
        return

    published_at = _parse_yyyymmdd(rcept_dt)
    report_nm = str(item.get("report_nm", "")).strip()
    corp_name = str(item.get("corp_name", "")).strip()

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO disclosures(company_code, rcept_no, report_type, title, published_at, raw_object_path, extracted)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (rcept_no)
            DO UPDATE SET
              report_type = EXCLUDED.report_type,
              title = EXCLUDED.title,
              published_at = EXCLUDED.published_at,
              extracted = EXCLUDED.extracted
            """,
            (
                stock_code,
                rcept_no,
                report_nm,
                f"{corp_name} | {report_nm}",
                published_at,
                None,
                json.dumps(item, ensure_ascii=False),
            ),
        )


def _pick_metric_name(account_nm: str) -> str | None:
    for metric_name, kws in METRIC_KEYWORDS.items():
        if any(kw.lower() in account_nm.lower() for kw in kws):
            return metric_name
    return None


def _insert_metric_observation(
    conn,
    stock_code: str,
    metric_name: str,
    metric_value: float,
    unit: str,
    published_at: datetime,
    prov_entity: str,
    extra: dict[str, Any],
) -> bool:
    account_nm = str(extra.get("account_nm", "") or "")
    thstrm_nm = str(extra.get("thstrm_nm", "") or "")
    with conn.cursor() as cur:
        # Idempotency guard:
        # Skip insert if we already stored the same DART metric row.
        cur.execute(
            """
            SELECT 1
            FROM metric_observations
            WHERE company_code = %s
              AND domain = %s
              AND metric_name = %s
              AND prov_entity = %s
              AND metric_value = %s
              AND COALESCE(extra->>'account_nm', '') = %s
              AND COALESCE(extra->>'thstrm_nm', '') = %s
            LIMIT 1
            """,
            (
                stock_code,
                "dart_financial",
                metric_name,
                prov_entity,
                metric_value,
                account_nm,
                thstrm_nm,
            ),
        )
        if cur.fetchone():
            return False

        cur.execute(
            """
            INSERT INTO metric_observations(
                company_code, domain, metric_name, metric_value, unit,
                is_proxy, proxy_for, observed_at, published_at, valid_from, valid_to,
                source_tier, confidence, prov_entity, prov_activity, prov_agent, source_url, extra
            ) VALUES (
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s::jsonb
            )
            """,
            (
                stock_code,
                "dart_financial",
                metric_name,
                metric_value,
                unit,
                False,
                None,
                published_at,
                published_at,
                published_at,
                None,
                1,
                0.95,
                prov_entity,
                "dart_collector.collect_dart_quarterly",
                "semiconductor_tech_agent",
                f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={prov_entity.split(':')[-1]}",
                json.dumps(extra, ensure_ascii=False),
            ),
        )
    return True


def collect_dart_new_filings(lookback_days: int | None = None) -> None:
    api_key = _dart_key()
    if not api_key:
        logger.warning("OPEN_DART_API_KEY is empty. Skip collect_dart_new_filings.")
        return

    targets = _load_targets(api_key)
    if not targets:
        logger.warning("DART target resolution failed. Set DART_TARGETS_JSON or DART_STOCK_CODES.")
        return

    now_kst = datetime.now()
    lb_days = _lookback_days() if lookback_days is None else int(lookback_days)
    bgn = (now_kst - timedelta(days=lb_days)).strftime("%Y%m%d")
    end = now_kst.strftime("%Y%m%d")
    logger.info("DART filings collect start | targets=%s lookback_days=%s range=%s-%s", len(targets), lb_days, bgn, end)

    with get_pg_conn() as conn:
        _upsert_companies(conn, targets)
        inserted = 0

        for stock_code, corp_code in targets.items():
            logger.info("DART filings target start | stock_code=%s corp_code=%s", stock_code, corp_code)
            page_no = 1
            while True:
                params = {
                    "crtfc_key": api_key,
                    "corp_code": corp_code,
                    "bgn_de": bgn,
                    "end_de": end,
                    "page_no": page_no,
                    "page_count": 100,
                    "last_reprt_at": "Y",
                }
                data = _request_json("list.json", params)
                status = str(data.get("status", ""))
                if status == "013":
                    logger.info("DART filings no data | stock_code=%s page=%s", stock_code, page_no)
                    break
                if status != "000":
                    logger.warning("DART list status=%s stock_code=%s msg=%s", status, stock_code, data.get("message"))
                    break

                items = data.get("list", []) or []
                if not items:
                    logger.info("DART filings empty list | stock_code=%s page=%s", stock_code, page_no)
                    break

                for item in items:
                    _upsert_disclosure(conn, stock_code, item)
                    inserted += 1

                total_count = int(data.get("total_count", 0))
                logger.info(
                    "DART filings page done | stock_code=%s page=%s items=%s total_count=%s cumulative=%s",
                    stock_code,
                    page_no,
                    len(items),
                    total_count,
                    inserted,
                )
                if page_no * 100 >= total_count:
                    break
                page_no += 1

        conn.commit()

    logger.info("collect_dart_new_filings done | upserted_disclosures=%s range=%s-%s", inserted, bgn, end)


def collect_dart_quarterly() -> None:
    api_key = _dart_key()
    if not api_key:
        logger.warning("OPEN_DART_API_KEY is empty. Skip collect_dart_quarterly.")
        return

    targets = _load_targets(api_key)
    if not targets:
        logger.warning("DART target resolution failed. Set DART_TARGETS_JSON or DART_STOCK_CODES.")
        return

    years = [datetime.now().year, datetime.now().year - 1]
    inserted = 0
    skipped_duplicates = 0
    logger.info("DART quarterly collect start | targets=%s years=%s", len(targets), years)

    with get_pg_conn() as conn:
        _upsert_companies(conn, targets)

        for stock_code, corp_code in targets.items():
            logger.info("DART quarterly target start | stock_code=%s corp_code=%s", stock_code, corp_code)
            for y in years:
                logger.info("DART quarterly year start | stock_code=%s year=%s", stock_code, y)
                for reprt_code in REPORT_CODES:
                    params = {
                        "crtfc_key": api_key,
                        "corp_code": corp_code,
                        "bsns_year": str(y),
                        "reprt_code": reprt_code,
                        "fs_div": "CFS",
                    }
                    data = _request_json("fnlttSinglAcntAll.json", params)
                    status = str(data.get("status", ""))
                    if status in {"013", "014"}:
                        logger.info("DART fnltt no data | stock_code=%s year=%s reprt_code=%s", stock_code, y, reprt_code)
                        continue
                    if status != "000":
                        logger.warning(
                            "DART fnltt status=%s stock_code=%s year=%s reprt_code=%s msg=%s",
                            status,
                            stock_code,
                            y,
                            reprt_code,
                            data.get("message"),
                        )
                        continue

                    rows = data.get("list", []) or []
                    before = inserted
                    for row in rows:
                        account_nm = str(row.get("account_nm", ""))
                        metric_name = _pick_metric_name(account_nm)
                        if not metric_name:
                            continue

                        value = _normalize_amount(row.get("thstrm_amount"))
                        if value is None:
                            continue

                        rcept_no = str(row.get("rcept_no", "")).strip()
                        if not rcept_no:
                            continue

                        observed = _utc_now()
                        prov_entity = f"dart:{y}:{stock_code}:{reprt_code}:{rcept_no}"
                        inserted_ok = _insert_metric_observation(
                            conn=conn,
                            stock_code=stock_code,
                            metric_name=metric_name,
                            metric_value=value,
                            unit=str(row.get("currency", "KRW")),
                            published_at=observed,
                            prov_entity=prov_entity,
                            extra={
                                "reprt_code": reprt_code,
                                "bsns_year": y,
                                "account_nm": account_nm,
                                "corp_code": corp_code,
                                "thstrm_nm": row.get("thstrm_nm"),
                            },
                        )
                        if inserted_ok:
                            inserted += 1
                        else:
                            skipped_duplicates += 1

                    logger.info(
                        "DART fnltt report done | stock_code=%s year=%s reprt_code=%s rows=%s inserted_delta=%s inserted_total=%s skipped_dup_total=%s",
                        stock_code,
                        y,
                        reprt_code,
                        len(rows),
                        inserted - before,
                        inserted,
                        skipped_duplicates,
                    )

        conn.commit()

    logger.info(
        "collect_dart_quarterly done | inserted_metrics=%s skipped_duplicates=%s",
        inserted,
        skipped_duplicates,
    )
