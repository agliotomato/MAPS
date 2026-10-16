"""
TimescaleDB Repository — 거시경제 시계열 CRUD + ECOS/FRED 데이터 정규화

핵심 함수:
    insert_macro_data()          — Batch Upsert (중복 방지)
    normalize_ecos_to_records()  — ECOS Tool 출력 → MetricRecord 변환
    normalize_fred_to_records()  — FRED Tool 출력 → MetricRecord 변환
    query_metric()               — 단일 지표 시계열 조회
    query_latest_snapshot()      — 모든 지표 최신값 스냅샷 조회
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import execute_values

from macro_agent.db.timescale.client import get_connection

logger = logging.getLogger(__name__)


# ── 도메인 모델 ────────────────────────────────────────────────────────

@dataclass
class MetricRecord:
    """macro_metrics 테이블의 단일 행을 표현하는 값 객체"""
    time: datetime          # UTC 타임스탬프
    metric_name: str        # 'ecos.usd_krw', 'fred.ism_pmi' 등
    value: float
    source: str             # 'ecos' | 'fred' | 'market'
    region: str | None = None
    unit: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class UpsertStats:
    """insert_macro_data 실행 결과 요약"""
    submitted: int = 0
    changed: int = 0         # 실제 INSERT 또는 값이 바뀐 UPDATE 수
    skipped: int = 0         # 동일 값 존재로 스킵된 수

    @property
    def unchanged(self) -> int:
        return self.submitted - self.changed


# ── SQL 상수 ───────────────────────────────────────────────────────────

_UPSERT_SQL = """
INSERT INTO macro_metrics (time, metric_name, value, source, region, unit, metadata)
VALUES %s
ON CONFLICT (time, metric_name, source)
DO UPDATE SET
    value      = EXCLUDED.value,
    unit       = EXCLUDED.unit,
    metadata   = EXCLUDED.metadata,
    updated_at = NOW()
WHERE
    macro_metrics.value    IS DISTINCT FROM EXCLUDED.value
 OR macro_metrics.metadata IS DISTINCT FROM EXCLUDED.metadata
"""

_QUERY_METRIC_SQL = """
SELECT time, metric_name, value, source, region, unit, metadata
FROM   macro_metrics
WHERE  metric_name = %(name)s
  AND  time >= %(start)s
  AND  (%(end)s IS NULL OR time <= %(end)s)
ORDER BY time DESC
LIMIT %(limit)s
"""

_QUERY_LATEST_SQL = """
SELECT DISTINCT ON (metric_name)
    time, metric_name, value, source, region, unit
FROM macro_metrics
ORDER BY metric_name, time DESC
"""

# ── 시계열 특성 피처 쿼리 (CTE 5단계) ────────────────────────────────
#
# 설계 원칙:
#   - INTERVAL '1 month' * N  : 정수 파라미터를 안전하게 INTERVAL로 변환
#   - DISTINCT ON + ABS(EPOCH) : 일봉/월봉 혼합 데이터에서 목표 시점 최근접 레코드 선택
#   - FILTER (WHERE time >= …) : 동일 GROUP BY 안에서 여러 날짜 범위를 한 번에 집계
#   - LEFT JOIN (모든 CTE)     : 데이터 부족 지표도 NULL로 포함 (Graceful Degradation)
#
_MACRO_TS_FEATURES_SQL = """
WITH
-- ① 각 지표의 최신 레코드 (기준점)
latest AS (
    SELECT DISTINCT ON (metric_name)
        metric_name,
        time    AS latest_time,
        value   AS current_value,
        source,
        region,
        unit
    FROM macro_metrics
    WHERE time >= %(start_date)s
    ORDER BY metric_name, time DESC
),

-- ② 조회 기간 내 전체 이력
history AS (
    SELECT metric_name, time, value
    FROM macro_metrics
    WHERE time >= %(start_date)s
),

-- ③ MoM 기준값
--    탐색 범위 20~45일: 일봉(당일 제외 가장 가까운 영업일)과
--    월봉(직전 달 말) 모두를 커버하는 비대칭 윈도우
mom_ref AS (
    SELECT DISTINCT ON (h.metric_name)
        h.metric_name,
        h.value AS prev_value_mom
    FROM history h
    JOIN latest l USING (metric_name)
    WHERE h.time BETWEEN l.latest_time - INTERVAL '45 days'
                     AND l.latest_time - INTERVAL '20 days'
    ORDER BY
        h.metric_name,
        ABS(EXTRACT(EPOCH FROM (h.time - (l.latest_time - INTERVAL '30 days'))))
),

-- ④ YoY 기준값
--    탐색 범위 10~14개월: 월봉은 정확히 12개월 전, 일봉은 ±60일 허용
yoy_ref AS (
    SELECT DISTINCT ON (h.metric_name)
        h.metric_name,
        h.value AS prev_value_yoy
    FROM history h
    JOIN latest l USING (metric_name)
    WHERE h.time BETWEEN l.latest_time - INTERVAL '14 months'
                     AND l.latest_time - INTERVAL '10 months'
    ORDER BY
        h.metric_name,
        ABS(EXTRACT(EPOCH FROM (h.time - (l.latest_time - INTERVAL '12 months'))))
),

-- ⑤ 날짜 범위 기반 집계 (이동평균·변동성·역사적 백분위)
--    FILTER(WHERE …) 로 날짜 범위별 집계를 단일 GROUP BY에서 처리
agg AS (
    SELECT
        h.metric_name,

        -- 30일 이동평균
        --   일봉 ≈ 약 22 거래일 평균  |  월봉 ≈ 최신 1개 레코드 평균
        AVG(h.value) FILTER (WHERE h.time >= l.latest_time - INTERVAL '30 days')
            AS ma_30,

        -- 90일 이동평균
        --   일봉 ≈ 약 65 거래일 평균  |  월봉 ≈ 최근 3개월 평균
        AVG(h.value) FILTER (WHERE h.time >= l.latest_time - INTERVAL '90 days')
            AS ma_90,

        -- 3개월 표준편차 (변동성 지표)
        --   샘플 수 < 2이면 STDDEV = NULL (미상 유지)
        STDDEV(h.value) FILTER (WHERE h.time >= l.latest_time - INTERVAL '90 days')
            AS volatility_3m,

        -- 3개월 내 샘플 수 (데이터 신뢰도 지표)
        COUNT(*) FILTER (WHERE h.time >= l.latest_time - INTERVAL '90 days')
            AS sample_count_3m,

        -- 역사적 백분위: 전체 이력에서 현재값 이하인 레코드 비율 (0~100)
        ROUND(
            CAST(COUNT(*) FILTER (WHERE h.value <= l.current_value) AS numeric)
            / NULLIF(COUNT(*), 0) * 100,
            1
        ) AS historical_pct,

        COUNT(*) AS total_samples
    FROM history h
    JOIN latest l USING (metric_name)
    GROUP BY h.metric_name
)

SELECT
    l.metric_name,
    l.latest_time,
    l.current_value,
    l.source,
    l.region,
    l.unit,

    -- 이동평균 (NULL-safe ROUND)
    ROUND(CAST(a.ma_30 AS numeric), 4)  AS ma_30,
    ROUND(CAST(a.ma_90 AS numeric), 4)  AS ma_90,

    -- 변동성: 샘플 1개일 때 STDDEV=NULL 유지
    ROUND(CAST(a.volatility_3m AS numeric), 4) AS volatility_3m,

    a.sample_count_3m,
    a.historical_pct,
    a.total_samples,

    -- MoM 변동률 (percent)
    CASE
        WHEN m.prev_value_mom IS NOT NULL AND m.prev_value_mom <> 0
        THEN ROUND(
            CAST((l.current_value - m.prev_value_mom) / ABS(m.prev_value_mom) * 100 AS numeric),
            2
        )
        ELSE NULL
    END AS mom_change,

    -- YoY 변동률 (percent)
    CASE
        WHEN y.prev_value_yoy IS NOT NULL AND y.prev_value_yoy <> 0
        THEN ROUND(
            CAST((l.current_value - y.prev_value_yoy) / ABS(y.prev_value_yoy) * 100 AS numeric),
            2
        )
        ELSE NULL
    END AS yoy_change

FROM       latest   l
LEFT JOIN  mom_ref  m USING (metric_name)
LEFT JOIN  yoy_ref  y USING (metric_name)
LEFT JOIN  agg      a USING (metric_name)
ORDER BY   l.metric_name
"""


# ── 내부 헬퍼 ─────────────────────────────────────────────────────────

def _record_to_tuple(r: MetricRecord) -> tuple:
    return (
        r.time,
        r.metric_name,
        r.value,
        r.source,
        r.region,
        r.unit,
        json.dumps(r.metadata, ensure_ascii=False) if r.metadata else None,
    )


def _parse_ecos_date(date_str: str) -> datetime | None:
    """ECOS 날짜 'YYYYMMDD'(일별) 또는 'YYYYMM'(월별) → datetime (UTC)"""
    s = date_str.strip()
    for fmt in ("%Y%m%d", "%Y%m"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.debug("ECOS 날짜 파싱 실패: %r", date_str)
    return None


def _parse_fred_date(date_str: str) -> datetime | None:
    """FRED 날짜 'YYYY-MM-DD' 또는 'YYYY-MM' → datetime (UTC)"""
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.debug("FRED 날짜 파싱 실패: %r", date_str)
    return None


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if (f != f) else f  # NaN 제거
    except (TypeError, ValueError):
        return None


# ── 메인 Upsert 함수 ──────────────────────────────────────────────────

def insert_macro_data(records: list[MetricRecord]) -> UpsertStats:
    """
    MetricRecord 목록을 macro_metrics에 배치 Upsert합니다.

    - 동일 (time, metric_name, source)가 이미 존재하면서 value·metadata가
      변경되지 않았다면 UPDATE를 수행하지 않아 불필요한 WAL 기록을 방지합니다.
    - psycopg2 execute_values로 page_size=500 단위 배치 전송합니다.

    Args:
        records: 삽입할 MetricRecord 목록

    Returns:
        UpsertStats(submitted, changed, skipped)
    """
    if not records:
        logger.debug("insert_macro_data: 입력 레코드 없음, 스킵")
        return UpsertStats()

    # 같은 배치 내 (time, metric_name, source) 중복 제거 — 마지막 값 우선
    seen: dict[tuple, MetricRecord] = {}
    for r in records:
        seen[(r.time, r.metric_name, r.source)] = r
    if len(seen) < len(records):
        logger.debug("중복 레코드 %d건 제거 (%d → %d)", len(records) - len(seen), len(records), len(seen))
    records = list(seen.values())

    tuples = [_record_to_tuple(r) for r in records]

    with get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(
                cur,
                _UPSERT_SQL,
                tuples,
                template="(%s, %s, %s, %s, %s, %s, %s::jsonb)",
                page_size=500,
            )
            changed = cur.rowcount if cur.rowcount >= 0 else 0

    stats = UpsertStats(
        submitted=len(records),
        changed=changed,
        skipped=len(records) - changed,
    )
    logger.info(
        "TimescaleDB upsert 완료 — 제출: %d, 변경: %d, 스킵(동일값): %d",
        stats.submitted,
        stats.changed,
        stats.skipped,
    )
    return stats


# ── ECOS 데이터 정규화 ────────────────────────────────────────────────

def normalize_ecos_to_records(ecos_data: dict) -> list[MetricRecord]:
    """
    fetch_ecos_data() 반환 dict를 MetricRecord 목록으로 변환합니다.

    처리 지표:
        ecos.usd_krw                    — 원/달러 환율
        ecos.semiconductor_export_value — 반도체 수출액
        ecos.semiconductor_shipment_index — 출하지수
        ecos.semiconductor_inventory_index — 재고지수
    """
    records: list[MetricRecord] = []
    source = "ecos"

    def _series(key: str, sub_key: str = "series") -> list[dict]:
        block = ecos_data.get(key, {})
        if isinstance(block, dict):
            return block.get(sub_key, [])
        return []

    # 1. 원/달러 환율
    for item in _series("usd_krw"):
        dt = _parse_ecos_date(item.get("date", ""))
        val = _safe_float(item.get("krw_per_usd"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="ecos.usd_krw", value=val,
                source=source, region="KR", unit="KRW/USD",
            ))

    # 2. 전자부품 제조업 출하지수 (production_shipment_inventory.shipment.series)
    psi = ecos_data.get("production_shipment_inventory", {})
    for item in psi.get("shipment", {}).get("series", []):
        dt = _parse_ecos_date(item.get("date", ""))
        val = _safe_float(item.get("index"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="ecos.electronics_shipment_index", value=val,
                source=source, region="KR", unit="INDEX",
            ))

    # 3. 전자부품 제조업 재고지수 (production_shipment_inventory.inventory.series)
    for item in psi.get("inventory", {}).get("series", []):
        dt = _parse_ecos_date(item.get("date", ""))
        val = _safe_float(item.get("index"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="ecos.electronics_inventory_index", value=val,
                source=source, region="KR", unit="INDEX",
            ))

    logger.info("ECOS 정규화 완료: %d건", len(records))
    return records


# ── FRED 데이터 정규화 ────────────────────────────────────────────────

def normalize_fred_to_records(fred_data: dict) -> list[MetricRecord]:
    """
    fetch_fred_data() 반환 dict를 MetricRecord 목록으로 변환합니다.

    처리 지표:
        market.sox_close   — SOX 반도체지수 월봉 종가 (yfinance)
        fred.ism_pmi       — ISM 제조업 PMI (FRED MPMANSICS)
        fred.fed_funds_rate — 연방기금 실효금리 (FRED FEDFUNDS)
    """
    records: list[MetricRecord] = []

    def _series(key: str) -> list[dict]:
        block = fred_data.get(key, {})
        return block.get("series", []) if isinstance(block, dict) else []

    # 1. SOX 반도체지수 (yfinance — source='market')
    for item in _series("sox_index"):
        dt = _parse_fred_date(item.get("date", ""))
        val = _safe_float(item.get("close"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="market.sox_close", value=val,
                source="market", region="US", unit="USD",
                metadata={
                    "high": item.get("high"),
                    "low": item.get("low"),
                    "volume": item.get("volume"),
                },
            ))

    # 2. 제조업 산업생산지수 (FRED IPMAN)
    for item in _series("manufacturing_ip"):
        dt = _parse_fred_date(item.get("date", ""))
        val = _safe_float(item.get("value"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="fred.manufacturing_ip", value=val,
                source="fred", region="US", unit="INDEX",
            ))

    # 3. 연방기금 실효금리 (FRED)
    for item in _series("federal_funds_rate"):
        dt = _parse_fred_date(item.get("date", ""))
        val = _safe_float(item.get("value"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="fred.fed_funds_rate", value=val,
                source="fred", region="US", unit="%",
            ))

    # 4. GSCPI (글로벌 공급망 압력 지수)
    for item in _series("gscpi"):
        dt = _parse_fred_date(item.get("date", ""))
        val = _safe_float(item.get("value"))
        if dt and val is not None:
            records.append(MetricRecord(
                time=dt, metric_name="fred.gscpi", value=val,
                source="fred", region="GLOBAL", unit="σ",
                metadata={"interpretation": item.get("interpretation")},
            ))

    logger.info("FRED 정규화 완료: %d건", len(records))
    return records


# ── 관세청 데이터 정규화 ──────────────────────────────────────────────

def normalize_customs_to_records(customs_data: dict) -> list[MetricRecord]:
    """
    fetch_customs_20day_data() 또는 fetch_customs_monthly_data() 반환 dict를
    MetricRecord 목록으로 변환합니다.

    처리 지표:
        data_type="20day":
            customs.export_20d_usd      — 20일 수출금액 (백만달러)
            customs.import_20d_usd      — 20일 수입금액 (백만달러)
            customs.trade_balance_20d   — 20일 무역수지 (백만달러)
            customs.export_20d_yoy      — 20일 수출 증감률 (%)
            customs.import_20d_yoy      — 20일 수입 증감률 (%)
        data_type="monthly":
            customs.export_monthly_usd      — 월별 수출금액 (백만달러)
            customs.import_monthly_usd      — 월별 수입금액 (백만달러)
            customs.trade_balance_monthly   — 월별 무역수지 (백만달러)
            customs.export_monthly_yoy      — 월별 수출 증감률 (%)
            customs.import_monthly_yoy      — 월별 수입 증감률 (%)

    HS코드 필터링:
        hs_cd가 빈 문자열("")이면 전체 품목 합계.
        hs_cd="8542"이면 metric_name에 ".semi" 접미사 추가
        (예: customs.export_monthly_usd.semi)
    """
    records: list[MetricRecord] = []
    source    = "customs"
    data_type = customs_data.get("data_type", "monthly")
    items     = customs_data.get("items", [])

    if not items:
        logger.debug("normalize_customs_to_records: items 없음 (data_type=%s)", data_type)
        return records

    # 지표명 접미사 결정 (20일 vs 월별)
    if data_type == "20day":
        suffix_export  = "export_20d_usd"
        suffix_import  = "import_20d_usd"
        suffix_balance = "trade_balance_20d"
        suffix_exp_yoy = "export_20d_yoy"
        suffix_imp_yoy = "import_20d_yoy"
    else:
        suffix_export  = "export_monthly_usd"
        suffix_import  = "import_monthly_usd"
        suffix_balance = "trade_balance_monthly"
        suffix_exp_yoy = "export_monthly_yoy"
        suffix_imp_yoy = "import_monthly_yoy"

    for item in items:
        dt = item.get("dt")
        if dt is None:
            continue
        # dt가 naive datetime이면 UTC 부착
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # HS코드 접미사: 전체="" → 접미사 없음, 반도체="8542" → ".semi"
        hs = item.get("hs_cd", "")
        hs_suffix = ".semi" if hs == "8542" else (".hs" + hs if hs else "")
        cnty = item.get("cnty_cd", "")
        cnty_suffix = ("." + cnty.lower()) if cnty else ""
        item_suffix = hs_suffix + cnty_suffix   # 예: ".semi.cn", ""

        meta: dict[str, Any] = {
            "hs_cd":   hs,
            "cnty_cd": cnty,
        }

        def _add(metric_suffix: str, value: float | None, unit: str) -> None:
            if value is None:
                return
            records.append(MetricRecord(
                time=dt,
                metric_name=f"customs.{metric_suffix}{item_suffix}",
                value=value,
                source=source,
                region="KR",
                unit=unit,
                metadata=meta,
            ))

        _add(suffix_export,  item.get("export_usd"),  "USD_M")  # 백만달러
        _add(suffix_import,  item.get("import_usd"),  "USD_M")
        _add(suffix_balance, item.get("balance"),     "USD_M")
        _add(suffix_exp_yoy, item.get("export_yoy"),  "%")
        _add(suffix_imp_yoy, item.get("import_yoy"),  "%")

    logger.info(
        "관세청 데이터 정규화 완료: %d건 → %d MetricRecord (data_type=%s)",
        len(items), len(records), data_type,
    )
    return records


# ── 조회 함수 ──────────────────────────────────────────────────────────

def query_metric(
    metric_name: str,
    start_time: datetime,
    end_time: datetime | None = None,
    limit: int = 500,
) -> list[dict]:
    """
    특정 지표의 시계열 데이터를 최신순으로 조회합니다.

    Args:
        metric_name: 지표 코드 (예: 'ecos.usd_krw')
        start_time:  조회 시작 시점 (UTC datetime)
        end_time:    조회 종료 시점 (None이면 현재까지)
        limit:       최대 반환 행 수

    Returns:
        [{"time": datetime, "metric_name": str, "value": float, ...}]
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_QUERY_METRIC_SQL, {
                "name": metric_name,
                "start": start_time,
                "end": end_time,
                "limit": limit,
            })
            return [dict(row) for row in cur.fetchall()]


def query_enriched_snapshot(months_back: int = 36) -> dict[str, dict]:
    """Legacy fields backed by date windows, never by daily row counts.

    Missing monthly comparisons stay unknown. history_samples counts
    observations; it does not claim that each observation is a month.
    """
    return {name: {**row, "metric_name": name, "time": row["as_of"],
                   "value": row["current_value"], "mom_change_pct": row["mom_change"],
                   "historical_percentile": row["historical_pct"],
                   "history_samples": row["total_samples"]}
            for name, row in get_macro_ts_features(months_back).items()}


def query_latest_snapshot() -> dict[str, dict]:
    """
    모든 지표의 가장 최신 데이터 포인트를 지표명 기준 dict로 반환합니다.
    에이전트가 현재 시장 상태를 빠르게 파악하는 데 사용합니다.

    Returns:
        {"ecos.usd_krw": {"time": ..., "value": 1380.0, ...}, ...}
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_QUERY_LATEST_SQL)
            rows = cur.fetchall()

    return {row["metric_name"]: dict(row) for row in rows}


# ── 시계열 특성 피처 ─────────────────────────────────────────────────

def _make_trend_label(
    mom_change: float | None,
    current_value: float | None,
    ma_90: float | None,
) -> str:
    """
    MoM 변동률(1차 신호)과 현재값 vs MA90 위치(2차 신호)를 조합해
    LLM 프롬프트에 바로 삽입할 수 있는 추세 레이블을 반환합니다.

    반환 예시:
        "상승 (MA90↑)"   — MoM 양수이고 현재값이 90일 평균 위
        "하락 (MA90↓)"   — MoM 음수이고 현재값이 90일 평균 아래
        "보합 (MA90≈)"   — 변화 미미
        "N/A"            — MoM 계산 불가 (데이터 부족)
    """
    if mom_change is None:
        return "N/A"

    # 1차 신호: MoM ±1% 기준
    if mom_change > 1.0:
        primary = "상승"
    elif mom_change < -1.0:
        primary = "하락"
    else:
        primary = "보합"

    # 2차 신호: 현재값이 MA90 대비 ±0.5% 이상 이탈하는지
    if current_value is not None and ma_90 is not None and ma_90 != 0:
        ratio = (current_value - ma_90) / abs(ma_90)
        if ratio > 0.005:
            secondary = "MA90↑"
        elif ratio < -0.005:
            secondary = "MA90↓"
        else:
            secondary = "MA90≈"
        return f"{primary} ({secondary})"

    return primary


def get_macro_ts_features(months_back: int = 36) -> dict[str, dict[str, Any]]:
    """
    TimescaleDB의 3년치 시계열 데이터에서 지표별 추세·모멘텀 피처를 계산합니다.

    단일 CTE SQL로 아래 6가지 통계를 한 번의 DB 왕복으로 산출합니다:
        current_value   최신 수치
        ma_30           최근 30일 이동평균  (일봉 ≈ 22거래일, 월봉 ≈ 1개월)
        ma_90           최근 90일 이동평균  (일봉 ≈ 65거래일, 월봉 ≈ 3개월)
        mom_change      전월 대비 변동률 (%, 20~45일 전 최근접 레코드 기준)
        yoy_change      전년 동기 대비 변동률 (%, 10~14개월 전 최근접 레코드 기준)
        volatility_3m   최근 3개월 표준편차 (샘플 2개 미만이면 None)

    추가 파생 필드:
        trend_label     MoM + MA90 위치를 조합한 추세 레이블 (예: "상승 (MA90↑)")
        historical_pct  전체 이력 내 현재값의 백분위 (0~100)
        sample_count_3m 3개월 내 샘플 수 (데이터 신뢰도 지표)
        total_samples   전체 이력 샘플 수

    Graceful Degradation:
        - 데이터 부족으로 계산 불가한 항목은 None
        - 특정 지표 처리 실패 시 해당 지표만 스킵, 나머지 지표는 정상 반환
        - DB 연결/쿼리 실패는 호출자에게 전달해 정상 빈 결과와 구분

    Args:
        months_back: 조회할 이력 기간 (월 단위, 기본값: 36)

    Returns:
        {
            "ecos.usd_krw": {
                "current_value":   1380.5,
                "unit":            "KRW/USD",
                "region":          "KR",
                "source":          "ecos",
                "as_of":           "2026-05-30",
                "ma_30":           1375.2,
                "ma_90":           1370.8,
                "mom_change":      0.72,      # % (None이면 기준값 없음)
                "yoy_change":      -2.15,     # % (None이면 1년치 미보유)
                "volatility_3m":   12.3,      # 표준편차 (None이면 샘플 부족)
                "sample_count_3m": 22,
                "historical_pct":  68.0,      # 0~100
                "total_samples":   756,
                "trend_label":     "상승 (MA90↑)",
            },
            ...
        }
    """
    # ── DB 조회 ───────────────────────────────────────────────────────
    try:
        from datetime import datetime, timezone, timedelta
        start_date = datetime.now(timezone.utc) - timedelta(days=30 * months_back)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_MACRO_TS_FEATURES_SQL, {"start_date": start_date})
                rows = [dict(r) for r in cur.fetchall()]
    except Exception as exc:
        logger.error(
            "get_macro_ts_features DB 조회 실패 (months_back=%d): %s",
            months_back, exc, exc_info=True,
        )
        raise

    # ── 행별 파싱 (지표 단위 예외 격리) ──────────────────────────────
    result: dict[str, dict[str, Any]] = {}

    for row in rows:
        metric_name: str = row.get("metric_name") or ""
        if not metric_name:
            continue

        try:
            current_value = _safe_float(row.get("current_value"))
            ma_30         = _safe_float(row.get("ma_30"))
            ma_90         = _safe_float(row.get("ma_90"))
            mom_change    = _safe_float(row.get("mom_change"))
            yoy_change    = _safe_float(row.get("yoy_change"))
            volatility_3m = _safe_float(row.get("volatility_3m"))

            result[metric_name] = {
                # ── 기본 정보 ────────────────────────────────────────
                "current_value":   current_value,
                "unit":            str(row.get("unit")   or ""),
                "region":          str(row.get("region") or ""),
                "source":          str(row.get("source") or ""),
                "as_of":           str(row.get("latest_time") or "")[:10],
                # ── 이동평균 ─────────────────────────────────────────
                "ma_30":           ma_30,
                "ma_90":           ma_90,
                # ── 변동률 ───────────────────────────────────────────
                "mom_change":      mom_change,    # % | None
                "yoy_change":      yoy_change,    # % | None
                # ── 변동성 ───────────────────────────────────────────
                "volatility_3m":   volatility_3m,
                # ── 메타 ─────────────────────────────────────────────
                "sample_count_3m": int(row.get("sample_count_3m") or 0),
                "historical_pct":  _safe_float(row.get("historical_pct")),
                "total_samples":   int(row.get("total_samples") or 0),
                # ── 파생 레이블 (프롬프트 직주입용) ──────────────────
                "trend_label":     _make_trend_label(mom_change, current_value, ma_90),
            }

        except Exception as exc:
            # 해당 지표만 스킵 — 나머지는 계속 처리
            logger.warning(
                "get_macro_ts_features 지표 처리 실패 — metric=%r, error=%s",
                metric_name, exc,
            )
            continue

    logger.info(
        "get_macro_ts_features 완료 — %d개 지표, months_back=%d",
        len(result), months_back,
    )
    return result
