"""
industry_collector.py
=====================
② 반도체 현물가격 (DRAM / NAND spot price)
   소스: TrendForce RSS 공개 기사 + DRAMeXchange 언급 파싱
         + FRED API (PPI for Semiconductors, 무료)
         + investing.com / Macrotrends 공개 데이터

⑤ WSTS 반도체 월별 출하량
   소스: WSTS public report RSS + 보도자료 파싱
         + Semiconductor Engineering WSTS 커버리지

DB:
  metric_observations (domain='semiconductor_price' | 'industry_shipment')
  tech_documents (TrendForce 기사)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import hashlib

import httpx

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    from email.utils import parsedate_to_datetime
    try:
        dt = parsedate_to_datetime(value)
        return dt.astimezone(timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
        except Exception:
            return None


def _upsert_doc(doc_uid, source, source_type, title, url, published_at, summary, tags, confidence, extra):
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tech_documents(
                  doc_uid, source, source_type, title, url, published_at, collected_at,
                  summary, content, tags, confidence, extra
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s::jsonb)
                ON CONFLICT (doc_uid) DO UPDATE SET
                  title=EXCLUDED.title, summary=COALESCE(EXCLUDED.summary,tech_documents.summary),
                  published_at=CASE WHEN EXCLUDED.source_type='wsts_report'
                    THEN EXCLUDED.published_at
                    ELSE COALESCE(EXCLUDED.published_at,tech_documents.published_at) END,
                  tags=EXCLUDED.tags, confidence=EXCLUDED.confidence,
                  extra=tech_documents.extra||EXCLUDED.extra,
                  collected_at=EXCLUDED.collected_at
                """,
                (doc_uid, source, source_type, title or "", url, published_at,
                 _now_utc(), (summary or "")[:3000], tags, confidence,
                 json.dumps(extra, ensure_ascii=False)),
            )
        conn.commit()


def _upsert_metric(
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
    extra: dict,
    is_proxy: bool = False,
) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
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
                ON CONFLICT DO NOTHING
                """,
                (
                    company_code, domain, metric_name, metric_value, unit,
                    is_proxy, observed_at, published_at, valid_from,
                    source_tier, confidence, prov_entity, source_url,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()


# ════════════════════════════════════════════════════════════════════
# 반도체 가격 관련 RSS 소스
# ════════════════════════════════════════════════════════════════════

PRICE_RSS_SOURCES = {
    "trendforce_news": {
        "url": "https://www.trendforce.com/feed/",
        "tags": ["trendforce", "dram_price", "nand_price", "memory_market"],
        "confidence": 0.85,
    },
    "digitimes_memory": {
        "url": "https://www.digitimes.com/rss/semiconductors.xml",
        "tags": ["digitimes", "memory", "dram", "nand", "semiconductor_market"],
        "confidence": 0.80,
    },
    "semiconductor_engineering_price": {
        "url": "https://semiengineering.com/feed/",
        "tags": ["semiengineering", "memory_price", "semiconductor_market"],
        "confidence": 0.80,
    },
}

# 가격 데이터 추출 패턴 (기사 내 언급 파싱)
PRICE_PATTERNS = [
    # "DRAM spot price rose to $X.XX"
    re.compile(
        r"(DRAM|NAND|HBM|DDR\d|LPDDR\d|GDDR\d)\s+(?:spot\s+)?price[s]?\s+"
        r"(?:rose|fell|increased|decreased|declined|dropped|up|down|reached|at|of)\s+"
        r"(?:to\s+)?[USD\$]?\s*(\d+\.?\d*)",
        re.IGNORECASE,
    ),
    # "$X.XX per GB"
    re.compile(
        r"[USD\$]\s*(\d+\.\d+)\s+per\s+(?:GB|Gb|gigabyte)",
        re.IGNORECASE,
    ),
    # "average selling price ... $X"
    re.compile(
        r"average\s+(?:selling\s+)?price\s+(?:of\s+)?[USD\$]?\s*(\d+\.?\d+)",
        re.IGNORECASE,
    ),
]

MEMORY_TYPE_PAT = re.compile(
    r"\b(HBM3[Ee]?|HBM[234]?|DDR5|DDR4|LPDDR5|LPDDR4|GDDR7|GDDR6|NAND|3D\s*NAND|QLC|TLC|MLC)\b",
    re.IGNORECASE,
)


def _extract_price_signals(title: str, summary: str, url: str, published_at: datetime | None) -> list[dict]:
    """기사 제목+요약에서 가격 시그널 추출."""
    text = f"{title} {summary}"
    signals = []

    mem_types = list(set(MEMORY_TYPE_PAT.findall(text))) or ["memory"]

    for pat in PRICE_PATTERNS:
        for m in pat.finditer(text):
            groups = m.groups()
            # 마지막 그룹이 숫자
            price_str = None
            mem_type = "memory"
            for g in groups:
                if g and re.match(r"^\d+\.?\d*$", g):
                    price_str = g
                elif g and re.match(r"^[A-Za-z]", g):
                    mem_type = g

            if price_str:
                try:
                    price = float(price_str)
                    if 0.01 < price < 50000:  # sanity check
                        signals.append({
                            "memory_type": mem_types[0] if mem_types else mem_type,
                            "price": price,
                            "context": text[max(0, m.start() - 50):m.end() + 50],
                            "url": url,
                        })
                except ValueError:
                    pass

    return signals


def collect_semiconductor_prices() -> None:
    """
    반도체 현물가격 수집.
    직접 가격 DB가 없으므로:
    1. TrendForce/Digitimes RSS 기사에서 가격 언급 파싱 → metric_observations
    2. 기사 자체는 tech_documents에 저장
    """
    logger.info("=== Semiconductor prices: START ===")
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.error("feedparser not installed")
        return

    PRICE_KEYWORDS = {
        "dram", "nand", "hbm", "memory price", "spot price", "contract price",
        "asp", "average selling price", "memory market", "dram market", "nand market",
        "memory oversupply", "memory undersupply", "bit growth",
    }

    total_articles, total_signals = 0, 0

    for source_key, cfg in PRICE_RSS_SOURCES.items():
        try:
            feed = feedparser.parse(cfg["url"])
            entries = getattr(feed, "entries", []) or []
            logger.info("RSS parsed: %s → %d entries", source_key, len(entries))

            for entry in entries:
                title   = str(getattr(entry, "title", "") or "").strip()
                link    = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                pub_raw = str(getattr(entry, "published", "") or "").strip()
                pub_at  = _to_dt(pub_raw)

                if not link:
                    continue

                text_l = f"{title} {summary}".lower()
                if not any(kw in text_l for kw in PRICE_KEYWORDS):
                    continue

                # tech_documents 저장
                doc_uid = hashlib.sha1(f"{source_key}|{link}".encode()).hexdigest()
                tags = cfg["tags"] + [kw for kw in PRICE_KEYWORDS if kw in text_l]
                _upsert_doc(
                    doc_uid=doc_uid, source=source_key,
                    source_type="semiconductor_price_article",
                    title=title or link, url=link,
                    published_at=pub_at, summary=summary[:2000],
                    tags=list(set(tags))[:20],
                    confidence=cfg["confidence"],
                    extra={"published_raw": pub_raw},
                )
                total_articles += 1

                # 가격 시그널 파싱 → metric_observations
                signals = _extract_price_signals(title, summary, link, pub_at)
                for sig in signals:
                    mem = sig["memory_type"].upper().replace(" ", "_")
                    metric_name = f"spot_price_{mem}_usd_per_gb"
                    prov = f"{source_key}|{link}|{metric_name}"
                    _upsert_metric(
                        company_code="MARKET",
                        domain="semiconductor_price",
                        metric_name=metric_name,
                        metric_value=sig["price"],
                        unit="USD/GB",
                        observed_at=pub_at or _now_utc(),
                        published_at=pub_at or _now_utc(),
                        valid_from=pub_at or _now_utc(),
                        source_tier=3,
                        confidence=0.60,  # 기사 파싱이므로 낮은 신뢰도
                        prov_entity=prov,
                        source_url=link,
                        extra={"context": sig["context"][:300], "source": source_key},
                        is_proxy=True,
                    )
                    total_signals += 1

            time.sleep(1.5)

        except Exception as e:
            logger.error("Price RSS failed: %s: %s", source_key, e)

    logger.info("=== Prices done: articles=%d, price_signals=%d ===", total_articles, total_signals)


# ════════════════════════════════════════════════════════════════════
# FRED API — PPI for Semiconductors & Related Devices (무료)
# https://fred.stlouisfed.org/series/PCU334413334413
# ════════════════════════════════════════════════════════════════════

FRED_SERIES = {
    # PPI: Semiconductor & Related Devices (검증됨)
    "PCU334413334413":  ("ppi_semiconductor_devices",        "index",  "semiconductor_price"),
    # Industrial Production: Semiconductor (검증됨)
    "IPG3341N":         ("industrial_production_semiconductor", "index", "semiconductor_price"),
    # Industrial Production: Computers & Electronics (검증됨)
    "IPG3344N":         ("industrial_production_electronics",  "index", "semiconductor_price"),
    # PPI: Printed Circuit Boards (검증됨)
    "PCU333618333618":  ("ppi_pcb",                           "index", "semiconductor_price"),
}


def collect_fred_semiconductor_indicators() -> None:
    """FRED API에서 반도체 관련 거시 지표 수집 (무료, API key 필요)."""
    api_key = os.getenv("FRED_API_KEY", "").strip()
    if not api_key:
        logger.info("FRED_API_KEY 없음 → 공개 JSON 엔드포인트로 시도")
        # FRED는 API key 없이도 일부 데이터 접근 가능

    logger.info("=== FRED: collect_fred_semiconductor_indicators START ===")
    total = 0

    for series_id, (metric_name, unit, domain) in FRED_SERIES.items():
        try:
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
            if api_key:
                url = (
                    f"https://api.stlouisfed.org/fred/series/observations"
                    f"?series_id={series_id}&api_key={api_key}&file_type=json"
                    f"&sort_order=desc&limit=60"
                )

            headers = {"User-Agent": "SemiconductorAgentBot/1.0"}
            with httpx.Client(timeout=20, follow_redirects=True, headers=headers) as c:
                r = c.get(url)
                r.raise_for_status()

            observations = []
            if api_key and "observations" in r.text:
                data = r.json()
                for obs in data.get("observations", []):
                    if obs.get("value", ".") == ".":
                        continue
                    try:
                        observations.append((obs["date"], float(obs["value"])))
                    except Exception:
                        pass
            else:
                # CSV 파싱
                for line in r.text.strip().splitlines()[1:]:
                    parts = line.split(",")
                    if len(parts) < 2 or parts[1].strip() == ".":
                        continue
                    try:
                        observations.append((parts[0].strip(), float(parts[1].strip())))
                    except Exception:
                        pass

            for date_str, value in observations[-60:]:  # 최근 60개월
                try:
                    obs_dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    continue
                prov = f"fred:{series_id}:{date_str}"
                _upsert_metric(
                    company_code="MARKET",
                    domain=domain,
                    metric_name=metric_name,
                    metric_value=value,
                    unit=unit,
                    observed_at=obs_dt,
                    published_at=obs_dt,
                    valid_from=obs_dt,
                    source_tier=2,
                    confidence=0.90,
                    prov_entity=prov,
                    source_url=f"https://fred.stlouisfed.org/series/{series_id}",
                    extra={"series_id": series_id},
                )
                total += 1

            logger.info("FRED series done: %s (%s) → %d obs", series_id, metric_name, len(observations))
            time.sleep(1.0)

        except Exception as e:
            logger.error("FRED failed: %s: %s", series_id, e)

    logger.info("=== FRED done: total_metrics=%d ===", total)


# ════════════════════════════════════════════════════════════════════
# ⑤ WSTS 반도체 출하량
# ════════════════════════════════════════════════════════════════════

WSTS_COVERAGE_RSS = [
    # WSTS 보도자료를 커버하는 매체 RSS
    "https://semiengineering.com/feed/",
    "https://www.eetasia.com/feed/",
]

WSTS_PATTERNS = [
    re.compile(
        r"(?:global\s+)?semiconductor\s+(?:sales|revenue|shipments?)\s+"
        r"(?:reached?|totaled?|were|hit|of)\s+[USD\$]?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(?:billion|million|bn|mn)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"WSTS\s+(?:reported?|data|shows?)\s+.*?[USD\$]?\s*([\d,]+(?:\.\d+)?)\s*"
        r"(?:billion|million|bn|mn)?",
        re.IGNORECASE,
    ),
    re.compile(
        r"book-to-bill\s+ratio\s+(?:of\s+|was\s+|at\s+)?([\d]+\.[\d]+)",
        re.IGNORECASE,
    ),
]


def collect_wsts_bluebook() -> None:
    """
    WSTS 반도체 출하량 수집.
    WSTS 보도자료는 유료지만 Semiconductor Engineering 등에서 수치 언급.
    1. RSS 기사에서 출하량/매출 수치 파싱
    2. WSTS 공개 보도자료 페이지 메타데이터 저장
    """
    logger.info("=== WSTS: collect_wsts_bluebook START ===")
    try:
        import feedparser  # type: ignore
    except ImportError:
        logger.error("feedparser not installed")
        return

    WSTS_KEYWORDS = {
        "wsts", "semiconductor sales", "global semiconductor",
        "chip sales", "semiconductor market", "book-to-bill",
        "semiconductor revenue", "memory revenue",
    }

    total_articles, total_signals = 0, 0

    for rss_url in WSTS_COVERAGE_RSS:
        try:
            feed = feedparser.parse(rss_url)
            entries = getattr(feed, "entries", []) or []

            for entry in entries:
                title   = str(getattr(entry, "title", "") or "").strip()
                link    = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                pub_raw = str(getattr(entry, "published", "") or "").strip()
                pub_at  = _to_dt(pub_raw)

                if not link:
                    continue
                text_l = f"{title} {summary}".lower()
                if not any(kw in text_l for kw in WSTS_KEYWORDS):
                    continue

                # 저장
                source_key = rss_url.split("//")[1].split("/")[0].replace("www.", "")
                doc_uid = hashlib.sha1(f"wsts|{link}".encode()).hexdigest()
                _upsert_doc(
                    doc_uid=doc_uid, source=f"wsts_coverage_{source_key}",
                    source_type="wsts_article",
                    title=title or link, url=link,
                    published_at=pub_at, summary=summary[:2000],
                    tags=["wsts", "semiconductor_shipment", "industry_data"],
                    confidence=0.78,
                    extra={"published_raw": pub_raw},
                )
                total_articles += 1

                # 수치 파싱
                text = f"{title} {summary}"
                for pat in WSTS_PATTERNS:
                    for m in pat.finditer(text):
                        val_str = m.group(1).replace(",", "")
                        try:
                            val = float(val_str)
                        except Exception:
                            continue

                        # "billion" 단위 변환
                        ctx = text[max(0, m.start() - 20):m.end() + 30].lower()
                        if "billion" in ctx or "bn" in ctx:
                            val_billion = val
                        elif "million" in ctx or "mn" in ctx:
                            val_billion = val / 1000.0
                        else:
                            val_billion = val

                        metric_name = (
                            "book_to_bill_ratio" if "book-to-bill" in m.pattern.pattern.lower()
                            else "global_semiconductor_sales_usd_bn"
                        )

                        if metric_name == "book_to_bill_ratio" and not (0.5 < val < 2.0):
                            continue
                        if metric_name != "book_to_bill_ratio" and not (1 < val_billion < 1000):
                            continue

                        metric_val = val if "book_to_bill" in metric_name else val_billion
                        prov = f"wsts_coverage|{link}|{metric_name}"
                        _upsert_metric(
                            company_code="MARKET",
                            domain="industry_shipment",
                            metric_name=metric_name,
                            metric_value=metric_val,
                            unit="ratio" if "book_to_bill" in metric_name else "USD_billions",
                            observed_at=pub_at or _now_utc(),
                            published_at=pub_at or _now_utc(),
                            valid_from=pub_at or _now_utc(),
                            source_tier=3,
                            confidence=0.65,
                            prov_entity=prov,
                            source_url=link,
                            extra={"context": ctx[:200], "source": source_key},
                            is_proxy=True,
                        )
                        total_signals += 1

            time.sleep(1.5)
        except Exception as e:
            logger.error("WSTS RSS failed: %s: %s", rss_url, e)

    # WSTS 공식 보도자료 메타데이터
    _upsert_doc(
        doc_uid=hashlib.sha1(b"wsts_official_monthly").hexdigest(),
        source="wsts",
        source_type="wsts_report",
        title="WSTS Monthly Semiconductor Sales Report",
        url="https://www.wsts.org/76/Recent-Historical-Data",
        # This is an undated directory, not a dated monthly report.
        published_at=None,
        summary=(
            "WSTS (World Semiconductor Trade Statistics) provides monthly global "
            "semiconductor sales data by product category (Memory, Logic, Analog, Discrete, etc.) "
            "and by region (Americas, Europe, Japan, Asia Pacific). "
            "Key metrics: total sales USD billions, YoY growth, book-to-bill ratio."
        ),
        tags=["wsts", "semiconductor_shipment", "global_market", "memory", "industry_data"],
        confidence=0.95,
        extra={"access": "paid_membership", "frequency": "monthly", "lag_weeks": 6},
    )

    logger.info("=== WSTS done: articles=%d, signals=%d ===", total_articles, total_signals)


def collect_industry_press_metrics() -> None:
    """SEMI/Gartner/IDC 보도자료에서 반도체 산업 지표 파싱."""
    logger.info("=== Industry press: START ===")
    try:
        import feedparser  # type: ignore
    except ImportError:
        return

    INDUSTRY_SOURCES = {
        "semi_org": "https://www.semi.org/en/rss.xml",
        "eetasia_industry": "https://www.eetasia.com/feed/",
    }
    INDUSTRY_KEYWORDS = {
        "fab capacity", "capex", "wafer starts", "utilization rate",
        "equipment spending", "wafer shipments", "silicon wafer",
        "semiconductor equipment", "fab investment",
    }

    total = 0
    for source_key, rss_url in INDUSTRY_SOURCES.items():
        try:
            feed = feedparser.parse(rss_url)
            entries = getattr(feed, "entries", []) or []
            for entry in entries:
                title   = str(getattr(entry, "title", "") or "").strip()
                link    = str(getattr(entry, "link", "") or "").strip()
                summary = str(getattr(entry, "summary", "") or "").strip()
                pub_raw = str(getattr(entry, "published", "") or "").strip()
                pub_at  = _to_dt(pub_raw)
                if not link:
                    continue
                text_l = f"{title} {summary}".lower()
                if not any(kw in text_l for kw in INDUSTRY_KEYWORDS):
                    continue

                doc_uid = hashlib.sha1(f"{source_key}|{link}".encode()).hexdigest()
                _upsert_doc(
                    doc_uid=doc_uid, source=source_key,
                    source_type="industry_press",
                    title=title or link, url=link,
                    published_at=pub_at, summary=summary[:2000],
                    tags=["semiconductor_industry", "capex", "fab", "equipment"],
                    confidence=0.78,
                    extra={"published_raw": pub_raw},
                )
                total += 1
            time.sleep(1.0)
        except Exception as e:
            logger.error("Industry press failed: %s: %s", source_key, e)

    logger.info("=== Industry press done: articles=%d ===", total)
