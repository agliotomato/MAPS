"""
뉴스 기사 분석 모듈 — FinBERT 감성 분석 + spaCy NER

FinBERT:
    ProsusAI/finbert 모델 (최초 실행 시 ~440MB 자동 다운로드)
    입력: 최대 512토큰 (긴 기사는 앞 부분 사용)
    출력: positive / negative / neutral 확률

spaCy NER:
    영문: en_core_web_sm (ORG·GPE·PRODUCT)
    한국어: ko_core_news_sm (ORG·LOC·PS)
    + 반도체 도메인 키워드 사전 기반 추가 추출

사용법:
    from macro_agent.analysis.news_analyzer import analyze_article
    result = analyze_article(text="...", lang="en")
    # result: {"sentiment_pos": 0.7, "sentiment_neg": 0.1, "sentiment_neu": 0.2,
    #          "entities": {"companies": [...], "countries": [...], "products": [...]},
    #          "keywords": [...]}
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# ── 반도체·거시경제 도메인 키워드 사전 ───────────────────────────────────────

_SEMI_KEYWORDS_EN: frozenset[str] = frozenset({
    # 기업
    "tsmc", "samsung", "sk hynix", "sk-hynix", "hynix",
    "intel", "nvidia", "amd", "qualcomm", "broadcom",
    "asml", "applied materials", "lam research", "kla",
    "micron", "western digital", "seagate",
    "arm", "synopsys", "cadence", "marvell",
    # 제품·기술
    "semiconductor", "chip", "wafer", "foundry", "fab",
    "dram", "nand", "hbm", "sram", "flash memory",
    "logic chip", "ai chip", "gpu", "cpu", "soc",
    "advanced packaging", "chiplet", "3nm", "2nm", "5nm",
    "euv", "lithography", "photomask",
    # 거시경제
    "federal reserve", "fed", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "ppi", "gdp", "pmi",
    "exchange rate", "usd", "krw", "yen", "yuan",
    "export", "import", "tariff", "trade deficit", "trade surplus",
    # 지정학
    "export control", "entity list", "chips act", "sanctions",
    "supply chain", "reshoring", "nearshoring",
    "taiwan strait", "geopolit",
    # 시장·업황
    "sox index", "semiconductor index", "inventory", "utilization",
    "wafer start", "capacity", "capex", "guidance",
    "downcycle", "upcycle", "recovery", "memory price",
})

_SEMI_KEYWORDS_KO: frozenset[str] = frozenset({
    "반도체", "칩", "웨이퍼", "파운드리", "팹", "메모리",
    "낸드", "드램", "hbm", "플래시",
    "삼성전자", "삼성", "sk하이닉스", "하이닉스", "tsmc",
    "인텔", "엔비디아", "nvidia", "퀄컴",
    "수출규제", "수출통제", "무역전쟁", "공급망",
    "금리", "기준금리", "환율", "달러", "원화", "위안화",
    "인플레이션", "pmi", "gdp", "무역수지",
    "대만해협", "지정학", "제재", "관세",
    "재고조정", "업황", "가동률", "설비투자",
})

# ── FinBERT 지연 로딩 ───────────────────────────────────────────────────────

_finbert_pipeline: Any = None
_FINBERT_MODEL = "ProsusAI/finbert"


def _get_finbert() -> Any:
    """FinBERT 파이프라인을 지연 로딩합니다. 최초 호출 시 모델 다운로드 (~440MB)."""
    global _finbert_pipeline
    if _finbert_pipeline is not None:
        return _finbert_pipeline

    try:
        from transformers import pipeline
        logger.info("FinBERT 모델 로딩 중 (%s)...", _FINBERT_MODEL)
        _finbert_pipeline = pipeline(
            "text-classification",
            model=_FINBERT_MODEL,
            top_k=None,          # 모든 레이블 확률 반환
            device=-1,           # CPU
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT 로딩 완료")
        return _finbert_pipeline
    except Exception as exc:
        logger.warning("FinBERT 로딩 실패 (감성 분석 스킵): %s", exc)
        return None


# ── spaCy NER 지연 로딩 ─────────────────────────────────────────────────────

_nlp_en: Any = None
_nlp_ko: Any = None


@lru_cache(maxsize=1)
def _get_nlp_en() -> Any:
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        logger.warning("spaCy en_core_web_sm 로딩 실패: %s", exc)
        return None


@lru_cache(maxsize=1)
def _get_nlp_ko() -> Any:
    try:
        import spacy
        return spacy.load("ko_core_news_sm")
    except Exception as exc:
        logger.warning("spaCy ko_core_news_sm 로딩 실패: %s", exc)
        return None


# ── 분석 함수 ───────────────────────────────────────────────────────────────

def analyze_sentiment(text: str) -> dict[str, float]:
    """
    FinBERT로 금융 감성을 분석합니다.

    Returns:
        {"positive": float, "negative": float, "neutral": float}
        모델 로딩 실패 시 {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    """
    default = {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    if not text or not text.strip():
        return default

    finbert = _get_finbert()
    if finbert is None:
        return default

    try:
        # FinBERT는 최대 512 토큰 — 긴 기사는 앞 2000자만 사용
        truncated = text[:2000].strip()
        result = finbert(truncated)

        # result: [[{"label": "positive", "score": 0.9}, ...]]
        labels = result[0] if isinstance(result, list) and result else []
        scores: dict[str, float] = {}
        for item in labels:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            scores[label] = score

        return {
            "positive": round(scores.get("positive", 0.0), 4),
            "negative": round(scores.get("negative", 0.0), 4),
            "neutral":  round(scores.get("neutral", 1.0), 4),
        }
    except Exception as exc:
        logger.warning("FinBERT 감성 분석 오류: %s", exc)
        return default


def extract_entities(text: str, lang: str = "en") -> dict[str, list[str]]:
    """
    spaCy NER + 도메인 키워드 사전으로 엔티티를 추출합니다.

    Returns:
        {
          "companies":  ["TSMC", "Samsung", ...],
          "countries":  ["US", "China", ...],
          "products":   ["HBM", "DRAM", ...],
          "keywords":   ["semiconductor", "export control", ...],
        }
    """
    result: dict[str, list[str]] = {
        "companies": [], "countries": [], "products": [], "keywords": [],
    }
    if not text:
        return result

    # ── spaCy NER ──────────────────────────────────────────────────────
    nlp = _get_nlp_en() if lang == "en" else _get_nlp_ko()
    if nlp is not None:
        try:
            doc = nlp(text[:5000])  # 처리 길이 제한
            for ent in doc.ents:
                label = ent.label_
                ent_text = ent.text.strip()
                if not ent_text or len(ent_text) < 2:
                    continue
                if label in ("ORG", "PERSON"):
                    result["companies"].append(ent_text)
                elif label in ("GPE", "LOC", "NORP"):
                    result["countries"].append(ent_text)
                elif label in ("PRODUCT", "WORK_OF_ART", "EVENT"):
                    result["products"].append(ent_text)
        except Exception as exc:
            logger.warning("spaCy NER 오류: %s", exc)

    # ── 도메인 키워드 사전 매칭 ────────────────────────────────────────
    text_lower = text.lower()
    keywords_set = _SEMI_KEYWORDS_EN if lang == "en" else _SEMI_KEYWORDS_KO

    found_keywords: list[str] = []
    for kw in keywords_set:
        if kw in text_lower:
            found_keywords.append(kw)

    result["keywords"] = sorted(found_keywords)

    # 중복 제거 + 최대 30개 제한
    for key in ("companies", "countries", "products"):
        seen: set[str] = set()
        deduped: list[str] = []
        for item in result[key]:
            norm = item.strip()
            if norm and norm not in seen:
                seen.add(norm)
                deduped.append(norm)
        result[key] = deduped[:30]

    return result


def analyze_article(
    text: str,
    lang: str = "en",
) -> dict[str, Any]:
    """
    기사 전문을 FinBERT + NER로 분석합니다.

    Args:
        text: 기사 본문 (또는 제목+요약)
        lang: "en" | "ko"

    Returns:
        {
          "sentiment_pos": float,
          "sentiment_neg": float,
          "sentiment_neu": float,
          "entities": {"companies": [...], "countries": [...], "products": [...]},
          "keywords": [...],
        }
    """
    sentiment = analyze_sentiment(text)
    entities = extract_entities(text, lang=lang)
    keywords = entities.pop("keywords", [])

    return {
        "sentiment_pos": sentiment["positive"],
        "sentiment_neg": sentiment["negative"],
        "sentiment_neu": sentiment["neutral"],
        "entities": entities,
        "keywords": keywords,
    }
