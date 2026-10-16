"""
스마트 청커 — 반도체·거시경제 키워드 기반 단락 필터링 청킹

기존 RecursiveCharacterTextSplitter(800자 고정 분할) 대신
단락(paragraph) 단위로 먼저 쪼갠 뒤 관련 키워드가 포함된 단락만 선택합니다.

처리 흐름:
    1. 원문을 빈 줄(\n\n) 기준으로 단락 분리
    2. 각 단락에서 반도체·거시경제 키워드 포함 여부 확인
    3. 관련 단락만 추출 (없으면 전체 사용 — 폴백)
    4. 너무 긴 단락은 RecursiveCharacterTextSplitter로 재분할

공개 함수:
    clean_firecrawl_markdown(text) → str   Firecrawl 마크다운 보일러플레이트 제거
    smart_chunk(text, lang)        → list[str]
"""

from __future__ import annotations

import re
from typing import Final

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── 키워드 사전 ─────────────────────────────────────────────────────────────

_KW_EN: frozenset[str] = frozenset({
    # 반도체 기업
    "tsmc", "samsung", "hynix", "intel", "nvidia", "amd", "qualcomm",
    "broadcom", "asml", "applied materials", "lam research", "kla",
    "micron", "arm", "synopsys", "cadence",
    # 반도체 기술·제품
    "semiconductor", "chip", "wafer", "foundry", "fab",
    "dram", "nand", "hbm", "sram", "flash",
    "logic", "memory", "packaging", "chiplet",
    "euv", "lithography", "3nm", "5nm", "7nm",
    "gpu", "cpu", "ai chip", "soc",
    # 거시경제
    "federal reserve", "fed funds", "interest rate", "rate hike", "rate cut",
    "inflation", "cpi", "ppi", "gdp", "pmi",
    "exchange rate", "usd", "krw", "yen", "yuan",
    "export", "tariff", "trade",
    # 지정학·규제 (6대 필터 패턴 키워드 보강)
    "export control", "chips act", "sanction", "entity list",
    "supply chain", "taiwan", "china", "korea",
    "geopolit", "military",
    "taiwan strait", "cross-strait", "south china sea",
    "critical mineral", "gallium", "germanium", "rare earth",
    "tariff retaliation", "friend-shoring", "friendshoring",
    "military conflict", "geopolitical risk",
    "bis ear", "fdpr", "foreign direct product",
    "subsidy", "reshoring", "nearshoring",
    # 업황·시장
    "inventory", "utilization", "capacity", "wafer start",
    "guidance", "revenue", "earnings", "downturn", "recovery",
    "sox", "semiconductor index",
})

_KW_KO: frozenset[str] = frozenset({
    "반도체", "칩", "웨이퍼", "파운드리", "팹", "메모리",
    "낸드", "드램", "hbm", "플래시",
    "삼성", "하이닉스", "tsmc", "인텔", "엔비디아",
    "수출규제", "수출통제", "공급망",
    "금리", "기준금리", "환율", "달러", "원화",
    "인플레이션", "pmi", "gdp", "무역수지",
    "대만", "중국", "한국", "지정학", "제재", "관세",
    "재고", "업황", "가동률", "설비투자",
    # 지정학 심화
    "희토류", "갈륨", "게르마늄", "희귀금속",
    "대만해협", "남중국해", "군사", "충돌",
    "보조금", "리쇼어링", "우방국", "동맹",
    "타리프", "보복관세",
})

_MAX_CHUNK: Final[int] = 800
_MIN_PARA: Final[int] = 80    # 너무 짧은 단락 무시
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=_MAX_CHUNK,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", "。", " ", ""],
    length_function=len,
)


# ── Firecrawl 마크다운 클리너 ─────────────────────────────────────────────────

# 보일러플레이트 단일 줄 패턴 (대소문자 무관)
_BOILERPLATE_LINE_RE = re.compile(
    r"(?:"
    r"^\s*share\s+(this\s+)?(article|story|post)\b"        # Share this article
    r"|^\s*follow\s+us\s+(on|at)\b"                         # Follow us on
    r"|\b(sign|log)\s+in\s+to\s+(read|continue|access)\b"  # Sign in to read
    r"|\bsubscribe\s+to\s+(read|continue|access|our)\b"     # Subscribe to read
    r"|\balready\s+a\s+subscriber\b"                        # Already a subscriber
    r"|^\s*©\s*\d{4}"                                       # © 2024
    r"|\ball\s+rights\s+reserved\b"                         # All rights reserved
    r"|\bcookie\s+(policy|notice|consent)\b"                # Cookie policy
    r"|\bprivacy\s+policy\b"                                # Privacy policy
    r"|\bterms\s+(of\s+use|and\s+conditions)\b"             # Terms of use
    r"|^\s*(advertisement|sponsored\s+content)\s*$"         # Advertisement
    r"|\bnewsletter\s+sign[\s-]?up\b"                       # Newsletter sign-up
    r"|\benable\s+javascript\b"                             # Enable JavaScript
    r")",
    re.I | re.M,
)

# "Related articles / Read more" 섹션 헤더 (이하 스킵)
_SECTION_SKIP_RE = re.compile(
    r"^#{1,3}\s*"
    r"(related(\s+(articles|stories|news|content))?|read\s+more"
    r"|more\s+from|see\s+also|recommended|editor.s\s+pick"
    r"|you\s+might\s+(also\s+)?like|trending\s+now)\s*$",
    re.I,
)

# 새 주요 섹션 헤더 (H1/H2) — skip 구간 탈출용
_MAJOR_HEADING_RE = re.compile(r"^#{1,2}\s+\w", re.M)

# 내비게이션 링크 감지: 짧은 줄에서 마크다운 링크 비율이 높으면 nav 블록으로 판단
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _is_link_nav_block(line: str) -> bool:
    """짧은 줄에서 마크다운 링크 텍스트 비율 ≥ 70%이고 링크 3개 이상이면 nav 블록."""
    stripped = line.strip()
    if len(stripped) > 400:
        return False
    links = _MD_LINK_RE.findall(stripped)
    if len(links) < 3:
        return False
    link_chars = sum(len(t) for t in links)
    non_url_text = re.sub(r"\([^)]+\)", "", stripped)
    return link_chars / max(len(non_url_text), 1) > 0.70


def clean_firecrawl_markdown(text: str) -> str:
    """
    Firecrawl 마크다운 출력에서 보일러플레이트를 제거합니다.

    제거 항목:
        - 마크다운 이미지 태그 (![]())
        - 내비게이션 링크 블록 (링크 비율 ≥ 70%인 짧은 줄)
        - 소셜 공유·구독 유도·쿠키·저작권·개인정보 고지 문구
        - "Related articles / Read more" 이후 섹션
        - 연속 빈 줄 정리 (3줄 이상 → 2줄)

    trafilatura 평문 본문에 실행해도 패턴이 매칭되지 않으므로 안전합니다.
    """
    if not text:
        return text

    # Step 1: 마크다운 이미지 태그 제거
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)

    # Step 2: 줄 단위 필터링
    lines = text.split("\n")
    cleaned: list[str] = []
    in_skip_section = False

    for line in lines:
        stripped = line.strip()

        # Related/Read more 섹션 헤더 감지 → 이후 건너뜀
        if _SECTION_SKIP_RE.match(stripped):
            in_skip_section = True
            continue

        # skip 구간 중 새 주요 헤더(H1/H2)가 나오면 복귀
        if in_skip_section:
            if _MAJOR_HEADING_RE.match(stripped) and not _SECTION_SKIP_RE.match(stripped):
                in_skip_section = False
            else:
                continue

        # 보일러플레이트 단일 줄 제거
        if stripped and _BOILERPLATE_LINE_RE.search(stripped):
            continue

        # 내비게이션 링크 블록 제거
        if _is_link_nav_block(line):
            continue

        cleaned.append(line)

    # Step 3: 연속 빈 줄 압축
    result = "\n".join(cleaned)
    result = re.sub(r"\n{3,}", "\n\n", result)

    return result.strip()


# ── 스마트 청킹 ──────────────────────────────────────────────────────────────

def _has_keyword(para: str, keywords: frozenset[str]) -> bool:
    """단락에 키워드가 하나라도 포함되면 True."""
    para_lower = para.lower()
    return any(kw in para_lower for kw in keywords)


def smart_chunk(text: str, lang: str = "en") -> list[str]:
    """
    반도체·거시경제 키워드 포함 단락만 선별해 청킹합니다.

    Args:
        text: 기사 원문 (title + body 또는 summary)
        lang: "en" | "ko" — 키워드 사전 선택

    Returns:
        ChromaDB에 저장할 청크 목록 (비어있으면 원문 전체 청킹으로 폴백)
    """
    if not text or not text.strip():
        return []

    keywords = _KW_EN if lang == "en" else _KW_KO

    # 1. 단락 분리 (빈 줄 기준, 짧은 단락 제거)
    raw_paras = re.split(r"\n{2,}", text.strip())
    paras = [p.strip() for p in raw_paras if len(p.strip()) >= _MIN_PARA]

    if not paras:
        paras = [text.strip()]

    # 2. 키워드 포함 단락 필터링
    relevant = [p for p in paras if _has_keyword(p, keywords)]

    # 3. 관련 단락이 없으면 전체 단락 사용 (폴백)
    selected = relevant if relevant else paras

    # 4. 너무 긴 단락은 RecursiveCharacterTextSplitter로 재분할
    chunks: list[str] = []
    for para in selected:
        if len(para) <= _MAX_CHUNK:
            chunks.append(para)
        else:
            sub_chunks = _SPLITTER.split_text(para)
            chunks.extend(sub_chunks)

    # 5. 최소 길이 미달 청크 제거
    return [c for c in chunks if len(c.strip()) >= 30]
