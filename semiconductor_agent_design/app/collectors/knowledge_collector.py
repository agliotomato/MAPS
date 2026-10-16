"""
knowledge_collector.py
======================
반도체 기술 지식 레이어 수집기.

Storage architecture (arXiv 패턴과 동일):
  원본 파일  → data/raw/knowledge/{irds,jedec,conferences,equip}/<file>
  메타데이터 → PostgreSQL  tech_documents  (source_type: 'irds'|'jedec'|'hotchips'|'equip_doc'|'semi_blog')
  청크       → PostgreSQL  tech_document_chunks
  벡터       → Qdrant Cloud  irds_chunks | jedec_chunks | tech_blog_chunks

임베딩 모델:
  - OPENAI_API_KEY 있으면 text-embedding-3-small (1536 dim)
  - 없으면 sentence-transformers all-MiniLM-L6-v2 (384 dim, 무료)
  - QDRANT_{COLLECTION}_DIM env로 컬렉션별 dim 오버라이드 가능

수집 대상:
  A. IRDS (IEEE International Roadmap for Devices and Systems)
     - 연 1회 (11월) 최신 에디션 다운로드
     - Focus Teams: More Moore, Packaging, Yield, Memory, etc.
  B. JEDEC 표준 메타데이터 + 무료 공개 PDF
     - HBM2/3/3E/4 (JESD235/238), DDR5 (JESD79-5), LPDDR5 (JESD209-5)
  C. Hot Chips / FMS 공개 자료
     - hotchips.org archives + IEEE open-access
  D. 반도체 장비사 기술 문서
     - ASML, Lam Research, Applied Materials, KLA 공개 technical papers
  E. 삼성/Micron 기술 블로그 (SK Hynix는 news_collector.py에 있음)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Iterator

import httpx

from app.db.postgres import get_pg_conn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 환경 변수
# ---------------------------------------------------------------------------

def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "1" if default else "0").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# 임베딩 엔진 (OpenAI ada-002 우선 / fallback → sentence-transformers)
# ---------------------------------------------------------------------------

_st_model = None          # sentence-transformers 인스턴스 (lazy)
_openai_client = None     # OpenAI 클라이언트 (lazy)
_embed_dim: int | None = None
_LEGACY_KNOWLEDGE_COLLECTIONS = {"irds_chunks", "jedec_chunks", "tech_blog_chunks"}


def _get_embed_dim() -> int:
    global _embed_dim
    if _embed_dim is not None:
        return _embed_dim
    if os.getenv("OPENAI_API_KEY", "").strip():
        _embed_dim = 1536
    else:
        _embed_dim = _env_int("EMBED_DIM", 384)
    return _embed_dim


def _embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트 목록 → 벡터 목록. OpenAI 우선, fallback → sentence-transformers."""
    if not texts:
        return []
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        return _embed_openai(texts, openai_key)
    return _embed_st(texts)


def _embed_texts_for_collection(collection_name: str, texts: list[str]) -> list[list[float]]:
    """
    Legacy knowledge collections were historically created with MiniLM 384-dim vectors.
    Keep them on the legacy embedding path so JEDEC/IRDS indexing remains compatible.
    """
    if collection_name in _LEGACY_KNOWLEDGE_COLLECTIONS:
        return _embed_st(texts)
    return _embed_texts(texts)


def _collection_embed_dim(collection_name: str) -> int:
    if collection_name in _LEGACY_KNOWLEDGE_COLLECTIONS:
        return _env_int("EMBED_DIM", 384)
    return _get_embed_dim()


def _embed_openai(texts: list[str], api_key: str) -> list[list[float]]:
    global _openai_client
    try:
        import openai  # type: ignore
    except ImportError:
        logger.warning("openai package not installed; falling back to sentence-transformers")
        return _embed_st(texts)

    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=api_key)
    try:
        response = _openai_client.embeddings.create(
            model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
            input=texts,
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error("OpenAI embedding failed: %s — falling back to ST", e)
        return _embed_st(texts)


def _embed_st(texts: list[str]) -> list[list[float]]:
    global _st_model
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        logger.error("sentence-transformers not installed. pip install sentence-transformers --break-system-packages")
        return [[] for _ in texts]
    if _st_model is None:
        model_name = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
        logger.info("Loading sentence-transformers model: %s", model_name)
        _st_model = SentenceTransformer(model_name)
    vecs = _st_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]


def _upsert_qdrant_chunks(
    collection_name: str,
    doc_uid: str,
    chunks: list[str],
    payload_extra: dict | None = None,
) -> int:
    """청크 텍스트를 임베딩해서 Qdrant에 upsert. 실패해도 나머지 파이프라인 계속."""
    if not chunks:
        return 0
    try:
        from qdrant_client import QdrantClient  # type: ignore
        from qdrant_client.http import models as qm  # type: ignore
        from app.db.qdrant import get_qdrant_client, ensure_collection

        dim = _collection_embed_dim(collection_name)
        ensure_collection(collection_name, size=dim)

        client = get_qdrant_client()
        BATCH = _env_int("EMBED_BATCH_SIZE", 32)
        total = 0

        for i in range(0, len(chunks), BATCH):
            batch = chunks[i : i + BATCH]
            vecs = _embed_texts_for_collection(collection_name, batch)
            if not vecs or not vecs[0]:
                logger.warning("Empty embeddings for collection=%s batch=%d", collection_name, i)
                continue

            points = []
            for j, (vec, text) in enumerate(zip(vecs, batch)):
                chunk_idx = i + j
                point_id = int(hashlib.sha256(f"{doc_uid}_{chunk_idx}".encode()).hexdigest()[:15], 16)
                payload = {
                    "doc_uid": doc_uid,
                    "chunk_index": chunk_idx,
                    "text": text[:2000],
                }
                if payload_extra:
                    payload.update(payload_extra)
                points.append(
                    qm.PointStruct(id=point_id, vector=vec, payload=payload)
                )

            client.upsert(collection_name=collection_name, points=points)
            total += len(points)

        return total
    except Exception as e:
        logger.error("Qdrant upsert failed collection=%s doc=%s: %s", collection_name, doc_uid, e)
        return 0


# ---------------------------------------------------------------------------
# PDF 유틸
# ---------------------------------------------------------------------------

def _pdf_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "raw" / "knowledge"


def _pdf_dir(category: str) -> Path:
    p = _pdf_root() / category
    p.mkdir(parents=True, exist_ok=True)
    return p


def _extract_pdf_text(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        try:
            from PyPDF2 import PdfReader  # type: ignore
        except ImportError:
            logger.error("pypdf/PyPDF2 not installed")
            return ""
    try:
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        return "\n".join(pages).strip()
    except Exception as e:
        logger.error("PDF text extraction failed %s: %s", pdf_path, e)
        return ""


def _sanitize(text: str | None) -> str:
    if not text:
        return ""
    return text.replace("\x00", "")


def _chunk_text(text: str, size: int = 1500, overlap: int = 200) -> list[str]:
    if not text:
        return []
    step = max(1, size - overlap)
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def _doc_uid(source: str, url: str, extra: str = "") -> str:
    base = f"{source}|{url}|{extra}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _doc_has_chunks(doc_uid: str) -> bool:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM tech_document_chunks WHERE doc_uid=%s LIMIT 1", (doc_uid,))
            return cur.fetchone() is not None


def _upsert_doc(
    doc_uid: str,
    source: str,
    source_type: str,
    title: str,
    url: str | None,
    published_at: datetime | None,
    summary: str,
    tags: list[str],
    confidence: float,
    extra: dict,
    content: str | None = None,
) -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tech_documents(
                  doc_uid, source, source_type, title, url, published_at, collected_at,
                  summary, content, tags, confidence, extra
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s::jsonb
                )
                ON CONFLICT (doc_uid)
                DO UPDATE SET
                  title = EXCLUDED.title,
                  summary = COALESCE(EXCLUDED.summary, tech_documents.summary),
                  content = COALESCE(EXCLUDED.content, tech_documents.content),
                  tags = EXCLUDED.tags,
                  confidence = EXCLUDED.confidence,
                  extra = tech_documents.extra || EXCLUDED.extra,
                  collected_at = EXCLUDED.collected_at
                """,
                (
                    doc_uid,
                    source,
                    source_type,
                    _sanitize(title) or "",
                    url,
                    published_at,
                    _now_utc(),
                    _sanitize(summary) or "",
                    _sanitize(content),
                    tags,
                    confidence,
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        conn.commit()


def _upsert_content_and_chunks(doc_uid: str, text: str, chunks: list[str], extra: dict) -> None:
    """full_text와 청크를 DB에 저장."""
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE tech_documents
                SET content = %s,
                    extra = extra || %s::jsonb,
                    collected_at = %s
                WHERE doc_uid = %s
                """,
                (_sanitize(text) or "", json.dumps(extra, ensure_ascii=False), _now_utc(), doc_uid),
            )
            cur.execute("DELETE FROM tech_document_chunks WHERE doc_uid=%s", (doc_uid,))
            for idx, chunk in enumerate(chunks):
                cur.execute(
                    """
                    INSERT INTO tech_document_chunks(
                      doc_uid, chunk_index, chunk_text, char_len, token_estimate, created_at, extra
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                    ON CONFLICT (doc_uid, chunk_index)
                    DO UPDATE SET
                      chunk_text = EXCLUDED.chunk_text,
                      char_len = EXCLUDED.char_len,
                      token_estimate = EXCLUDED.token_estimate,
                      created_at = EXCLUDED.created_at
                    """,
                    (
                        doc_uid, idx, _sanitize(chunk) or "",
                        len(chunk), max(1, len(chunk) // 4),
                        _now_utc(), json.dumps({}, ensure_ascii=False),
                    ),
                )
        conn.commit()


def _download_file(url: str, dest: Path, timeout: float = 60.0) -> bool:
    """URL → 파일 다운로드. 이미 있으면 skip."""
    if dest.exists() and dest.stat().st_size > 1024:
        logger.info("Already exists: %s", dest.name)
        return True
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; SemiconductorAgentBot/1.0; "
                "+https://github.com/semiconductor-agent)"
            )
        }
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            r = client.get(url)
            r.raise_for_status()
            dest.write_bytes(r.content)
        logger.info("Downloaded: %s (%d bytes)", dest.name, len(r.content))
        return True
    except Exception as e:
        logger.warning("Download failed %s: %s", url, e)
        if dest.exists():
            dest.unlink()
        return False


def _process_knowledge_pdf(
    doc_uid: str,
    pdf_path: Path,
    collection_name: str,
    payload_extra: dict | None = None,
    force: bool = False,
) -> int:
    """PDF 텍스트 추출 → 청크 → DB 저장 → Qdrant 업서트. 청크 수 반환."""
    if not force and _doc_has_chunks(doc_uid):
        logger.info("Chunks already exist, skipping: %s", doc_uid)
        return 0

    text = _extract_pdf_text(pdf_path)
    if not text:
        logger.warning("Empty PDF text: %s", pdf_path)
        return 0

    chunk_size = _env_int("KNOWLEDGE_CHUNK_SIZE", 1500)
    overlap = _env_int("KNOWLEDGE_CHUNK_OVERLAP", 200)
    chunks = _chunk_text(text, chunk_size, overlap)
    if not chunks:
        return 0

    _upsert_content_and_chunks(
        doc_uid, text, chunks,
        extra={
            "pdf_path": str(pdf_path),
            "chunk_count": len(chunks),
            "chunk_size": chunk_size,
            "chunk_overlap": overlap,
            "text_char_len": len(text),
        },
    )

    if _env_bool("KNOWLEDGE_ENABLE_EMBED", True):
        n_vec = _upsert_qdrant_chunks(collection_name, doc_uid, chunks, payload_extra)
        logger.info("Qdrant upsert done: %d vectors → %s", n_vec, collection_name)

    return len(chunks)


# ===========================================================================
# A. IRDS (IEEE International Roadmap for Devices and Systems)
# ===========================================================================

# IRDS Focus Team 약어 → 전체 제목
IRDS_FOCUS_TEAMS = {
    "ES":               "Executive Summary",
    "MM":               "More Moore",
    "MoreThanMoore":    "More Than Moore",
    "PP":               "Packaging & Integration",
    "YE":               "Yield Enhancement",
    "MtM":              "Memory and Storage",
    "FI":               "Factory Integration",
    "ERM":              "Emerging Research Materials",
    "ERD":              "Emerging Research Devices",
    "IEF":              "Integrated Electronic Failure Analysis",
    "SC":               "Semiconductor Component",
}

# 최신 에디션 연도 목록 (최신 → 구버전 순서로 시도)
IRDS_EDITIONS = [2024, 2023, 2022]

# 공개 PDF URL 패턴
IRDS_URL_TEMPLATE = "https://irds.ieee.org/images/files/pdf/{year}/{year}IRDS_{ft}.pdf"

# IEEE IRDS 공개 RSS 피드 및 기술 기사 페이지
IRDS_RSS_URL = "https://irds.ieee.org/feed/"
IRDS_ARTICLES_URL = "https://irds.ieee.org/technical-articles/"
IRDS_EDITIONS_URL_TEMPLATE = "https://irds.ieee.org/editions/{year}"


def _collect_irds_rss() -> int:
    """IRDS 웹사이트 RSS 피드 수집 — 기술 기사/뉴스."""
    entries = _parse_rss_feed(IRDS_RSS_URL)
    inserted = 0
    for entry in entries:
        title = entry["title"]
        link = entry["link"]
        summary = entry["summary"]
        if not link:
            continue
        uid = _doc_uid("irds", link)
        pub_at = _to_dt(entry["published"])
        tags = ["irds", "roadmap", "technical_article"]
        # 관련 FT 태그 자동 추가
        text_lower = f"{title} {summary}".lower()
        for ft, ft_title in IRDS_FOCUS_TEAMS.items():
            if ft.lower() in text_lower or ft_title.lower()[:10] in text_lower:
                tags.append(ft.lower())

        _upsert_doc(
            doc_uid=uid,
            source="irds",
            source_type="irds_article",
            title=title or link,
            url=link,
            published_at=pub_at,
            summary=summary[:2000] if summary else "",
            tags=list(set(tags)),
            confidence=0.90,
            extra={"feed_url": IRDS_RSS_URL},
        )
        if summary and _env_bool("KNOWLEDGE_ENABLE_EMBED", True) and not _doc_has_chunks(uid):
            chunks = _chunk_text(summary, 800, 100)
            if chunks:
                _upsert_content_and_chunks(uid, summary, chunks, extra={"source": "irds_rss"})
                _upsert_qdrant_chunks("irds_chunks", uid, chunks, payload_extra={"source": "irds"})
        inserted += 1
    return inserted


def _collect_irds_edition_metadata(year: int) -> None:
    """IRDS 에디션 페이지 메타데이터 저장 + Focus Team 카탈로그 구축."""
    url = IRDS_EDITIONS_URL_TEMPLATE.format(year=year)
    uid = _doc_uid("irds", url)
    published_at = datetime(year, 11, 1, tzinfo=timezone.utc)

    # 에디션 카탈로그 문서 저장
    all_ft_desc = "; ".join(f"{ft}: {title}" for ft, title in IRDS_FOCUS_TEAMS.items())
    _upsert_doc(
        doc_uid=uid,
        source="irds",
        source_type="irds",
        title=f"IEEE IRDS {year} Edition — International Roadmap for Devices and Systems",
        url=url,
        published_at=published_at,
        summary=(
            f"IEEE IRDS {year} annual roadmap publication covering: {all_ft_desc}. "
            "The International Roadmap for Devices and Systems provides 15-year technology "
            "outlook for semiconductor devices, memory, logic scaling, packaging, and yield."
        ),
        tags=["irds", "roadmap", str(year), "technical_standard", "semiconductor_roadmap"],
        confidence=0.97,
        extra={"year": year, "focus_teams": list(IRDS_FOCUS_TEAMS.keys()), "access": "paid_membership"},
    )

    # 각 FT별 개별 메타데이터 레코드
    for ft, ft_title in IRDS_FOCUS_TEAMS.items():
        ft_uid = _doc_uid("irds", f"irds:{year}:{ft}")
        ft_url = IRDS_URL_TEMPLATE.format(year=year, ft=ft)
        _upsert_doc(
            doc_uid=ft_uid,
            source="irds",
            source_type="irds",
            title=f"IRDS {year} — {ft_title} (Focus Team: {ft})",
            url=ft_url,
            published_at=published_at,
            summary=(
                f"IEEE IRDS {year} Focus Team Report on {ft_title}. "
                f"Provides technology roadmap and projections for {ft_title.lower()} "
                f"in the semiconductor industry through 2037."
            ),
            tags=["irds", "roadmap", ft.lower(), str(year), "technical_standard"],
            confidence=0.97,
            extra={
                "year": year,
                "focus_team": ft,
                "focus_team_title": ft_title,
                "access": "paid_membership",
                "note": "Full PDF requires IEEE IRDS membership. Metadata only.",
            },
        )
    logger.info("IRDS metadata stored: year=%d focus_teams=%d", year, len(IRDS_FOCUS_TEAMS))


def _try_ingest_local_irds_pdfs() -> int:
    """
    사용자가 수동으로 data/raw/knowledge/irds/에 넣은 PDF 파일 인덱싱.
    파일명 컨벤션: IRDS_{year}_{FT}.pdf (예: IRDS_2023_MM.pdf)
    """
    irds_dir = _pdf_dir("irds")
    pdf_files = list(irds_dir.glob("*.pdf"))
    if not pdf_files:
        return 0

    pat = re.compile(r"IRDS_(\d{4})_(.+)\.pdf", re.IGNORECASE)
    ingested = 0
    for pdf_path in pdf_files:
        m = pat.match(pdf_path.name)
        if not m:
            logger.info("Skipping non-IRDS PDF: %s", pdf_path.name)
            continue
        year, ft = int(m.group(1)), m.group(2)
        ft_title = IRDS_FOCUS_TEAMS.get(ft, ft)
        uid = _doc_uid("irds", f"irds:{year}:{ft}")
        published_at = datetime(year, 11, 1, tzinfo=timezone.utc)

        # DB 메타데이터 보장
        _upsert_doc(
            doc_uid=uid,
            source="irds",
            source_type="irds",
            title=f"IRDS {year} — {ft_title}",
            url=IRDS_URL_TEMPLATE.format(year=year, ft=ft),
            published_at=published_at,
            summary=f"IEEE IRDS {year} Focus Team Report: {ft_title}.",
            tags=["irds", "roadmap", ft.lower(), str(year)],
            confidence=0.97,
            extra={"year": year, "focus_team": ft, "pdf_filename": pdf_path.name},
        )

        n_chunks = _process_knowledge_pdf(
            doc_uid=uid,
            pdf_path=pdf_path,
            collection_name="irds_chunks",
            payload_extra={"source": "irds", "year": year, "focus_team": ft},
        )
        if n_chunks > 0:
            ingested += 1
            logger.info("IRDS local PDF ingested: %s chunks=%d", pdf_path.name, n_chunks)

    return ingested


def download_irds_new_edition() -> None:
    """
    연간 IRDS 업데이트 (11월 스케줄).
    - PDF 다운로드는 IEEE 멤버십 필요 → 메타데이터 카탈로그 저장
    - IRDS 웹사이트 RSS 기사 수집
    - 로컬에 넣어둔 PDF 있으면 자동 인덱싱
    """
    logger.info("=== IRDS: download_irds_new_edition START ===")
    target_year = _env_int("IRDS_TARGET_YEAR", IRDS_EDITIONS[0])

    # 1) 에디션 메타데이터 카탈로그
    _collect_irds_edition_metadata(target_year)

    # 2) RSS 기사 수집
    rss_count = _collect_irds_rss()
    logger.info("IRDS RSS articles collected: %d", rss_count)

    # 3) 로컬 PDF 인덱싱 (사용자가 수동 추가한 경우)
    local_count = _try_ingest_local_irds_pdfs()
    logger.info("IRDS local PDFs ingested: %d", local_count)

    logger.info("=== IRDS done: year=%d rss=%d local_pdf=%d ===", target_year, rss_count, local_count)


def ingest_irds_all_editions() -> None:
    """초기 부트스트랩용: 모든 에디션 메타데이터 + RSS + 로컬 PDF."""
    logger.info("=== IRDS: ingest_all_editions START ===")

    for year in IRDS_EDITIONS:
        _collect_irds_edition_metadata(year)

    rss_count = _collect_irds_rss()
    logger.info("IRDS RSS articles: %d", rss_count)

    local_count = _try_ingest_local_irds_pdfs()
    logger.info("IRDS local PDFs ingested: %d", local_count)

    logger.info("=== IRDS: ingest_all_editions DONE (metadata=%d editions × %d FTs, rss=%d, local=%d) ===",
                len(IRDS_EDITIONS), len(IRDS_FOCUS_TEAMS), rss_count, local_count)


# ===========================================================================
# B. JEDEC 표준 (무료 공개 + 메타데이터 카탈로그)
# ===========================================================================

# 알려진 무료 JEDEC 표준 + 다운로드 URL
# 표준에 따라 무료/유료가 다름; 여기엔 공식 무료 공개본만 포함
JEDEC_FREE_STANDARDS: list[dict] = [
    {
        "std_no": "JESD79-5B",
        "title": "DDR5 SDRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd79-5b",
        "tags": ["ddr5", "sdram", "memory_standard", "jedec"],
        "year": 2024,
    },
    {
        "std_no": "JESD209-5B",
        "title": "LPDDR5/LPDDR5X SDRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd209-5b",
        "tags": ["lpddr5", "mobile_dram", "memory_standard", "jedec"],
        "year": 2023,
    },
    {
        "std_no": "JESD235D",
        "title": "High Bandwidth Memory (HBM) DRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd235d",
        "tags": ["hbm", "hbm2", "memory_standard", "jedec", "high_bandwidth"],
        "year": 2021,
    },
    {
        "std_no": "JESD238A",
        "title": "High Bandwidth Memory (HBM3) DRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd238a",
        "tags": ["hbm3", "memory_standard", "jedec", "high_bandwidth"],
        "year": 2023,
    },
    {
        "std_no": "JESD79-4C",
        "title": "DDR4 SDRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd79-4c",
        "tags": ["ddr4", "sdram", "memory_standard", "jedec"],
        "year": 2020,
    },
    {
        "std_no": "JESD94B",
        "title": "Application Specific Qualification Using Sample Sizes",
        "url": "https://www.jedec.org/standards-technology/docs/jesd94b",
        "tags": ["reliability", "qualification", "jedec"],
        "year": 2019,
    },
    {
        "std_no": "JESD22-A108G",
        "title": "Temperature, Bias, and Operating Life",
        "url": "https://www.jedec.org/standards-technology/docs/jesd22-a108g",
        "tags": ["reliability", "qualification", "jedec", "burn_in"],
        "year": 2022,
    },
]

# 접근 불가 PDF는 메타데이터만 저장 (무료 다운로드 안 되는 표준)
JEDEC_METADATA_ONLY: list[dict] = [
    {
        "std_no": "JESD239",
        "title": "High Bandwidth Memory (HBM3E) DRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd239",
        "tags": ["hbm3e", "memory_standard", "jedec", "high_bandwidth"],
        "year": 2024,
    },
    {
        "std_no": "JESD240",
        "title": "High Bandwidth Memory (HBM4) DRAM",
        "url": "https://www.jedec.org/standards-technology/docs/jesd240",
        "tags": ["hbm4", "memory_standard", "jedec", "high_bandwidth"],
        "year": 2025,
    },
    {
        "std_no": "JESD79-5C",
        "title": "DDR5 SDRAM (Rev C)",
        "url": "https://www.jedec.org/standards-technology/docs/jesd79-5c",
        "tags": ["ddr5", "sdram", "memory_standard", "jedec"],
        "year": 2025,
    },
    {
        "std_no": "JESD300-1",
        "title": "Compute Express Link (CXL) 2.0",
        "url": "https://www.jedec.org/standards-technology/docs/jesd300-1",
        "tags": ["cxl", "interconnect", "jedec", "memory_expansion"],
        "year": 2023,
    },
]


def _jedec_doc_uid(std_no: str) -> str:
    return _doc_uid("jedec", f"jedec:{std_no}")


def download_jedec_updates() -> None:
    """JEDEC 표준 업데이트 수집 (월 1회)."""
    logger.info("=== JEDEC: download_jedec_updates START ===")
    dest_dir = _pdf_dir("jedec")

    # 1) 무료 공개 표준: PDF 다운로드 + 텍스트 인덱싱
    free_ok = 0
    for std in JEDEC_FREE_STANDARDS:
        uid = _jedec_doc_uid(std["std_no"])
        filename = f"JEDEC_{std['std_no'].replace('/', '_')}.pdf"
        dest = dest_dir / filename
        pub_at = datetime(std.get("year", 2020), 1, 1, tzinfo=timezone.utc)

        # DB 메타데이터 먼저 저장
        _upsert_doc(
            doc_uid=uid,
            source="jedec",
            source_type="jedec",
            title=f"JEDEC {std['std_no']} — {std['title']}",
            url=std["url"],
            published_at=pub_at,
            summary=(
                f"JEDEC standard {std['std_no']}: {std['title']}. "
                "Industry standard specification for semiconductor memory interface."
            ),
            tags=std["tags"],
            confidence=0.97,
            extra={"std_no": std["std_no"], "access": "free"},
        )

        # PDF 다운로드 시도
        pdf_url = str(std.get("pdf_url") or "").strip()
        downloaded = False
        if pdf_url:
            downloaded = _download_file(pdf_url, dest, timeout=60.0)
            time.sleep(1.0)

        if downloaded:
            n_chunks = _process_knowledge_pdf(
                doc_uid=uid,
                pdf_path=dest,
                collection_name="jedec_chunks",
                payload_extra={"source": "jedec", "std_no": std["std_no"]},
            )
            logger.info("JEDEC indexed: %s chunks=%d", std["std_no"], n_chunks)
            free_ok += 1
        else:
            logger.info("JEDEC metadata saved without PDF indexing: %s", std["std_no"])

    # 2) 유료/접근 불가 표준: 메타데이터만 저장
    for std in JEDEC_METADATA_ONLY:
        uid = _jedec_doc_uid(std["std_no"])
        pub_at = datetime(std.get("year", 2024), 1, 1, tzinfo=timezone.utc)
        _upsert_doc(
            doc_uid=uid,
            source="jedec",
            source_type="jedec",
            title=f"JEDEC {std['std_no']} — {std['title']}",
            url=std["url"],
            published_at=pub_at,
            summary=(
                f"JEDEC standard {std['std_no']}: {std['title']}. "
                "Access requires JEDEC membership or purchase."
            ),
            tags=std["tags"],
            confidence=0.95,
            extra={"std_no": std["std_no"], "access": "paid_metadata_only"},
        )
        logger.info("JEDEC metadata saved: %s", std["std_no"])

    logger.info("=== JEDEC done: free_indexed=%d metadata_only=%d ===", free_ok, len(JEDEC_METADATA_ONLY))


# ===========================================================================
# C. Hot Chips / 주요 반도체 학술 행사 공개 자료
# ===========================================================================

# 공개된 Hot Chips 발표 자료 (IEEE open-access + 공식 아카이브)
HOTCHIPS_ARCHIVE_URLS: list[dict] = [
    {
        "year": 2024,
        "session": "HC36",
        "title": "Hot Chips 2024 (HC36) — AI Accelerators, HBM, Advanced Packaging",
        "url": "https://ieeexplore.ieee.org/xpl/conhome/10664316/proceeding",
        "source_type": "hotchips",
        "tags": ["hotchips", "hc36", "2024", "ai_chip", "hbm", "advanced_packaging"],
        "confidence": 0.88,
    },
    {
        "year": 2023,
        "session": "HC35",
        "title": "Hot Chips 2023 (HC35) — Custom Silicon, Memory Systems",
        "url": "https://ieeexplore.ieee.org/xpl/conhome/10242200/proceeding",
        "source_type": "hotchips",
        "tags": ["hotchips", "hc35", "2023", "ai_chip", "hbm"],
        "confidence": 0.88,
    },
    {
        "year": 2022,
        "session": "HC34",
        "title": "Hot Chips 2022 (HC34) — Chiplets, Memory, AI Processors",
        "url": "https://ieeexplore.ieee.org/xpl/conhome/9902671/proceeding",
        "source_type": "hotchips",
        "tags": ["hotchips", "hc34", "2022", "memory", "chiplet"],
        "confidence": 0.85,
    },
]

# FMS (Flash Memory Summit / Storage Developer Conference) 공개 발표자료 피드
FMS_URLS: list[dict] = [
    {
        "year": 2024,
        "title": "Flash Memory Summit 2024 Presentations",
        "url": "https://www.flashmemorysummit.com/English/Conference/Proceedings.html",
        "source_type": "fms",
        "tags": ["fms", "nand", "flash_memory", "storage", "2024"],
        "confidence": 0.82,
    },
]

# IEDM / ISSCC / VLSI — IEEE open-access 논문 arXiv 미러 매핑
# (실제 논문은 paper_collector.py arXiv 파이프라인이 처리하므로 여기선 메타 정보만)
CONFERENCE_METADATA: list[dict] = [
    {
        "source": "iedm",
        "title": "IEEE International Electron Devices Meeting (IEDM) 2024",
        "url": "https://www.ieee-iedm.org/program/",
        "published_at": datetime(2024, 12, 7, tzinfo=timezone.utc),
        "tags": ["iedm", "2024", "semiconductor_device", "transistor", "memory"],
        "summary": (
            "IEDM 2024 covers advanced semiconductor device engineering including "
            "logic scaling (GAA nanosheet), DRAM/NAND 3D integration, "
            "advanced packaging, and neuromorphic devices."
        ),
        "confidence": 0.92,
    },
    {
        "source": "isscc",
        "title": "IEEE International Solid-State Circuits Conference (ISSCC) 2025",
        "url": "https://isscc.org/2025/",
        "published_at": datetime(2025, 2, 16, tzinfo=timezone.utc),
        "tags": ["isscc", "2025", "circuit", "hbm", "lpddr5"],
        "summary": (
            "ISSCC 2025 Memory sub-conference includes HBM3E/4 controller circuits, "
            "DDR5/LPDDR5 PHY design, NAND 3D stacking circuit challenges."
        ),
        "confidence": 0.90,
    },
    {
        "source": "vlsi_symposium",
        "title": "IEEE VLSI Technology & Circuits Symposium 2024",
        "url": "https://vlsisymposium.org/2024/",
        "published_at": datetime(2024, 6, 16, tzinfo=timezone.utc),
        "tags": ["vlsi", "2024", "cmos_scaling", "memory", "advanced_node"],
        "summary": (
            "VLSI 2024 Technology symposium covers GAA nanosheet process, "
            "back-side power delivery, 3D DRAM integration, and EUV high-NA."
        ),
        "confidence": 0.90,
    },
]


def _collect_hotchips_metadata(entry: dict) -> None:
    """Hot Chips 프로그램 페이지 메타데이터를 DB에 저장하고 공개 자료 파싱 시도."""
    uid = _doc_uid(entry.get("source_type", "hotchips"), entry["url"])
    pub_at = datetime(entry["year"], 8, 20, tzinfo=timezone.utc)

    _upsert_doc(
        doc_uid=uid,
        source=entry.get("source_type", "hotchips"),
        source_type=entry.get("source_type", "hotchips"),
        title=entry["title"],
        url=entry["url"],
        published_at=pub_at,
        summary=(
            f"{entry['title']}. Annual industry conference covering "
            "custom silicon, AI accelerators, memory systems, and advanced packaging."
        ),
        tags=entry["tags"],
        confidence=entry["confidence"],
        extra={"year": entry["year"], "session": entry.get("session", "")},
    )

    # 공개 HTML에서 발표 제목 크롤링 시도
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; SemiconductorAgentBot/1.0)"}
        with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
            r = client.get(entry["url"])
            r.raise_for_status()
            html = r.text

        # 발표 제목 패턴 추출
        title_pat = re.compile(
            r'<(?:h[23456]|td|li)[^>]*>\s*([A-Z][^<]{20,200}(?:Memory|HBM|DRAM|NAND|'
            r'Package|Packaging|Chiplet|Die|Process|EUV|AI|GPU|NPU|Accelerator)[^<]{0,100})\s*</',
            re.IGNORECASE,
        )
        found_titles = list(set(title_pat.findall(html)))[:30]

        if found_titles:
            summary_addon = " | ".join(found_titles[:5])
            with get_pg_conn() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE tech_documents
                        SET extra = extra || %s::jsonb
                        WHERE doc_uid = %s
                        """,
                        (json.dumps({"scraped_titles": found_titles}, ensure_ascii=False), uid),
                    )
                conn.commit()
            logger.info("HotChips titles scraped: %d | %s", len(found_titles), entry["session"])
    except Exception as e:
        logger.warning("HotChips scrape failed %s: %s", entry["url"], e)


def collect_hotchips_materials() -> None:
    """Hot Chips + FMS + 학술 행사 메타데이터 수집 (연 1회, 8월)."""
    logger.info("=== HotChips/FMS: collect_hotchips_materials START ===")

    for entry in HOTCHIPS_ARCHIVE_URLS:
        _collect_hotchips_metadata(entry)
        time.sleep(2.0)

    for entry in FMS_URLS:
        uid = _doc_uid("fms", entry["url"])
        pub_at = datetime(entry["year"], 8, 1, tzinfo=timezone.utc)
        _upsert_doc(
            doc_uid=uid,
            source="fms",
            source_type=entry.get("source_type", "fms"),
            title=entry["title"],
            url=entry["url"],
            published_at=pub_at,
            summary=(
                f"{entry['title']}. Flash Memory Summit covers NAND flash scaling, "
                "3D NAND architecture, SSD controller design, and storage systems."
            ),
            tags=entry["tags"],
            confidence=entry["confidence"],
            extra={"year": entry["year"]},
        )
        time.sleep(1.0)

    for meta in CONFERENCE_METADATA:
        uid = _doc_uid(meta["source"], meta["url"])
        _upsert_doc(
            doc_uid=uid,
            source=meta["source"],
            source_type="conference_metadata",
            title=meta["title"],
            url=meta["url"],
            published_at=meta["published_at"],
            summary=meta["summary"],
            tags=meta["tags"],
            confidence=meta["confidence"],
            extra={"access": "metadata_only"},
        )

    logger.info("=== HotChips/FMS/Conf done ===")


# ===========================================================================
# D. 반도체 장비사 기술 문서 RSS / 블로그
# ===========================================================================

EQUIP_SOURCES: dict[str, dict] = {
    # 반도체 장비/소재 전문 기술 미디어 (실제 RSS 동작 확인된 소스)
    "eetasia": {
        "type": "rss",
        "url": "https://www.eetasia.com/feed/",
        "tags": ["eetasia", "semiconductor", "equipment", "process", "memory"],
        "confidence": 0.80,
    },
    "edn_semiconductor": {
        "type": "rss",
        "url": "https://www.edn.com/feed/",
        "tags": ["edn", "semiconductor", "ic_design", "memory", "process"],
        "confidence": 0.78,
    },
    "semiconductor_engineering": {
        # semiengineering은 news_collector.py에도 있지만 여기선 장비/공정 기사 필터
        "type": "rss",
        "url": "https://semiengineering.com/feed/",
        "tags": ["semiconductor_engineering", "equipment", "process", "yield", "packaging"],
        "confidence": 0.85,
    },
    "irds_rss": {
        "type": "rss",
        "url": "https://irds.ieee.org/feed/",
        "tags": ["irds", "roadmap", "semiconductor", "technology"],
        "confidence": 0.92,
    },
    "semiwiki": {
        "type": "rss",
        "url": "https://semiwiki.com/feed/",
        "tags": ["semiwiki", "eda", "ip", "process", "foundry", "memory"],
        "confidence": 0.78,
    },
}

SEMI_BLOG_SOURCES: dict[str, dict] = {
    # 반도체 기업 기술 블로그 (실제 동작하는 RSS)
    "skhynix_newsroom": {
        # news_collector.py에도 있지만 기술 심층 기사 필터 강화
        "type": "rss",
        "url": "https://news.skhynix.com/feed/",
        "tags": ["sk_hynix", "hbm", "dram", "nand", "memory", "semi_blog"],
        "confidence": 0.90,
    },
    "chips_and_cheese": {
        "type": "rss",
        "url": "https://chipsandcheese.substack.com/feed",
        "tags": ["chips_and_cheese", "microarchitecture", "memory", "analysis", "semi_blog"],
        "confidence": 0.85,
    },
}

# 반도체 관련 키워드 (장비/공정 도메인 포함)
EQUIP_SEMI_KEYWORDS = {
    "euv", "duv", "lithography", "euv scanner", "high-na",
    "etch", "deposition", "cvd", "pvd", "ald", "cmp",
    "yield", "defect", "metrology", "inspection", "overlay",
    "wafer", "process", "node", "advanced node",
    "hbm", "dram", "nand", "flash", "memory",
    "packaging", "chiplet", "hybrid bonding", "tc bonding", "thermo-compression",
    "cowos", "tsv", "advanced packaging",
    "samsung", "sk hynix", "micron", "tsmc", "intel",
    "hanmi", "한미",
}


def _semi_relevant(title: str, summary: str) -> tuple[bool, list[str]]:
    text = f"{title} {summary}".lower()
    hits = [kw for kw in EQUIP_SEMI_KEYWORDS if kw in text]
    return len(hits) > 0, hits


def _parse_rss_feed(url: str) -> list[dict]:
    try:
        import feedparser  # type: ignore
        feed = feedparser.parse(url)
        entries = getattr(feed, "entries", []) or []
        results = []
        for e in entries:
            results.append({
                "title": str(getattr(e, "title", "") or "").strip(),
                "link": str(getattr(e, "link", "") or "").strip(),
                "summary": str(getattr(e, "summary", "") or "").strip(),
                "published": str(getattr(e, "published", "") or "").strip(),
            })
        return results
    except Exception as ex:
        logger.warning("RSS parse failed %s: %s", url, ex)
        return []


def _to_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        from email.utils import parsedate_to_datetime
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None


def _collect_rss_source(
    source_key: str,
    cfg: dict,
    source_type: str,
    qdrant_collection: str = "tech_blog_chunks",
) -> int:
    entries = _parse_rss_feed(cfg["url"])
    inserted = 0
    for entry in entries:
        title = entry["title"]
        link = entry["link"]
        summary = entry["summary"]
        pub_str = entry["published"]
        if not link:
            continue
        ok, kw_hits = _semi_relevant(title, summary)
        if not ok:
            continue
        pub_at = _to_dt(pub_str)
        uid = _doc_uid(source_key, link, pub_str)
        tags = cfg["tags"] + kw_hits
        _upsert_doc(
            doc_uid=uid,
            source=source_key,
            source_type=source_type,
            title=title or link,
            url=link,
            published_at=pub_at,
            summary=summary[:2000],
            tags=list(set(tags)),
            confidence=cfg["confidence"],
            extra={"published_raw": pub_str, "keyword_hits": kw_hits},
        )
        # 요약문 청크 + 벡터화 (본문은 RSS에 없으므로 summary 기반)
        if summary and _env_bool("KNOWLEDGE_ENABLE_EMBED", True):
            chunks = _chunk_text(summary, 800, 100)
            if chunks and not _doc_has_chunks(uid):
                _upsert_content_and_chunks(uid, summary, chunks, extra={"source": source_key})
                _upsert_qdrant_chunks(
                    qdrant_collection, uid, chunks,
                    payload_extra={"source": source_key, "source_type": source_type},
                )
        inserted += 1
    logger.info("Equip/blog source done: %s inserted=%d", source_key, inserted)
    return inserted


def collect_equipment_docs() -> None:
    """반도체 장비사 기술 블로그 RSS 수집 (news_collector 보완)."""
    logger.info("=== Equipment docs: START ===")
    total = 0
    for key, cfg in EQUIP_SOURCES.items():
        n = _collect_rss_source(key, cfg, source_type="equip_doc", qdrant_collection="tech_blog_chunks")
        total += n
        time.sleep(1.5)
    logger.info("=== Equipment docs done: total=%d ===", total)


def collect_semi_vendor_blogs() -> None:
    """삼성/Micron/한미 기술 블로그 수집 (tech_blog_chunks)."""
    logger.info("=== Semi vendor blogs: START ===")
    total = 0
    for key, cfg in SEMI_BLOG_SOURCES.items():
        n = _collect_rss_source(key, cfg, source_type="semi_blog", qdrant_collection="tech_blog_chunks")
        total += n
        time.sleep(1.5)
    logger.info("=== Semi vendor blogs done: total=%d ===", total)


# ===========================================================================
# E. 부트스트랩: 첫 실행 시 일괄 수집
# ===========================================================================

def bootstrap_knowledge_layer() -> None:
    """
    최초 1회 실행 — 모든 기술 지식 레이어 일괄 수집.
    (이후엔 jobs.py 스케줄러가 주기적으로 업데이트)
    """
    logger.info("=" * 60)
    logger.info("KNOWLEDGE LAYER BOOTSTRAP START")
    logger.info("=" * 60)

    logger.info("[1/5] IRDS all editions …")
    ingest_irds_all_editions()

    logger.info("[2/5] JEDEC standards …")
    download_jedec_updates()

    logger.info("[3/5] Hot Chips / FMS / Conference metadata …")
    collect_hotchips_materials()

    logger.info("[4/5] Equipment docs …")
    collect_equipment_docs()

    logger.info("[5/5] Semi vendor blogs …")
    collect_semi_vendor_blogs()

    logger.info("=" * 60)
    logger.info("KNOWLEDGE LAYER BOOTSTRAP COMPLETE")
    logger.info("=" * 60)
