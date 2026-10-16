from __future__ import annotations

import json
from datetime import datetime, timezone

from app.db.postgres import get_pg_conn


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _chunk_text(text: str, size: int = 1200, overlap: int = 120) -> list[str]:
    text = (text or "").replace("\x00", "").strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(1, size - overlap)
    for start in range(0, len(text), step):
        chunk = text[start : start + size].strip()
        if chunk:
            chunks.append(chunk)
        if start + size >= len(text):
            break
    return chunks


def backfill_document_summary_chunks(limit: int = 1000) -> int:
    """
    Create searchable chunks from tech_documents.content/summary when no chunks exist.

    This makes lexical RAG usable immediately for metadata-only OpenAlex, official IR,
    and press documents even before expensive PDF/BGE indexing succeeds.
    """
    select_sql = """
        SELECT d.doc_uid, d.title, d.summary, d.content, d.source, d.source_type
        FROM tech_documents d
        WHERE NOT EXISTS (
            SELECT 1 FROM tech_document_chunks c WHERE c.doc_uid = d.doc_uid
        )
        AND COALESCE(NULLIF(d.content, ''), NULLIF(d.summary, '')) IS NOT NULL
        ORDER BY COALESCE(d.published_at, d.collected_at) DESC
        LIMIT %s
    """
    inserted = 0
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(select_sql, (limit,))
            rows = cur.fetchall()
            for doc_uid, title, summary, content, source, source_type in rows:
                text = f"{title or ''}\n\n{content or summary or ''}".strip()
                chunks = _chunk_text(text)
                for idx, chunk in enumerate(chunks):
                    cur.execute(
                        """
                        INSERT INTO tech_document_chunks(
                          doc_uid, chunk_index, chunk_text, char_len,
                          token_estimate, created_at, extra
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s::jsonb)
                        ON CONFLICT (doc_uid, chunk_index)
                        DO UPDATE SET
                          chunk_text=EXCLUDED.chunk_text,
                          char_len=EXCLUDED.char_len,
                          token_estimate=EXCLUDED.token_estimate,
                          created_at=EXCLUDED.created_at,
                          extra=EXCLUDED.extra
                        """,
                        (
                            doc_uid,
                            idx,
                            chunk,
                            len(chunk),
                            max(1, len(chunk) // 4),
                            _now_utc(),
                            json.dumps(
                                {
                                    "backfilled_from": "content_or_summary",
                                    "source": source,
                                    "source_type": source_type,
                                },
                                ensure_ascii=False,
                            ),
                        ),
                    )
                    inserted += 1
        conn.commit()
    return inserted
