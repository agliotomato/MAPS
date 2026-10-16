"""
TimescaleDB 커넥션 풀 관리
Thread-safe ThreadedConnectionPool 기반 싱글톤 풀을 제공합니다.

필요 환경변수:
    TIMESCALE_DSN: postgresql://user:password@host:5432/macro_db
    TIMESCALE_POOL_MIN: 커넥션 풀 최소 크기 (기본값: 1)
    TIMESCALE_POOL_MAX: 커넥션 풀 최대 크기 (기본값: 10)
    TIMESCALE_SCHEMA: 사용할 기존 스키마 (기본값: public)
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Generator

import psycopg2
from psycopg2 import pool as pg_pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_pool: pg_pool.ThreadedConnectionPool | None = None
_pool_lock = Lock()


# ── 커넥션 풀 관리 ─────────────────────────────────────────────────────

def _get_dsn() -> str:
    dsn = os.environ.get("TIMESCALE_DSN", "")
    if not dsn:
        raise EnvironmentError(
            "TIMESCALE_DSN 환경변수가 설정되지 않았습니다.\n"
            "예: TIMESCALE_DSN=postgresql://user:password@localhost:5432/macro_db"
        )
    return dsn


def get_pool() -> pg_pool.ThreadedConnectionPool:
    """
    싱글톤 ThreadedConnectionPool을 반환합니다.
    최초 호출 시 초기화, 이후 재사용됩니다. Thread-safe.
    """
    global _pool
    if _pool is not None and not _pool.closed:
        return _pool

    with _pool_lock:
        if _pool is not None and not _pool.closed:
            return _pool

        dsn = _get_dsn()
        schema = os.environ.get("TIMESCALE_SCHEMA", "public")
        if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", schema):
            raise ValueError("TIMESCALE_SCHEMA must be a lowercase SQL identifier")
        min_conn = int(os.environ.get("TIMESCALE_POOL_MIN", "1"))
        max_conn = int(os.environ.get("TIMESCALE_POOL_MAX", "10"))

        candidate = pg_pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            dsn,
            cursor_factory=RealDictCursor,  # 쿼리 결과를 dict로 반환
            connect_timeout=10,
            options=f"-c statement_timeout=30000 -c search_path={schema},public",
        )
        # PostgreSQL silently skips a missing/inaccessible schema. Refuse that
        # fallback before any query can create or read tables in another schema.
        try:
            conn = candidate.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT current_schema() AS schema")
                    if cur.fetchone()["schema"] != schema:
                        raise RuntimeError("Configured TIMESCALE_SCHEMA is unavailable")
            finally:
                conn.rollback()
                candidate.putconn(conn)
        except Exception:
            candidate.closeall()
            raise
        _pool = candidate
        logger.info(
            "TimescaleDB 커넥션 풀 초기화 (min=%d, max=%d, dsn=%s)",
            min_conn,
            max_conn,
            _mask_dsn(dsn),
        )
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg2.extensions.connection, None, None]:
    """
    풀에서 커넥션을 빌려 반환하는 컨텍스트 매니저.
    - 정상 종료: COMMIT
    - 예외 발생: ROLLBACK 후 재전파
    - finally: 항상 풀에 커넥션 반납

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(...)
    """
    p = get_pool()
    conn = p.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        p.putconn(conn)


def close_pool() -> None:
    """애플리케이션 종료 시 커넥션 풀을 안전하게 닫습니다."""
    global _pool
    with _pool_lock:
        if _pool and not _pool.closed:
            _pool.closeall()
            _pool = None
            logger.info("TimescaleDB 커넥션 풀 종료")


# ── 스키마 초기화 ──────────────────────────────────────────────────────

def initialize_schema(schema_path: Path | None = None) -> None:
    """
    schema.sql을 실행하여 hypertable과 인덱스를 생성합니다.
    멱등성 보장 (IF NOT EXISTS 조건으로 재실행 안전).

    Args:
        schema_path: schema.sql 경로 (None이면 패키지 내부 경로 자동 탐색)
    """
    path = schema_path or (Path(__file__).parent / "schema.sql")
    if not path.exists():
        raise FileNotFoundError(f"schema.sql을 찾을 수 없습니다: {path}")

    sql = path.read_text(encoding="utf-8")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'")
            if cur.fetchone() is None:
                cur.execute("CREATE EXTENSION timescaledb WITH SCHEMA public CASCADE")
            cur.execute(sql)

    logger.info("TimescaleDB 스키마 초기화 완료 (%s)", path.name)


def healthcheck() -> dict[str, str]:
    """DB 연결 상태 및 TimescaleDB 버전을 반환합니다."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT version(), extversion "
                    "FROM pg_extension WHERE extname = 'timescaledb'"
                )
                row = cur.fetchone()
                return {
                    "status": "ok",
                    "pg_version": row["version"][:40] if row else "unknown",
                    "timescaledb_version": row["extversion"] if row else "not installed",
                }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


def _mask_dsn(dsn: str) -> str:
    """로그 출력 시 비밀번호를 마스킹합니다."""
    try:
        import re
        return re.sub(r":[^:@/]+@", ":***@", dsn)
    except Exception:
        return "***"
