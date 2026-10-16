from app.config import settings

try:
    from psycopg import connect as pg_connect  # psycopg3
except Exception:  # pragma: no cover
    import psycopg2

    def pg_connect(dsn: str):
        return psycopg2.connect(dsn)


def get_pg_conn():
    return pg_connect(settings.postgres_dsn)
