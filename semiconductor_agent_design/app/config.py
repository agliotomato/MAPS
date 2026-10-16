import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv


load_dotenv()

_timescale_credentials = os.getenv("TIMESCALE_CREDENTIALS_PATH", "")
if _timescale_credentials:
    load_dotenv(Path(_timescale_credentials), override=False)


@dataclass(frozen=True)
class Settings:
    timescale_service_url: str = os.getenv("TIMESCALE_SERVICE_URL", "")
    postgres_dsn_env: str = os.getenv("POSTGRES_DSN", "")
    postgres_user: str = os.getenv("POSTGRES_USER", os.getenv("PGUSER", "sema"))
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", os.getenv("PGPASSWORD", "sema_pass"))
    postgres_db: str = os.getenv("POSTGRES_DB", os.getenv("PGDATABASE", "sema_db"))
    postgres_host: str = os.getenv("POSTGRES_HOST", os.getenv("PGHOST", "localhost"))
    postgres_port: int = int(os.getenv("POSTGRES_PORT", os.getenv("PGPORT", "5432")))
    postgres_sslmode: str = os.getenv("POSTGRES_SSLMODE", os.getenv("PGSSLMODE", "disable"))

    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))

    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", os.getenv("NEO4J_USERNAME", "neo4j"))
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j_password")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")

    tz: str = os.getenv("TZ", "Asia/Seoul")

    @property
    def postgres_dsn(self) -> str:
        if self.timescale_service_url:
            return self.timescale_service_url
        if self.postgres_dsn_env:
            return self.postgres_dsn_env
        return (
            f"postgresql://{quote_plus(self.postgres_user)}:{quote_plus(self.postgres_password)}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            f"?sslmode={self.postgres_sslmode}"
        )


settings = Settings()
