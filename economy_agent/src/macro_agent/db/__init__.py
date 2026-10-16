"""
macro_agent.db — 데이터베이스 레이어

TimescaleDB (정량): macro_agent.db.timescale
Vector DB    (정성): macro_agent.db.vector
파이프라인  (통합): macro_agent.db.pipeline
"""

from macro_agent.db.pipeline import IngestionResult, ingest_report, run_ingestion_pipeline

__all__ = [
    "run_ingestion_pipeline",
    "ingest_report",
    "IngestionResult",
]
