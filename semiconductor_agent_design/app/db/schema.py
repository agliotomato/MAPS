from app.db.postgres import get_pg_conn


POSTGRES_SCHEMA_STATEMENTS = [
    # Timescale extension may already exist or be managed by cloud provider.
    "CREATE EXTENSION IF NOT EXISTS timescaledb;",
    """
    CREATE TABLE IF NOT EXISTS companies (
      company_code TEXT PRIMARY KEY,
      company_name TEXT NOT NULL,
      market TEXT,
      country TEXT DEFAULT 'KR'
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS metric_observations (
      id BIGSERIAL PRIMARY KEY,
      company_code TEXT REFERENCES companies(company_code),
      domain TEXT NOT NULL,
      metric_name TEXT NOT NULL,
      metric_value DOUBLE PRECISION NOT NULL,
      unit TEXT,
      is_proxy BOOLEAN DEFAULT FALSE,
      proxy_for TEXT,
      observed_at TIMESTAMPTZ NOT NULL,
      published_at TIMESTAMPTZ NOT NULL,
      valid_from TIMESTAMPTZ NOT NULL,
      valid_to TIMESTAMPTZ,
      source_tier SMALLINT NOT NULL CHECK (source_tier BETWEEN 1 AND 4),
      confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
      prov_entity TEXT NOT NULL,
      prov_activity TEXT,
      prov_agent TEXT,
      source_url TEXT,
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_metric_obs_company_time ON metric_observations(company_code, observed_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_metric_obs_metric_time ON metric_observations(metric_name, observed_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_metric_obs_domain ON metric_observations(domain);",
    "CREATE INDEX IF NOT EXISTS idx_metric_obs_dedupe_lookup ON metric_observations(company_code, domain, metric_name, prov_entity, metric_value);",
    "CREATE INDEX IF NOT EXISTS idx_metric_obs_jsonb ON metric_observations USING GIN(extra);",
    """
    CREATE TABLE IF NOT EXISTS disclosures (
      id BIGSERIAL PRIMARY KEY,
      company_code TEXT REFERENCES companies(company_code),
      rcept_no TEXT UNIQUE,
      report_type TEXT,
      title TEXT,
      published_at TIMESTAMPTZ NOT NULL,
      raw_object_path TEXT,
      extracted JSONB DEFAULT '{}'::jsonb
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS price_daily (
      company_code TEXT REFERENCES companies(company_code),
      trade_date DATE NOT NULL,
      open NUMERIC(18,4),
      high NUMERIC(18,4),
      low NUMERIC(18,4),
      close NUMERIC(18,4),
      volume BIGINT,
      PRIMARY KEY (company_code, trade_date)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_store (
      id BIGSERIAL PRIMARY KEY,
      company_code TEXT REFERENCES companies(company_code),
      horizon TEXT NOT NULL,
      feature_name TEXT NOT NULL,
      feature_value DOUBLE PRECISION NOT NULL,
      as_of TIMESTAMPTZ NOT NULL,
      published_cutoff TIMESTAMPTZ NOT NULL,
      confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
      provenance_refs TEXT[] DEFAULT '{}'
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_feature_store_key ON feature_store(company_code, horizon, as_of DESC);",
    """
    CREATE TABLE IF NOT EXISTS tech_documents (
      id BIGSERIAL PRIMARY KEY,
      doc_uid TEXT UNIQUE NOT NULL,
      source TEXT NOT NULL,
      source_type TEXT NOT NULL,
      title TEXT NOT NULL,
      url TEXT,
      published_at TIMESTAMPTZ,
      collected_at TIMESTAMPTZ NOT NULL,
      summary TEXT,
      content TEXT,
      tags TEXT[] DEFAULT '{}',
      confidence DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_tech_documents_source_time ON tech_documents(source, published_at DESC);",
    "CREATE INDEX IF NOT EXISTS idx_tech_documents_collected_at ON tech_documents(collected_at DESC);",
    """
    CREATE TABLE IF NOT EXISTS tech_document_chunks (
      id BIGSERIAL PRIMARY KEY,
      doc_uid TEXT NOT NULL REFERENCES tech_documents(doc_uid) ON DELETE CASCADE,
      chunk_index INTEGER NOT NULL,
      chunk_text TEXT NOT NULL,
      char_len INTEGER NOT NULL,
      token_estimate INTEGER,
      created_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb,
      UNIQUE(doc_uid, chunk_index)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_tech_doc_chunks_uid ON tech_document_chunks(doc_uid);",
    """
    CREATE TABLE IF NOT EXISTS paper_sections (
      id BIGSERIAL PRIMARY KEY,
      doc_uid TEXT NOT NULL REFERENCES tech_documents(doc_uid) ON DELETE CASCADE,
      section_index INTEGER NOT NULL,
      section_title TEXT NOT NULL,
      section_level INTEGER DEFAULT 1,
      start_char INTEGER,
      end_char INTEGER,
      section_text TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb,
      UNIQUE(doc_uid, section_index)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_paper_sections_uid ON paper_sections(doc_uid);",
    """
    CREATE TABLE IF NOT EXISTS paper_tables (
      id BIGSERIAL PRIMARY KEY,
      doc_uid TEXT NOT NULL REFERENCES tech_documents(doc_uid) ON DELETE CASCADE,
      table_index INTEGER NOT NULL,
      caption TEXT,
      page_hint INTEGER,
      raw_text TEXT,
      image_path TEXT,
      created_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb,
      UNIQUE(doc_uid, table_index)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_paper_tables_uid ON paper_tables(doc_uid);",
    """
    CREATE TABLE IF NOT EXISTS paper_figures (
      id BIGSERIAL PRIMARY KEY,
      doc_uid TEXT NOT NULL REFERENCES tech_documents(doc_uid) ON DELETE CASCADE,
      figure_index INTEGER NOT NULL,
      caption TEXT,
      page_hint INTEGER,
      raw_text TEXT,
      image_path TEXT,
      created_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb,
      UNIQUE(doc_uid, figure_index)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_paper_figures_uid ON paper_figures(doc_uid);",
    "ALTER TABLE paper_tables ADD COLUMN IF NOT EXISTS image_path TEXT;",
    """
    CREATE TABLE IF NOT EXISTS event_candidates (
      event_id TEXT PRIMARY KEY,
      event_date TIMESTAMPTZ NOT NULL,
      event_type TEXT NOT NULL,
      source TEXT NOT NULL,
      source_tier SMALLINT NOT NULL CHECK (source_tier BETWEEN 1 AND 4),
      title TEXT NOT NULL,
      summary TEXT,
      related_company TEXT REFERENCES companies(company_code),
      related_domain TEXT,
      evidence_doc_uid TEXT REFERENCES tech_documents(doc_uid) ON DELETE SET NULL,
      confidence DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
      status TEXT DEFAULT 'new',
      created_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_event_candidates_date ON event_candidates(event_date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_event_candidates_company ON event_candidates(related_company, event_date DESC);",
    "CREATE INDEX IF NOT EXISTS idx_event_candidates_domain ON event_candidates(related_domain);",
    """
    CREATE TABLE IF NOT EXISTS event_outcomes (
      event_id TEXT PRIMARY KEY REFERENCES event_candidates(event_id) ON DELETE CASCADE,
      related_company TEXT REFERENCES companies(company_code),
      event_date DATE NOT NULL,
      ret_1d DOUBLE PRECISION,
      ret_5d DOUBLE PRECISION,
      ret_20d DOUBLE PRECISION,
      ret_60d DOUBLE PRECISION,
      volume_change_5d DOUBLE PRECISION,
      label TEXT,
      computed_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_event_outcomes_company ON event_outcomes(related_company, event_date DESC);",
    """
    CREATE TABLE IF NOT EXISTS price_forecasts (
      id BIGSERIAL PRIMARY KEY,
      company_code TEXT REFERENCES companies(company_code),
      horizon_days INTEGER NOT NULL,
      as_of TIMESTAMPTZ NOT NULL,
      published_cutoff TIMESTAMPTZ NOT NULL,
      target_date DATE NOT NULL,
      base_price NUMERIC(18,4) NOT NULL,
      expected_return DOUBLE PRECISION NOT NULL,
      low_return DOUBLE PRECISION NOT NULL,
      high_return DOUBLE PRECISION NOT NULL,
      expected_price NUMERIC(18,4) NOT NULL,
      low_price NUMERIC(18,4) NOT NULL,
      high_price NUMERIC(18,4) NOT NULL,
      method TEXT NOT NULL,
      signals JSONB DEFAULT '{}'::jsonb,
      features JSONB DEFAULT '{}'::jsonb,
      scenarios JSONB DEFAULT '[]'::jsonb,
      notes TEXT,
      created_at TIMESTAMPTZ NOT NULL,
      UNIQUE(company_code, horizon_days, as_of)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_price_forecasts_company_time ON price_forecasts(company_code, horizon_days, as_of DESC);",
    """
    CREATE TABLE IF NOT EXISTS price_forecast_evaluations (
      forecast_id BIGINT PRIMARY KEY REFERENCES price_forecasts(id) ON DELETE CASCADE,
      company_code TEXT REFERENCES companies(company_code),
      as_of TIMESTAMPTZ NOT NULL,
      horizon_days INTEGER NOT NULL,
      target_date DATE NOT NULL,
      realized_at DATE,
      base_price NUMERIC(18,4) NOT NULL,
      realized_close NUMERIC(18,4),
      realized_return DOUBLE PRECISION,
      expected_return DOUBLE PRECISION NOT NULL,
      abs_error DOUBLE PRECISION,
      interval_hit BOOLEAN,
      scenario_hit TEXT,
      feedback TEXT,
      evaluated_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_price_forecast_eval_company_time ON price_forecast_evaluations(company_code, evaluated_at DESC);",
    """
    CREATE TABLE IF NOT EXISTS forecast_experience_cases (
      id BIGSERIAL PRIMARY KEY,
      forecast_id BIGINT UNIQUE REFERENCES price_forecasts(id) ON DELETE CASCADE,
      company_code TEXT REFERENCES companies(company_code),
      as_of TIMESTAMPTZ NOT NULL,
      horizon_days INTEGER NOT NULL,
      target_date DATE NOT NULL,
      success_label TEXT NOT NULL,
      primary_pattern TEXT,
      error_tags TEXT[] DEFAULT '{}',
      related_domain TEXT,
      event_signature TEXT,
      context JSONB DEFAULT '{}'::jsonb,
      outcome JSONB DEFAULT '{}'::jsonb,
      created_at TIMESTAMPTZ NOT NULL
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forecast_experience_company_time ON forecast_experience_cases(company_code, as_of DESC);",
    "CREATE INDEX IF NOT EXISTS idx_forecast_experience_pattern ON forecast_experience_cases(primary_pattern, horizon_days);",
    "CREATE INDEX IF NOT EXISTS idx_forecast_experience_domain ON forecast_experience_cases(related_domain, horizon_days);",
    """
    CREATE TABLE IF NOT EXISTS forecast_experience_stats (
      stat_key TEXT PRIMARY KEY,
      stat_group TEXT NOT NULL,
      company_code TEXT REFERENCES companies(company_code),
      horizon_days INTEGER,
      related_domain TEXT,
      primary_pattern TEXT,
      sample_size INTEGER NOT NULL,
      success_rate DOUBLE PRECISION,
      interval_hit_rate DOUBLE PRECISION,
      direction_accuracy DOUBLE PRECISION,
      avg_abs_error DOUBLE PRECISION,
      avg_expected_return DOUBLE PRECISION,
      avg_realized_return DOUBLE PRECISION,
      updated_at TIMESTAMPTZ NOT NULL,
      extra JSONB DEFAULT '{}'::jsonb
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_forecast_experience_stats_group ON forecast_experience_stats(stat_group, company_code, horizon_days);",
]


HYPERTABLE_STATEMENTS = [
    # Timescale requires unique constraints to include partition key.
    # Drop default PK(id) and use composite uniqueness compatible with observed_at partitioning.
    "ALTER TABLE metric_observations DROP CONSTRAINT IF EXISTS metric_observations_pkey;",
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_metric_obs_id_observed_at_unique ON metric_observations(id, observed_at);",
    "SELECT create_hypertable('metric_observations', 'observed_at', if_not_exists => TRUE);",
]


def ensure_postgres_schema() -> None:
    with get_pg_conn() as conn:
        with conn.cursor() as cur:
            for stmt in POSTGRES_SCHEMA_STATEMENTS:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    # Keep bootstrap idempotent across managed providers.
                    # Example: extension permission limitations.
                    print(f"[WARN] Postgres schema statement skipped: {e}")

            for stmt in HYPERTABLE_STATEMENTS:
                try:
                    cur.execute(stmt)
                except Exception as e:
                    print(f"[WARN] Hypertable setup skipped: {e}")

        conn.commit()
    print("[OK] Postgres schema bootstrap complete")
