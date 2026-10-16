-- =============================================================================
-- TimescaleDB Schema — Macro Analysis Agent
-- 실행 전제: TimescaleDB 확장이 설치된 PostgreSQL 14+ 인스턴스
-- 최초 1회 실행 후 멱등성 보장 (IF NOT EXISTS 조건 전체 적용)
-- =============================================================================

-- 1. TimescaleDB 확장은 initialize_schema()가 public에 최초 한 번 설치합니다.
-- 이미 설치된 공유 DB에서는 확장 생성 명령을 다시 실행하지 않습니다.

-- =============================================================================
-- 2. 거시경제 시계열 테이블 (Hypertable 변환 대상)
-- =============================================================================
CREATE TABLE IF NOT EXISTS macro_metrics (
    time         TIMESTAMPTZ      NOT NULL,              -- 데이터 기준 시점 (UTC)
    metric_name  VARCHAR(120)     NOT NULL,              -- 지표 코드 (예: 'ecos.usd_krw')
    value        DOUBLE PRECISION NOT NULL,              -- 수치 값
    source       VARCHAR(60)      NOT NULL,              -- 데이터 출처 ('ecos' | 'fred' | 'market')
    region       VARCHAR(60),                            -- 지역 코드 ('KR' | 'US' | 'GLOBAL')
    unit         VARCHAR(60),                            -- 단위 ('KRW/USD' | 'INDEX' | '%')
    metadata     JSONB,                                  -- 원본 API 부가 필드 보존
    created_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- 3. Hypertable 변환 (TimescaleDB 핵심 기능)
--    chunk_time_interval = 1 month: 월별 청크 파티션으로 쿼리 pruning 최적화
-- =============================================================================
SELECT create_hypertable(
    'macro_metrics',
    'time',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists       => TRUE
);

-- =============================================================================
-- 4. Upsert 무결성 제약 (time + metric_name + source 조합 중복 방지)
-- =============================================================================
CREATE UNIQUE INDEX IF NOT EXISTS uix_macro_metrics_key
    ON macro_metrics (time, metric_name, source);

-- =============================================================================
-- 5. 조회 성능 최적화 인덱스
-- =============================================================================

-- 지표별 최신순 조회 (가장 자주 사용되는 패턴)
CREATE INDEX IF NOT EXISTS idx_macro_metrics_name_time
    ON macro_metrics (metric_name, time DESC);

-- 출처별 전체 조회
CREATE INDEX IF NOT EXISTS idx_macro_metrics_source_time
    ON macro_metrics (source, time DESC);

-- 지역별 필터 조회
CREATE INDEX IF NOT EXISTS idx_macro_metrics_region_time
    ON macro_metrics (region, time DESC)
    WHERE region IS NOT NULL;

-- JSONB 메타데이터 부분 인덱스 (필요 시 확장)
-- CREATE INDEX IF NOT EXISTS idx_macro_metrics_metadata
--     ON macro_metrics USING gin (metadata);

-- =============================================================================
-- 6. updated_at 자동 갱신 트리거
-- =============================================================================
CREATE OR REPLACE FUNCTION _set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_macro_metrics_updated_at ON macro_metrics;
CREATE TRIGGER trg_macro_metrics_updated_at
    BEFORE UPDATE ON macro_metrics
    FOR EACH ROW EXECUTE FUNCTION _set_updated_at();

-- =============================================================================
-- 7. 데이터 압축 정책 (선택 — 90일 이상 청크 자동 압축, 스토리지 약 90% 절감)
-- =============================================================================
-- ALTER TABLE macro_metrics SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'metric_name, source',
--     timescaledb.compress_orderby   = 'time DESC'
-- );
-- SELECT add_compression_policy('macro_metrics', INTERVAL '90 days');

-- =============================================================================
-- 8. 데이터 보존 정책 (선택 — 10년 초과 데이터 자동 삭제)
-- =============================================================================
-- SELECT add_retention_policy('macro_metrics', INTERVAL '10 years');

-- =============================================================================
-- 9. 연속 집계 뷰 (선택 — 분기별 평균을 실시간으로 집계)
-- =============================================================================
-- CREATE MATERIALIZED VIEW IF NOT EXISTS macro_metrics_quarterly
-- WITH (timescaledb.continuous) AS
--     SELECT
--         time_bucket('3 months', time) AS bucket,
--         metric_name,
--         source,
--         AVG(value)   AS avg_value,
--         MIN(value)   AS min_value,
--         MAX(value)   AS max_value,
--         COUNT(*)     AS data_points
--     FROM macro_metrics
--     GROUP BY bucket, metric_name, source
-- WITH NO DATA;
-- SELECT add_continuous_aggregate_policy('macro_metrics_quarterly',
--     start_offset => INTERVAL '1 year',
--     end_offset   => INTERVAL '1 day',
--     schedule_interval => INTERVAL '1 week');

-- =============================================================================
-- 10. 뉴스 기사 원문 저장 테이블 (news_articles)
-- =============================================================================
CREATE TABLE IF NOT EXISTS news_articles (
    id              BIGSERIAL       PRIMARY KEY,
    url             TEXT            UNIQUE NOT NULL,
    title           TEXT            NOT NULL,
    body            TEXT,
    summary         TEXT,
    publisher       TEXT,
    author          TEXT,
    published_date  DATE,
    lang            CHAR(2)         NOT NULL DEFAULT 'en',
    query_tag       TEXT,
    crawled_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- FinBERT 감성 분석
    sentiment_pos   FLOAT,
    sentiment_neg   FLOAT,
    sentiment_neu   FLOAT,
    analyzed_at     TIMESTAMPTZ,

    -- NER 엔티티 + 키워드
    entities        JSONB,
    keywords        TEXT[],

    -- Vector DB 적재 상태
    vector_ingested BOOLEAN         NOT NULL DEFAULT FALSE,
    vector_chunks   INT             NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_news_articles_published
    ON news_articles (published_date DESC);

CREATE INDEX IF NOT EXISTS idx_news_articles_publisher
    ON news_articles (publisher, published_date DESC);

CREATE INDEX IF NOT EXISTS idx_news_articles_entities
    ON news_articles USING GIN (entities)
    WHERE entities IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_news_articles_not_ingested
    ON news_articles (id)
    WHERE vector_ingested = FALSE;

CREATE INDEX IF NOT EXISTS idx_news_articles_not_analyzed
    ON news_articles (id)
    WHERE analyzed_at IS NULL;

-- =============================================================================
-- 11. 표준 지표 코드 참조 주석
-- =============================================================================
-- ecos.usd_krw                   — 원/달러 환율 (KRW/USD)
-- ecos.semiconductor_export_value — 반도체 수출액
-- ecos.semiconductor_shipment_index — 반도체 출하지수
-- ecos.semiconductor_inventory_index — 반도체 재고지수
-- fred.ism_pmi                   — ISM 제조업 PMI 종합지수
-- fred.fed_funds_rate            — 미국 연방기금 실효금리 (%)
-- market.sox_close               — 필라델피아 반도체지수 SOX 월봉 종가
