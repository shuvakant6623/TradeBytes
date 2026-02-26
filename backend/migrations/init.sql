-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- ─── REGIME ENGINE ────────────────────────────────────────────────────────────
CREATE TABLE regime_states (
    time          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    asset_id      TEXT         NOT NULL,
    regime        TEXT         NOT NULL CHECK (regime IN ('trending','mean_reverting','high_volatility','low_volatility','news_driven')),
    confidence    JSONB        NOT NULL DEFAULT '{}',
    model_version TEXT,
    features      JSONB
);
SELECT create_hypertable('regime_states', 'time');
CREATE INDEX ON regime_states (asset_id, time DESC);

CREATE TABLE regime_model_versions (
    id            UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    version       TEXT    UNIQUE NOT NULL,
    trained_at    TIMESTAMPTZ DEFAULT NOW(),
    train_end     DATE,
    metrics       JSONB,
    is_active     BOOLEAN DEFAULT FALSE,
    model_path    TEXT
);

-- ─── NEWS INTELLIGENCE ────────────────────────────────────────────────────────
CREATE TABLE news_articles (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    published_at    TIMESTAMPTZ NOT NULL,
    source          TEXT,
    headline        TEXT        NOT NULL,
    body_excerpt    TEXT,
    url             TEXT        UNIQUE,
    fingerprint     TEXT        UNIQUE,
    sentiment_score FLOAT       CHECK (sentiment_score BETWEEN -1 AND 1),
    sentiment_label TEXT        CHECK (sentiment_label IN ('positive','neutral','negative')),
    embedding       vector(384),
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
SELECT create_hypertable('news_articles', 'published_at');
CREATE INDEX ON news_articles (published_at DESC);
CREATE INDEX ON news_articles USING hnsw (embedding vector_cosine_ops);

CREATE TABLE news_ticker_map (
    article_id  UUID  REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker_id   TEXT  NOT NULL,
    confidence  FLOAT DEFAULT 1.0,
    PRIMARY KEY (article_id, ticker_id)
);
CREATE INDEX ON news_ticker_map (ticker_id);

CREATE MATERIALIZED VIEW sentiment_rolling_1h
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', published_at) AS bucket,
    ntm.ticker_id,
    AVG(na.sentiment_score)              AS avg_sentiment,
    STDDEV(na.sentiment_score)           AS sentiment_variance,
    COUNT(*)                             AS article_count,
    SUM(CASE WHEN na.sentiment_score > 0.3 THEN 1 ELSE 0 END)  AS bullish_count,
    SUM(CASE WHEN na.sentiment_score < -0.3 THEN 1 ELSE 0 END) AS bearish_count
FROM news_articles na
JOIN news_ticker_map ntm ON ntm.article_id = na.id
GROUP BY bucket, ntm.ticker_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('sentiment_rolling_1h',
    start_offset => INTERVAL '7 days',
    end_offset   => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

-- ─── RISK ENGINE ─────────────────────────────────────────────────────────────
CREATE TABLE portfolio_snapshots (
    id                  UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID        NOT NULL,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    window_days         INT         DEFAULT 252,
    volatility          FLOAT,
    beta                FLOAT,
    max_drawdown        FLOAT,
    sharpe              FLOAT,
    hhi                 FLOAT,
    diversification     FLOAT,
    health_score        FLOAT,
    correlation_matrix  JSONB,
    weights             JSONB,
    warnings            JSONB       DEFAULT '[]'
);
SELECT create_hypertable('portfolio_snapshots', 'computed_at');
CREATE INDEX ON portfolio_snapshots (user_id, computed_at DESC);

CREATE TABLE price_history (
    time      TIMESTAMPTZ NOT NULL,
    asset_id  TEXT        NOT NULL,
    open      FLOAT,
    high      FLOAT,
    low       FLOAT,
    close     FLOAT       NOT NULL,
    volume    BIGINT
);
SELECT create_hypertable('price_history', 'time');
CREATE INDEX ON price_history (asset_id, time DESC);

-- ─── BEHAVIORAL ENGINE ────────────────────────────────────────────────────────
CREATE TABLE trades (
    id              UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID        NOT NULL,
    asset_id        TEXT        NOT NULL,
    side            TEXT        CHECK (side IN ('long','short')),
    entry_price     FLOAT       NOT NULL,
    exit_price      FLOAT,
    quantity        FLOAT       NOT NULL,
    entry_time      TIMESTAMPTZ NOT NULL,
    exit_time       TIMESTAMPTZ,
    pnl             FLOAT,
    pnl_pct         FLOAT,
    status          TEXT        DEFAULT 'open' CHECK (status IN ('open','closed')),
    sector          TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
CREATE INDEX ON trades (user_id, entry_time DESC);
CREATE INDEX ON trades (user_id, status);

CREATE TABLE behavioral_profiles (
    user_id              UUID        PRIMARY KEY,
    updated_at           TIMESTAMPTZ DEFAULT NOW(),
    feature_vector       FLOAT[],
    archetype            TEXT,
    risk_score           FLOAT,
    profit_factor        FLOAT,
    win_loss_asymmetry   FLOAT,
    overtrading_z        FLOAT,
    diversification_beh  FLOAT,
    loss_recovery_speed  FLOAT,
    disposition_effect   FLOAT,
    overconfidence_score FLOAT,
    loss_aversion_ratio  FLOAT,
    trade_duration_p50   FLOAT,
    bias_flags           JSONB       DEFAULT '{}'
);

-- ─── GAMIFICATION ENGINE ─────────────────────────────────────────────────────
CREATE TABLE user_gamification (
    user_id             UUID    PRIMARY KEY,
    total_xp            BIGINT  DEFAULT 0,
    level               INT     DEFAULT 1,
    streak_days         INT     DEFAULT 0,
    streak_grace_used   BOOLEAN DEFAULT FALSE,
    last_active_day     DATE,
    unlocked_features   TEXT[]  DEFAULT '{}',
    badges              JSONB   DEFAULT '[]',
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE xp_transactions (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID        NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    action_type TEXT        NOT NULL,
    base_xp     INT         NOT NULL,
    multipliers JSONB       DEFAULT '{}',
    final_xp    INT         NOT NULL,
    ref_id      UUID
);
SELECT create_hypertable('xp_transactions', 'ts');
CREATE INDEX ON xp_transactions (user_id, ts DESC);

CREATE TABLE leaderboard_scores (
    user_id     UUID        PRIMARY KEY,
    score       FLOAT       DEFAULT 0,
    cohort      TEXT,
    rank        INT,
    sharpe_z    FLOAT,
    xp_z        FLOAT,
    winrate_z   FLOAT,
    health_z    FLOAT,
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── SHARED USERS TABLE ───────────────────────────────────────────────────────
CREATE TABLE users (
    id          UUID    PRIMARY KEY DEFAULT uuid_generate_v4(),
    username    TEXT    UNIQUE NOT NULL,
    email       TEXT    UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
