CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    event_id        UUID NOT NULL UNIQUE,
    event_timestamp TIMESTAMPTZ NOT NULL,
    features        JSONB NOT NULL,
    prediction      INTEGER NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    model_version   VARCHAR(64) NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_predictions_timestamp ON predictions (event_timestamp);
CREATE INDEX idx_predictions_model_version ON predictions (model_version);