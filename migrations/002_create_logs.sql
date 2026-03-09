CREATE TABLE IF NOT EXISTS request_logs (
    id            TEXT PRIMARY KEY,
    api_key_id    TEXT NOT NULL,
    tenant_id     TEXT NOT NULL,
    endpoint      TEXT NOT NULL,
    model         TEXT NOT NULL,
    tokens_in     INTEGER NOT NULL DEFAULT 0,
    tokens_out    INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER NOT NULL,
    device        TEXT NOT NULL,
    status        INTEGER NOT NULL,
    error         TEXT,
    timestamp     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_logs_tenant    ON request_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_logs_key       ON request_logs(api_key_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_status    ON request_logs(status);
