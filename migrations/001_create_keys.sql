CREATE TABLE IF NOT EXISTS api_keys (
    id                    TEXT PRIMARY KEY,
    key_hash              TEXT NOT NULL UNIQUE,
    name                  TEXT NOT NULL,
    tenant_id             TEXT NOT NULL,
    active                INTEGER NOT NULL DEFAULT 1,
    rate_limit_override   INTEGER,
    created_at            TEXT NOT NULL,
    revoked_at            TEXT,
    last_used_at          TEXT
);

CREATE INDEX IF NOT EXISTS idx_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_keys_tenant ON api_keys(tenant_id);
