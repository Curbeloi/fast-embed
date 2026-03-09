use std::env;

#[derive(Debug, Clone)]
pub struct AppConfig {
    pub bind_address: String,
    pub log_level: String,
    pub default_embed_model: String,
    pub model_cache_dir: String,
    pub hf_token: Option<String>,
    pub db_path: String,
    pub admin_key_hash: Option<String>,
    pub override_max_concurrent: Option<usize>,
    pub override_max_per_minute: Option<u32>,
}

impl AppConfig {
    pub fn from_env() -> Self {
        Self {
            bind_address: env::var("BIND_ADDRESS")
                .unwrap_or_else(|_| "0.0.0.0:11435".to_string()),
            log_level: env::var("LOG_LEVEL").unwrap_or_else(|_| "info".to_string()),
            default_embed_model: env::var("DEFAULT_EMBED_MODEL")
                .unwrap_or_else(|_| "nomic-embed-text-v1.5".to_string()),
            model_cache_dir: env::var("MODEL_CACHE_DIR")
                .unwrap_or_else(|_| "./models".to_string()),
            hf_token: env::var("HF_TOKEN").ok().filter(|s| !s.is_empty()),
            db_path: env::var("DB_PATH")
                .unwrap_or_else(|_| "./data/inference.db".to_string()),
            admin_key_hash: env::var("ADMIN_KEY_HASH")
                .ok()
                .filter(|s| !s.is_empty()),
            override_max_concurrent: env::var("OVERRIDE_MAX_CONCURRENT")
                .ok()
                .and_then(|v| v.parse().ok()),
            override_max_per_minute: env::var("OVERRIDE_MAX_PER_MINUTE")
                .ok()
                .and_then(|v| v.parse().ok()),
        }
    }
}
