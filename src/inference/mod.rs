pub mod embeddings;
pub mod nomic_bert;

use candle_core::Device;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::hardware::HardwareProfile;

pub struct InferenceEngine {
    pub embeddings: Arc<RwLock<Option<embeddings::EmbeddingModel>>>,
    pub device: Device,
    pub hardware: Arc<HardwareProfile>,
    pub model_cache_dir: String,
    pub default_embed_model: String,
    pub hf_token: Option<String>,
}

impl InferenceEngine {
    pub fn new(
        hardware: Arc<HardwareProfile>,
        model_cache_dir: String,
        default_embed_model: String,
        hf_token: Option<String>,
    ) -> Self {
        let device = hardware.candle_device();
        Self {
            embeddings: Arc::new(RwLock::new(None)),
            device,
            hardware,
            model_cache_dir,
            default_embed_model,
            hf_token,
        }
    }

    pub async fn ensure_embedding_model(&self) -> anyhow::Result<()> {
        let read = self.embeddings.read().await;
        if read.is_some() {
            return Ok(());
        }
        drop(read);

        let mut write = self.embeddings.write().await;
        if write.is_some() {
            return Ok(());
        }

        tracing::info!("Loading embedding model: {}", self.default_embed_model);
        let model = embeddings::EmbeddingModel::load(
            &self.default_embed_model,
            &self.model_cache_dir,
            &self.device,
            self.hf_token.as_deref(),
        )?;
        *write = Some(model);
        tracing::info!("Embedding model loaded successfully");

        Ok(())
    }

    pub fn loaded_models(&self) -> Vec<String> {
        let mut models = Vec::new();
        if self.embeddings.try_read().map(|r| r.is_some()).unwrap_or(false) {
            models.push(self.default_embed_model.clone());
        }
        models
    }
}
