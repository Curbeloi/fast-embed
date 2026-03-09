use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use tokenizers::Tokenizer;

use super::nomic_bert::{NomicBertConfig, NomicBertModel};

pub struct EmbeddingModel {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
    max_seq_len: usize,
}

impl EmbeddingModel {
    pub fn load(
        model_id: &str,
        _cache_dir: &str,
        device: &Device,
        hf_token: Option<&str>,
    ) -> Result<Self> {
        let repo_id = match model_id {
            "nomic-embed-text-v1.5" => "nomic-ai/nomic-embed-text-v1.5",
            other => other,
        };

        let mut builder = ApiBuilder::new();
        if let Some(token) = hf_token {
            builder = builder.with_token(Some(token.to_string()));
        }
        let api = builder.build()?;

        let repo = api.repo(Repo::new(repo_id.to_string(), RepoType::Model));

        tracing::info!("Downloading model files from {}", repo_id);
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo.get("model.safetensors")?;

        let config_str = std::fs::read_to_string(&config_path)?;
        let config: NomicBertConfig = serde_json::from_str(&config_str)?;
        let max_seq_len = config.max_position_embeddings;
        tracing::info!(
            "Model config: hidden_size={}, layers={}, heads={}, max_seq_len={}",
            config.hidden_size, config.num_hidden_layers, config.num_attention_heads, max_seq_len
        );

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // F16 halves memory and improves Metal/CUDA throughput
        let dtype = candle_core::DType::F16;
        tracing::info!("Loading weights with dtype: {:?}", dtype);
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_path], dtype, device)?
        };

        let model = NomicBertModel::load(vb, &config, device)?;

        Ok(Self {
            model,
            tokenizer,
            device: device.clone(),
            max_seq_len,
        })
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Max items per sub-batch. Keeps GPU memory bounded and
    /// avoids giant [batch × seq × hidden] tensors.
    const SUB_BATCH_SIZE: usize = 16;

    pub fn embed(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u32)> {
        // 1. Tokenize all texts up-front and validate context window
        let mut encodings = Vec::with_capacity(texts.len());
        for (i, text) in texts.iter().enumerate() {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

            let n_tokens = encoding.get_ids().len();
            if n_tokens > self.max_seq_len {
                return Err(anyhow::anyhow!(
                    "Input {} exceeds maximum context window: {} tokens (max {}). \
                     Split your text into smaller chunks.",
                    i, n_tokens, self.max_seq_len
                ));
            }
            encodings.push(encoding);
        }

        let total_tokens: u32 = encodings.iter().map(|e| e.get_ids().len() as u32).sum();
        let mut all_embeddings = Vec::with_capacity(encodings.len());

        // 2. Process in sub-batches to bound GPU memory
        for chunk in encodings.chunks(Self::SUB_BATCH_SIZE) {
            let sub_embeddings = self.embed_batch(chunk)?;
            all_embeddings.extend(sub_embeddings);
        }

        Ok((all_embeddings, total_tokens))
    }

    /// Run a single batched forward pass for a sub-batch of encodings.
    fn embed_batch(&self, encodings: &[tokenizers::Encoding]) -> Result<Vec<Vec<f32>>> {
        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        // Build padded tensors
        let mut batch_ids = Vec::with_capacity(batch_size * max_len);
        let mut batch_type_ids = Vec::with_capacity(batch_size * max_len);
        let mut attention_mask = Vec::with_capacity(batch_size * max_len);
        let mut real_lengths = Vec::with_capacity(batch_size);

        for enc in encodings {
            let ids = enc.get_ids();
            let type_ids = enc.get_type_ids();
            let len = ids.len();
            real_lengths.push(len);

            batch_ids.extend_from_slice(ids);
            batch_type_ids.extend_from_slice(type_ids);
            attention_mask.extend(std::iter::repeat(1.0f32).take(len));

            let pad = max_len - len;
            if pad > 0 {
                batch_ids.extend(std::iter::repeat(0u32).take(pad));
                batch_type_ids.extend(std::iter::repeat(0u32).take(pad));
                attention_mask.extend(std::iter::repeat(0.0f32).take(pad));
            }
        }

        let token_ids = Tensor::new(batch_ids, &self.device)?
            .reshape((batch_size, max_len))?;
        let token_type_ids = Tensor::new(batch_type_ids, &self.device)?
            .reshape((batch_size, max_len))?;
        let mask = Tensor::new(attention_mask, &self.device)?
            .reshape((batch_size, max_len))?;

        // Single forward pass for this sub-batch
        let hidden = self.model.forward(&token_ids, &token_type_ids)?;

        // Masked mean pooling + L2 normalization
        // Cast mask to match hidden dtype (F16), pool in F16, then convert to F32
        let mask_expanded = mask.to_dtype(hidden.dtype())?.unsqueeze(2)?;
        let masked = hidden.broadcast_mul(&mask_expanded)?;
        let sum = masked.sum(1)?.to_dtype(candle_core::DType::F32)?;

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let emb_sum: Vec<f32> = sum.i(i)?.to_vec1()?;
            let len = real_lengths[i] as f32;

            let mut sq_sum = 0.0f32;
            let mean: Vec<f32> = emb_sum.iter().map(|&x| {
                let m = x / len;
                sq_sum += m * m;
                m
            }).collect();

            let norm = sq_sum.sqrt();
            let normalized = if norm > 0.0 {
                mean.iter().map(|&x| x / norm).collect()
            } else {
                mean
            };
            results.push(normalized);
        }

        Ok(results)
    }
}
