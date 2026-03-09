//! NomicBERT model implementation for Candle
//!
//! Custom implementation matching nomic-ai/nomic-embed-text-v1.5 architecture:
//! - Fused QKV projections (no bias)
//! - Rotary position embeddings (RoPE)
//! - SwiGLU activation in MLP
//! - Pre-norm (LayerNorm before attention and MLP)

use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{layer_norm, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

#[derive(Debug, Clone, Deserialize)]
pub struct NomicBertConfig {
    #[serde(alias = "vocab_size")]
    pub vocab_size: usize,
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_inner")]
    pub intermediate_size: usize,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_rotary_emb_base")]
    pub rotary_emb_base: f32,
    #[serde(default = "default_rotary_emb_fraction")]
    pub rotary_emb_fraction: f32,
}

fn default_type_vocab_size() -> usize { 2 }
fn default_layer_norm_eps() -> f64 { 1e-12 }
fn default_rotary_emb_base() -> f32 { 1000.0 }
fn default_rotary_emb_fraction() -> f32 { 1.0 }

impl NomicBertConfig {
    fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn rotary_dim(&self) -> usize {
        (self.head_dim() as f32 * self.rotary_emb_fraction) as usize
    }
}

// ── Rotary Position Embeddings ──────────────────────────────────

struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl RotaryEmbedding {
    fn new(config: &NomicBertConfig, device: &Device) -> Result<Self> {
        let dim = config.rotary_dim();
        let max_seq = config.max_position_embeddings;
        let base = config.rotary_emb_base as f64;

        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / (base.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        let positions: Vec<f32> = (0..max_seq).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq = inv_freq.unsqueeze(0)?;

        let freqs = positions.matmul(&inv_freq)?;
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;

        Ok(Self { cos, sin })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        // Cast cos/sin to match q/k dtype (e.g. F16)
        let cos = self.cos.i(..seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.i(..seq_len)?.to_dtype(q.dtype())?;

        let q_rot = apply_rotary(q, &cos, &sin)?;
        let k_rot = apply_rotary(k, &cos, &sin)?;

        Ok((q_rot, k_rot))
    }
}

fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (batch, num_heads, seq_len, dim) = x.dims4()?;
    let half_dim = dim / 2;

    // cos/sin are [max_seq, half_dim], we need [batch, heads, seq, half_dim]
    let cos = cos.i(..seq_len)?                          // [seq, half_dim]
        .unsqueeze(0)?                                    // [1, seq, half_dim]
        .unsqueeze(0)?                                    // [1, 1, seq, half_dim]
        .expand((batch, num_heads, seq_len, half_dim))?
        .contiguous()?;
    let sin = sin.i(..seq_len)?
        .unsqueeze(0)?
        .unsqueeze(0)?
        .expand((batch, num_heads, seq_len, half_dim))?
        .contiguous()?;

    let x1 = x.narrow(D::Minus1, 0, half_dim)?.contiguous()?;
    let x2 = x.narrow(D::Minus1, half_dim, half_dim)?.contiguous()?;

    let r1 = (x1.mul(&cos)? - x2.mul(&sin)?)?;
    let r2 = (x1.mul(&sin)? + x2.mul(&cos)?)?;

    Tensor::cat(&[&r1, &r2], D::Minus1)
}

// ── Attention ───────────────────────────────────────────────────

struct NomicAttention {
    wqkv: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl NomicAttention {
    fn load(vb: VarBuilder, config: &NomicBertConfig, device: &Device) -> Result<Self> {
        let wqkv = linear_no_bias(config.hidden_size, 3 * config.hidden_size, vb.pp("Wqkv"))?;
        let out_proj = linear_no_bias(config.hidden_size, config.hidden_size, vb.pp("out_proj"))?;
        let rotary = RotaryEmbedding::new(config, device)?;

        Ok(Self {
            wqkv,
            out_proj,
            num_heads: config.num_attention_heads,
            head_dim: config.head_dim(),
            rotary,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _hidden) = x.dims3()?;

        // Fused QKV: [batch, seq, hidden] -> [batch, seq, 3 * hidden]
        let qkv = self.wqkv.forward(x)?;
        let qkv = qkv.reshape((batch, seq_len, 3, self.num_heads, self.head_dim))?;

        let q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?; // [batch, heads, seq, dim]
        let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
        let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;

        // Apply rotary embeddings
        let (q, k) = self.rotary.apply(&q, &k, seq_len)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .contiguous()?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // Output projection
        self.out_proj.forward(&attn_output)
    }
}

// ── SwiGLU MLP ──────────────────────────────────────────────────

struct NomicMlp {
    fc11: Linear, // gate
    fc12: Linear, // up
    fc2: Linear,  // down
}

impl NomicMlp {
    fn load(vb: VarBuilder, config: &NomicBertConfig) -> Result<Self> {
        let fc11 = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("fc11"))?;
        let fc12 = linear_no_bias(config.hidden_size, config.intermediate_size, vb.pp("fc12"))?;
        let fc2 = linear_no_bias(config.intermediate_size, config.hidden_size, vb.pp("fc2"))?;

        Ok(Self { fc11, fc12, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.fc11.forward(x)?.silu()?;
        let up = self.fc12.forward(x)?;
        let hidden = (gate * up)?;
        self.fc2.forward(&hidden)
    }
}

// ── Transformer Layer ───────────────────────────────────────────

struct NomicBertLayer {
    norm1: LayerNorm,
    attn: NomicAttention,
    norm2: LayerNorm,
    mlp: NomicMlp,
}

impl NomicBertLayer {
    fn load(vb: VarBuilder, config: &NomicBertConfig, device: &Device) -> Result<Self> {
        let norm1 = layer_norm(config.hidden_size, config.layer_norm_epsilon, vb.pp("norm1"))?;
        let attn = NomicAttention::load(vb.pp("attn"), config, device)?;
        let norm2 = layer_norm(config.hidden_size, config.layer_norm_epsilon, vb.pp("norm2"))?;
        let mlp = NomicMlp::load(vb.pp("mlp"), config)?;

        Ok(Self { norm1, attn, norm2, mlp })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Pre-norm attention with residual
        let residual = x.clone();
        let x = self.norm1.forward(x)?;
        let x = self.attn.forward(&x)?;
        let x = (x + residual)?;

        // Pre-norm MLP with residual
        let residual = x.clone();
        let x = self.norm2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        x + residual
    }
}

// ── Full Model ──────────────────────────────────────────────────

pub struct NomicBertModel {
    word_embeddings: candle_nn::Embedding,
    token_type_embeddings: candle_nn::Embedding,
    emb_ln: LayerNorm,
    layers: Vec<NomicBertLayer>,
}

impl NomicBertModel {
    pub fn load(vb: VarBuilder, config: &NomicBertConfig, device: &Device) -> Result<Self> {
        let emb_vb = vb.pp("embeddings");
        let word_embeddings = candle_nn::embedding(
            config.vocab_size,
            config.hidden_size,
            emb_vb.pp("word_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            config.type_vocab_size,
            config.hidden_size,
            emb_vb.pp("token_type_embeddings"),
        )?;

        let emb_ln = layer_norm(
            config.hidden_size,
            config.layer_norm_epsilon,
            vb.pp("emb_ln"),
        )?;

        let enc_vb = vb.pp("encoder").pp("layers");
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(NomicBertLayer::load(enc_vb.pp(i), config, device)?);
        }

        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            emb_ln,
            layers,
        })
    }

    pub fn forward(&self, token_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let word_emb = self.word_embeddings.forward(token_ids)?;
        let type_emb = self.token_type_embeddings.forward(token_type_ids)?;

        let mut hidden = (word_emb + type_emb)?;
        hidden = self.emb_ln.forward(&hidden)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden)?;
        }

        Ok(hidden)
    }
}
