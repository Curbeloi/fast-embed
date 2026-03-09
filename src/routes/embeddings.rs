use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;

use crate::state::AppState;

#[derive(Debug, Deserialize)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "nomic-embed-text-v1.5".to_string()
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

pub async fn embeddings_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<EmbeddingRequest>,
) -> impl IntoResponse {
    let texts: Vec<String> = match payload.input {
        EmbeddingInput::Single(s) => vec![s],
        EmbeddingInput::Batch(b) => b,
    };

    if texts.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "invalid_request", "message": "Input cannot be empty"})),
        )
            .into_response();
    }

    // Ensure model is loaded (lazy loading)
    if let Err(e) = state.engine.ensure_embedding_model().await {
        tracing::error!("Failed to load embedding model: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "model_load_error", "message": format!("Failed to load model: {}", e)})),
        )
            .into_response();
    }

    // Run inference in a blocking task to avoid blocking the async runtime
    let engine_embeddings = state.engine.embeddings.clone();
    let result = tokio::task::spawn_blocking(move || {
        let guard = engine_embeddings.blocking_read();
        match guard.as_ref() {
            Some(model) => model.embed(&texts),
            None => Err(anyhow::anyhow!("Model not loaded")),
        }
    })
    .await;

    let (embeddings, total_tokens) = match result {
        Ok(Ok((emb, tokens))) => (emb, tokens),
        Ok(Err(e)) => {
            let err_msg = format!("{}", e);
            // Context window errors → 400, other errors → 500
            if err_msg.contains("exceeds maximum context window") {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(json!({"error": "context_window_exceeded", "message": err_msg})),
                )
                    .into_response();
            }
            tracing::error!("Embedding inference error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "inference_error", "message": err_msg})),
            )
                .into_response();
        }
        Err(e) => {
            tracing::error!("Task join error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal_error", "message": "Inference task failed"})),
            )
                .into_response();
        }
    };

    let data: Vec<EmbeddingData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, emb)| EmbeddingData {
            object: "embedding".to_string(),
            index: i,
            embedding: emb,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: payload.model,
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    (StatusCode::OK, Json(serde_json::to_value(response).unwrap())).into_response()
}
