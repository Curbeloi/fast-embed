use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;

use crate::state::AppState;
use crate::store::{keys, logs};

#[derive(Debug, Deserialize)]
pub struct StatsQuery {
    pub from: Option<String>,
    pub to: Option<String>,
}

pub async fn stats_handler(
    State(state): State<Arc<AppState>>,
    Query(query): Query<StatsQuery>,
) -> impl IntoResponse {
    let from = query
        .from
        .unwrap_or_else(|| "2000-01-01T00:00:00Z".to_string());
    let to = query
        .to
        .unwrap_or_else(|| "2099-12-31T23:59:59Z".to_string());

    match logs::get_stats(&state.db_pool, &from, &to) {
        Ok(stats) => (StatusCode::OK, Json(json!({
            "period": { "from": from, "to": to },
            "totals": {
                "requests": stats.total_requests,
                "tokens_in": stats.total_tokens_in,
                "tokens_out": stats.total_tokens_out,
                "errors": stats.total_errors,
                "avg_duration_ms": stats.avg_duration_ms
            },
            "by_tenant": stats.by_tenant
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "query_error", "message": format!("{}", e)})),
        ).into_response(),
    }
}

pub async fn tenant_stats_handler(
    State(state): State<Arc<AppState>>,
    Path(tenant_id): Path<String>,
    Query(query): Query<StatsQuery>,
) -> impl IntoResponse {
    let from = query.from.unwrap_or_else(|| "2000-01-01T00:00:00Z".to_string());
    let to = query.to.unwrap_or_else(|| "2099-12-31T23:59:59Z".to_string());

    match logs::get_tenant_stats(&state.db_pool, &tenant_id, &from, &to) {
        Ok(Some(stats)) => (StatusCode::OK, Json(serde_json::to_value(stats).unwrap())).into_response(),
        Ok(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "not_found", "message": "No data for this tenant"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "query_error", "message": format!("{}", e)})),
        ).into_response(),
    }
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyRequest {
    pub tenant_id: String,
    pub name: String,
    pub rate_limit_override: Option<u32>,
}

pub async fn create_key_handler(
    State(state): State<Arc<AppState>>,
    Json(payload): Json<CreateKeyRequest>,
) -> impl IntoResponse {
    match keys::create_key(
        &payload.tenant_id,
        &payload.name,
        payload.rate_limit_override,
        &state.db_pool,
    ) {
        Ok((raw_key, info)) => (StatusCode::CREATED, Json(json!({
            "id": info.id,
            "key": raw_key,
            "tenant_id": info.tenant_id,
            "name": info.name,
            "created_at": info.created_at
        }))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "create_error", "message": format!("{}", e)})),
        ).into_response(),
    }
}

pub async fn revoke_key_handler(
    State(state): State<Arc<AppState>>,
    Path(key_id): Path<String>,
) -> impl IntoResponse {
    match keys::revoke_key(&key_id, &state.db_pool) {
        Ok(true) => (StatusCode::OK, Json(json!({"message": "Key revoked successfully"}))).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(json!({"error": "not_found", "message": "Key not found or already revoked"})),
        ).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "revoke_error", "message": format!("{}", e)})),
        ).into_response(),
    }
}

pub async fn list_keys_handler(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match keys::list_keys(&state.db_pool) {
        Ok(all_keys) => (StatusCode::OK, Json(serde_json::to_value(all_keys).unwrap())).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": "query_error", "message": format!("{}", e)})),
        ).into_response(),
    }
}
