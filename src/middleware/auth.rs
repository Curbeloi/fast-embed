use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use std::sync::Arc;

use crate::store::{keys, DbPool};

#[derive(Debug, Clone)]
pub struct AuthenticatedKey {
    pub id: String,
    pub tenant_id: String,
    pub rate_limit_override: Option<u32>,
}

pub async fn auth_middleware(
    request: Request,
    next: Next,
) -> Response {
    let pool = request
        .extensions()
        .get::<Arc<DbPool>>()
        .cloned();

    let pool = match pool {
        Some(p) => p,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal_error", "message": "Database pool not available"})),
            )
                .into_response();
        }
    };

    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let api_key = match api_key {
        Some(k) => k,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "missing_api_key", "message": "X-API-Key header is required"})),
            )
                .into_response();
        }
    };

    let key_info = match keys::validate_key(&api_key, &pool) {
        Ok(Some(info)) => info,
        Ok(None) => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "invalid_api_key", "message": "The provided API key is invalid or revoked"})),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal_error", "message": "Failed to validate API key"})),
            )
                .into_response();
        }
    };

    let auth_key = AuthenticatedKey {
        id: key_info.id,
        tenant_id: key_info.tenant_id,
        rate_limit_override: key_info.rate_limit_override,
    };

    let mut request = request;
    request.extensions_mut().insert(auth_key);

    next.run(request).await
}

pub async fn admin_auth_middleware(
    request: Request,
    next: Next,
) -> Response {
    let pool = request
        .extensions()
        .get::<Arc<DbPool>>()
        .cloned();

    let pool = match pool {
        Some(p) => p,
        None => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal_error", "message": "Database pool not available"})),
            )
                .into_response();
        }
    };

    let api_key = request
        .headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let api_key = match api_key {
        Some(k) => k,
        None => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(json!({"error": "missing_api_key", "message": "X-API-Key header is required"})),
            )
                .into_response();
        }
    };

    match keys::is_admin_key(&api_key, &pool) {
        Ok(true) => {}
        Ok(false) => {
            return (
                StatusCode::FORBIDDEN,
                Json(json!({"error": "forbidden", "message": "Admin access required"})),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal_error", "message": "Failed to validate API key"})),
            )
                .into_response();
        }
    }

    next.run(request).await
}
