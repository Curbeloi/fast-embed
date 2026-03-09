use axum::{
    extract::Request,
    http::HeaderValue,
    middleware::Next,
    response::Response,
};
use chrono::Utc;
use std::sync::Arc;
use std::time::Instant;

use crate::store::{logs, DbPool};

pub async fn metering_middleware(
    request: Request,
    next: Next,
) -> Response {
    let start = Instant::now();
    let request_id = uuid::Uuid::now_v7().to_string();
    let endpoint = request.uri().path().to_string();

    let pool = request
        .extensions()
        .get::<Arc<DbPool>>()
        .cloned();

    let auth_key = request
        .extensions()
        .get::<crate::middleware::auth::AuthenticatedKey>()
        .cloned();

    let device = request
        .extensions()
        .get::<Arc<crate::hardware::HardwareProfile>>()
        .map(|h| h.device.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let mut response = next.run(request).await;

    let duration_ms = start.elapsed().as_millis() as u64;
    let status = response.status().as_u16();

    // Add headers
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("X-Request-Id", v);
    }
    if let Ok(v) = HeaderValue::from_str(&duration_ms.to_string()) {
        response.headers_mut().insert("X-Duration-Ms", v);
    }

    // Persist log asynchronously
    if let (Some(pool), Some(auth_key)) = (pool, auth_key) {
        let log = logs::RequestLog {
            request_id,
            api_key_id: auth_key.id,
            tenant_id: auth_key.tenant_id,
            endpoint,
            model: String::new(), // will be populated by handler if needed
            tokens_in: 0,
            tokens_out: 0,
            duration_ms,
            device,
            status,
            error: if status >= 400 {
                Some(format!("HTTP {}", status))
            } else {
                None
            },
            timestamp: Utc::now().to_rfc3339(),
        };

        tokio::spawn(async move {
            if let Err(e) = logs::insert(&log, &pool) {
                tracing::warn!("Failed to persist request log: {}", e);
            }
        });
    }

    response
}
