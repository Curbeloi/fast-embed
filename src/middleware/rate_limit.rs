use axum::{
    extract::Request,
    http::{HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use dashmap::DashMap;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::hardware::HardwareProfile;
use crate::middleware::auth::AuthenticatedKey;

pub type RateLimitStore = Arc<DashMap<String, VecDeque<Instant>>>;

pub fn new_store() -> RateLimitStore {
    Arc::new(DashMap::new())
}

pub async fn rate_limit_middleware(
    request: Request,
    next: Next,
) -> Response {
    let store = request
        .extensions()
        .get::<RateLimitStore>()
        .cloned();
    let hardware = request
        .extensions()
        .get::<Arc<HardwareProfile>>()
        .cloned();
    let auth_key = request
        .extensions()
        .get::<AuthenticatedKey>()
        .cloned();

    let (store, hardware, auth_key) = match (store, hardware, auth_key) {
        (Some(s), Some(h), Some(k)) => (s, h, k),
        _ => return next.run(request).await,
    };

    let limit = auth_key
        .rate_limit_override
        .unwrap_or(hardware.max_per_minute);

    let window_secs = 60u64;
    let now = Instant::now();
    let cutoff = now - Duration::from_secs(window_secs);

    let mut timestamps = store.entry(auth_key.id.clone()).or_default();
    timestamps.retain(|&t| t > cutoff);

    let count = timestamps.len() as u32;

    if count >= limit {
        let retry_after = timestamps
            .front()
            .map(|t| window_secs.saturating_sub(now.duration_since(*t).as_secs()))
            .unwrap_or(window_secs);

        let mut response = (
            StatusCode::TOO_MANY_REQUESTS,
            Json(json!({
                "error": "rate_limit_exceeded",
                "message": format!("Rate limit of {} requests per minute exceeded", limit),
                "retry_after": retry_after
            })),
        )
            .into_response();

        response.headers_mut().insert(
            "Retry-After",
            HeaderValue::from_str(&retry_after.to_string()).unwrap(),
        );
        response.headers_mut().insert(
            "X-RateLimit-Limit",
            HeaderValue::from_str(&limit.to_string()).unwrap(),
        );
        response.headers_mut().insert(
            "X-RateLimit-Remaining",
            HeaderValue::from(0),
        );

        return response;
    }

    timestamps.push_back(now);
    let remaining = limit - count - 1;
    drop(timestamps);

    let mut response = next.run(request).await;

    response.headers_mut().insert(
        "X-RateLimit-Limit",
        HeaderValue::from_str(&limit.to_string()).unwrap(),
    );
    response.headers_mut().insert(
        "X-RateLimit-Remaining",
        HeaderValue::from_str(&remaining.to_string()).unwrap(),
    );

    response
}
