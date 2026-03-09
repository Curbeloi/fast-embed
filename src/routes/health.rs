use axum::{extract::State, Json};
use serde_json::{json, Value};
use std::sync::Arc;

use crate::state::AppState;

pub async fn health_handler(State(state): State<Arc<AppState>>) -> Json<Value> {
    let hw = &state.hardware;
    let uptime = state.start_time.elapsed().as_secs();

    Json(json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": uptime,
        "hardware": {
            "device": hw.device.to_string(),
            "vram_gb": hw.vram_gb,
            "ram_gb": hw.ram_gb,
            "cpu_cores": hw.cpu_cores
        },
        "limits": {
            "max_concurrent": hw.max_concurrent,
            "max_per_minute": hw.max_per_minute,
            "max_queue": hw.max_queue
        },
        "models_loaded": state.engine.loaded_models()
    }))
}
