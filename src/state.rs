use std::sync::Arc;
use std::time::Instant;

use crate::hardware::HardwareProfile;
use crate::inference::InferenceEngine;
use crate::store::DbPool;

pub struct AppState {
    pub db_pool: Arc<DbPool>,
    pub hardware: Arc<HardwareProfile>,
    pub engine: Arc<InferenceEngine>,
    pub start_time: Instant,
}
