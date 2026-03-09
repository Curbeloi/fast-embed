use sysinfo::System;

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda(u32),
}

impl std::fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceKind::Cpu => write!(f, "cpu"),
            DeviceKind::Metal => write!(f, "metal"),
            DeviceKind::Cuda(id) => write!(f, "cuda:{id}"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub device: DeviceKind,
    pub vram_gb: f32,
    pub ram_gb: f32,
    pub cpu_cores: u32,
    pub max_concurrent: usize,
    pub max_per_minute: u32,
    pub max_queue: usize,
    pub max_batch_tokens: usize,
}

impl HardwareProfile {
    pub fn candle_device(&self) -> candle_core::Device {
        match &self.device {
            #[cfg(feature = "cuda")]
            DeviceKind::Cuda(id) => {
                candle_core::Device::new_cuda(*id as usize).unwrap_or(candle_core::Device::Cpu)
            }
            #[cfg(feature = "metal")]
            DeviceKind::Metal => {
                candle_core::Device::new_metal(0).unwrap_or(candle_core::Device::Cpu)
            }
            _ => candle_core::Device::Cpu,
        }
    }
}

pub fn detect() -> HardwareProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    let ram_gb = sys.total_memory() as f32 / 1_073_741_824.0;
    let cores = sys.cpus().len() as u32;

    // Runtime auto-detection: try GPU first, fallback to CPU
    let (device, vram_gb) = detect_best_device(ram_gb);

    let (max_concurrent, max_per_minute, max_queue) = match &device {
        DeviceKind::Cuda(_) => {
            let factor = (vram_gb / 8.0).floor() as usize;
            let factor = factor.max(1);
            (factor * 8, factor as u32 * 50, factor * 25)
        }
        DeviceKind::Metal => {
            let factor = (vram_gb / 4.0).floor() as usize;
            let factor = factor.max(1);
            (factor * 4, factor as u32 * 30, factor * 15)
        }
        DeviceKind::Cpu => {
            let c = cores.max(2) as usize;
            (c / 2, cores.max(2) * 8, c * 4)
        }
    };

    HardwareProfile {
        device,
        vram_gb,
        ram_gb,
        cpu_cores: cores,
        max_concurrent,
        max_per_minute,
        max_queue,
        max_batch_tokens: 8192,
    }
}

/// Try CUDA → Metal → CPU in order. Each backend is only attempted
/// if the binary was compiled with the corresponding feature flag.
/// If the GPU init fails at runtime, it falls back gracefully to CPU.
fn detect_best_device(ram_gb: f32) -> (DeviceKind, f32) {
    // Suppress unused warning when no GPU features are enabled
    let _ = ram_gb;

    // 1. Try CUDA
    #[cfg(feature = "cuda")]
    {
        match candle_core::Device::new_cuda(0) {
            Ok(_) => {
                tracing::info!("CUDA GPU detected");
                // VRAM detection is approximate; default to 8GB if unknown
                return (DeviceKind::Cuda(0), 8.0);
            }
            Err(e) => {
                tracing::warn!("CUDA compiled but not available: {e}. Trying next backend...");
            }
        }
    }

    // 2. Try Metal
    #[cfg(feature = "metal")]
    {
        match candle_core::Device::new_metal(0) {
            Ok(_) => {
                tracing::info!("Metal GPU detected");
                // On Apple Silicon, GPU shares system RAM
                return (DeviceKind::Metal, ram_gb);
            }
            Err(e) => {
                tracing::warn!("Metal compiled but not available: {e}. Falling back to CPU...");
            }
        }
    }

    // 3. CPU fallback
    tracing::info!("Using CPU backend");
    (DeviceKind::Cpu, 0.0)
}
