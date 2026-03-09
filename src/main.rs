mod config;
mod hardware;
mod inference;
mod middleware;
mod routes;
mod state;
mod store;

use axum::{
    middleware as axum_mw,
    routing::{delete, get, post},
    Router,
};
use clap::{Parser, Subcommand};
use std::sync::Arc;
use std::time::Instant;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

use crate::config::AppConfig;
use crate::middleware::rate_limit;
use crate::state::AppState;

#[derive(Parser, Debug)]
#[command(
    name = "fast-embed",
    version,
    about = "Fast Embed — Private embedding inference server",
    long_about = "fast-embed is a high-performance embedding server built in Rust.\n\
                  It provides OpenAI-compatible embedding endpoints with built-in auth,\n\
                  rate limiting, and hardware auto-detection (CPU/Metal/CUDA)."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the inference server
    Serve {
        /// Port to listen on (overrides BIND_ADDRESS)
        #[arg(short, long)]
        port: Option<u16>,

        /// Bind address (overrides BIND_ADDRESS)
        #[arg(short, long)]
        bind: Option<String>,
    },

    /// Manage API keys
    #[command(subcommand)]
    Keys(KeysCommand),

    /// Show detected hardware and computed limits
    Info,
}

#[derive(Subcommand, Debug)]
enum KeysCommand {
    /// Create a new API key for a tenant
    Create {
        /// Tenant identifier
        #[arg(short, long)]
        tenant: String,

        /// Descriptive name for the key
        #[arg(short, long)]
        name: Option<String>,

        /// Override rate limit (requests per minute)
        #[arg(short, long)]
        rate_limit: Option<u32>,
    },

    /// Create an admin API key
    CreateAdmin,

    /// List all API keys
    List,

    /// Revoke an API key by ID
    Revoke {
        /// Key ID to revoke
        id: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = dotenvy::dotenv();
    let cfg = AppConfig::from_env();

    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(&cfg.log_level)),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Serve { port, bind } => cmd_serve(cfg, port, bind).await,
        Commands::Keys(keys_cmd) => cmd_keys(cfg, keys_cmd),
        Commands::Info => cmd_info(),
    }
}

// ── fast-embed serve ────────────────────────────────────────────────

async fn cmd_serve(cfg: AppConfig, port: Option<u16>, bind: Option<String>) -> anyhow::Result<()> {
    let db_pool = store::init_pool(&cfg.db_path)?;
    tracing::info!("Database initialized at {}", cfg.db_path);

    let hardware = Arc::new(hardware::detect());
    tracing::info!(
        "Hardware: {} (VRAM: {:.1} GB, RAM: {:.1} GB, {} cores)",
        hardware.device, hardware.vram_gb, hardware.ram_gb, hardware.cpu_cores
    );
    tracing::info!(
        "Limits: {} concurrent, {} req/min/key, {} queue",
        hardware.max_concurrent, hardware.max_per_minute, hardware.max_queue
    );

    let engine = Arc::new(inference::InferenceEngine::new(
        hardware.clone(),
        cfg.model_cache_dir.clone(),
        cfg.default_embed_model.clone(),
        cfg.hf_token.clone(),
    ));

    let db_pool = Arc::new(db_pool);
    let app_state = Arc::new(AppState {
        db_pool: db_pool.clone(),
        hardware: hardware.clone(),
        engine: engine.clone(),
        start_time: Instant::now(),
    });

    let rate_limit_store = rate_limit::new_store();

    let public_routes = Router::new()
        .route("/health", get(routes::health::health_handler));

    let inference_routes = Router::new()
        .route("/v1/embeddings", post(routes::embeddings::embeddings_handler))
        .layer(axum_mw::from_fn(crate::middleware::metering::metering_middleware))
        .layer(axum_mw::from_fn(crate::middleware::rate_limit::rate_limit_middleware))
        .layer(axum_mw::from_fn(crate::middleware::auth::auth_middleware));

    let admin_routes = Router::new()
        .route("/admin/stats", get(routes::admin::stats_handler))
        .route("/admin/stats/{tenant_id}", get(routes::admin::tenant_stats_handler))
        .route("/admin/keys", post(routes::admin::create_key_handler))
        .route("/admin/keys", get(routes::admin::list_keys_handler))
        .route("/admin/keys/{id}", delete(routes::admin::revoke_key_handler))
        .layer(axum_mw::from_fn(crate::middleware::auth::admin_auth_middleware));

    let app = Router::new()
        .merge(public_routes)
        .merge(inference_routes)
        .merge(admin_routes)
        .layer(axum::Extension(db_pool))
        .layer(axum::Extension(hardware))
        .layer(axum::Extension(rate_limit_store))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(app_state);

    // Resolve bind address: CLI flags > env > default
    let bind_address = if let Some(b) = bind {
        b
    } else if let Some(p) = port {
        format!("0.0.0.0:{}", p)
    } else {
        cfg.bind_address
    };

    let listener = tokio::net::TcpListener::bind(&bind_address).await?;
    tracing::info!("fast-embed running on http://{}", bind_address);

    axum::serve(listener, app).await?;
    Ok(())
}

// ── fast-embed keys ─────────────────────────────────────────────────

fn cmd_keys(cfg: AppConfig, cmd: KeysCommand) -> anyhow::Result<()> {
    let db_pool = store::init_pool(&cfg.db_path)?;

    match cmd {
        KeysCommand::CreateAdmin => {
            let (raw_key, info) = store::keys::create_admin_key(&db_pool)?;
            println!();
            println!("  Admin key created");
            println!("  ─────────────────────────────────────");
            println!("  ID:    {}", info.id);
            println!("  Key:   {}", raw_key);
            println!("  ─────────────────────────────────────");
            println!("  ⚠ Save this key now — it won't be shown again.");
            println!();
        }
        KeysCommand::Create { tenant, name, rate_limit } => {
            let key_name = name.unwrap_or_else(|| format!("{} API Key", tenant));
            let (raw_key, info) = store::keys::create_key(&tenant, &key_name, rate_limit, &db_pool)?;
            println!();
            println!("  API key created");
            println!("  ─────────────────────────────────────");
            println!("  ID:       {}", info.id);
            println!("  Tenant:   {}", info.tenant_id);
            println!("  Name:     {}", info.name);
            println!("  Key:      {}", raw_key);
            if let Some(rl) = rate_limit {
                println!("  Limit:    {} req/min (override)", rl);
            } else {
                println!("  Limit:    auto (based on hardware)");
            }
            println!("  ─────────────────────────────────────");
            println!("  ⚠ Save this key now — it won't be shown again.");
            println!();
        }
        KeysCommand::List => {
            let keys = store::keys::list_keys(&db_pool)?;
            if keys.is_empty() {
                println!("\n  No API keys found. Create one with: fast-embed keys create-admin\n");
                return Ok(());
            }
            println!();
            println!("  {:<38} {:<15} {:<25} {:<8}", "ID", "TENANT", "NAME", "ACTIVE");
            println!("  {}", "─".repeat(88));
            for k in &keys {
                println!(
                    "  {:<38} {:<15} {:<25} {:<8}",
                    k.id, k.tenant_id, k.name,
                    if k.active { "yes" } else { "revoked" }
                );
            }
            println!("\n  Total: {} keys\n", keys.len());
        }
        KeysCommand::Revoke { id } => {
            if store::keys::revoke_key(&id, &db_pool)? {
                println!("\n  Key {} revoked successfully.\n", id);
            } else {
                println!("\n  Key {} not found or already revoked.\n", id);
            }
        }
    }
    Ok(())
}

// ── fast-embed info ─────────────────────────────────────────────────

fn cmd_info() -> anyhow::Result<()> {
    let hw = hardware::detect();

    println!();
    println!("  fast-embed v{}", env!("CARGO_PKG_VERSION"));
    println!("  Fast Embed — Private embedding inference server");
    println!("  ─────────────────────────────────────");
    println!("  Device:          {}", hw.device);
    println!("  VRAM:            {:.1} GB", hw.vram_gb);
    println!("  RAM:             {:.1} GB", hw.ram_gb);
    println!("  CPU cores:       {}", hw.cpu_cores);
    println!("  ─────────────────────────────────────");
    println!("  Max concurrent:  {}", hw.max_concurrent);
    println!("  Max req/min/key: {}", hw.max_per_minute);
    println!("  Max queue:       {}", hw.max_queue);
    println!("  Max batch tokens:{}", hw.max_batch_tokens);
    println!();

    Ok(())
}
