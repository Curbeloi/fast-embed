# fast-embed

A high-performance, private embedding inference server written in Rust. Runs locally and provides a REST API for generating text embeddings with built-in authentication, rate limiting, and automatic hardware detection.

## Features

- **Local REST API** — Private embedding endpoint via `/v1/embeddings`
- **Hardware auto-detection** — Automatically selects and optimizes for CPU, Apple Metal, or NVIDIA CUDA
- **Multi-tenant** — API key management with per-tenant isolation and usage tracking
- **Rate limiting** — Per-key sliding window rate limiter, auto-tuned to hardware capacity
- **Request logging** — Full request audit trail with per-tenant statistics
- **Lazy model loading** — Model downloads from Hugging Face Hub on first request
- **Sub-batch processing** — Bounds GPU memory usage to prevent OOM errors
- **F16 inference** — Half-precision for reduced memory footprint and faster GPU throughput

## Embedding Model: nomic-embed-text-v1.5

fast-embed is built around the **[nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)** model by Nomic AI, with a custom Rust implementation of the NomicBERT architecture using [Candle](https://github.com/huggingface/candle) (Meta's ML framework for Rust).

### Model Characteristics

| Property | Value |
|---|---|
| Architecture | NomicBERT (optimized BERT variant) |
| Parameters | ~137M |
| Embedding Dimensions | 768 |
| Max Sequence Length | 2048 tokens |
| Precision | F16 (half-precision) |
| Pooling Strategy | Masked mean pooling + L2 normalization |
| Source | [Hugging Face Hub](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) |

### Architecture Details

The NomicBERT architecture includes several improvements over standard BERT:

- **Rotary Position Embeddings (RoPE)** — Enables better length extrapolation compared to absolute position embeddings
- **SwiGLU Activation** — Used in MLP blocks for improved training dynamics
- **Fused QKV Projections** — Single linear projection for queries, keys, and values (no bias terms)
- **Pre-Normalization** — LayerNorm applied before attention and MLP blocks (instead of after), improving training stability
- **Token Type Embeddings** — Supports sentence pair inputs

## API Endpoints

### Public

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check with hardware info and uptime |

### Inference (requires API key via `X-API-Key` header)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/v1/embeddings` | Generate text embeddings |

**Request:**
```json
{
  "input": "Your text here",
  "model": "nomic-embed-text-v1.5"
}
```

Batch request:
```json
{
  "input": ["First text", "Second text", "Third text"],
  "model": "nomic-embed-text-v1.5"
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0091, ...]
    }
  ],
  "model": "nomic-embed-text-v1.5",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

### Admin (requires admin API key)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/admin/keys` | Create a tenant API key |
| `GET` | `/admin/keys` | List all API keys |
| `DELETE` | `/admin/keys/{id}` | Revoke an API key |
| `GET` | `/admin/stats` | Global usage statistics |
| `GET` | `/admin/stats/{tenant_id}` | Per-tenant statistics |

## Configuration

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `BIND_ADDRESS` | `0.0.0.0:11435` | Server listen address |
| `LOG_LEVEL` | `info` | Log level (`debug`, `info`, `warn`, `error`) |
| `DEFAULT_EMBED_MODEL` | `nomic-embed-text-v1.5` | Model identifier |
| `MODEL_CACHE_DIR` | `./models` | Directory for cached model files |
| `DB_PATH` | `./data/inference.db` | SQLite database path |
| `HF_TOKEN` | — | Hugging Face API token (for private models) |
| `ADMIN_KEY_HASH` | — | Pre-set admin key (SHA256 hash) |
| `OVERRIDE_MAX_CONCURRENT` | auto | Override auto-detected max concurrent requests |
| `OVERRIDE_MAX_PER_MINUTE` | auto | Override auto-detected rate limit per key |

Copy `.env.example` to `.env` and adjust as needed.

### Hardware Auto-Detection

fast-embed automatically detects available hardware and computes optimal limits:

| Device | Max Concurrent | Max Req/Min/Key | Max Queue |
|---|---|---|---|
| CUDA | `factor × 8` | `factor × 50` | `factor × 25` |
| Metal | `factor × 4` | `factor × 30` | `factor × 15` |
| CPU | `cores / 2` | `cores × 8` | `cores × 4` |

- CUDA factor: `max(vram_gb / 8.0, 1)`
- Metal factor: `max(vram_gb / 4.0, 1)` (shares system RAM on Apple Silicon)
- Max batch tokens: 8192 (constant across all devices)

Run `fast-embed info` to see detected hardware and computed limits.

## Getting Started

### Prerequisites

- Rust 1.82+ (for building from source)
- Docker (optional, for containerized deployment)
- NVIDIA GPU + CUDA toolkit (optional, for GPU acceleration)

### Build from Source

```bash
# CPU only
cargo build --release

# Apple Metal (macOS)
cargo build --release --features metal

# NVIDIA CUDA
cargo build --release --features cuda
```

### Run

```bash
# Start the server
./target/release/fast-embed serve

# Start on a specific port
./target/release/fast-embed serve --port 8080

# Show hardware info
./target/release/fast-embed info
```

### API Key Management (CLI)

```bash
# Create an admin key
fast-embed keys create-admin

# Create a tenant key
fast-embed keys create --tenant my-app --name "My App Key"

# Create a tenant key with custom rate limit
fast-embed keys create --tenant my-app --rate-limit 100

# List all keys
fast-embed keys list

# Revoke a key
fast-embed keys revoke <key-id>
```

### Docker

**CPU:**
```bash
docker compose up -d --build
```

**GPU (NVIDIA CUDA):**
```bash
docker compose -f docker-compose.gpu.yml up -d --build
```

### Deploy Script

The included `deploy.sh` automates the full deployment:

```bash
# CPU deployment
./deploy.sh

# GPU deployment
./deploy.sh gpu
```

The script builds the container, waits for the health check, and auto-creates an admin API key if none exist.

## Project Structure

```
fast-embed/
├── src/
│   ├── main.rs                 # CLI & server entry point
│   ├── config.rs               # Configuration from environment
│   ├── state.rs                # Shared application state
│   ├── hardware.rs             # Device detection & limit computation
│   ├── inference/
│   │   ├── mod.rs              # Inference engine
│   │   ├── embeddings.rs       # Model loading & embedding generation
│   │   └── nomic_bert.rs       # NomicBERT model implementation
│   ├── middleware/
│   │   ├── auth.rs             # API key authentication
│   │   ├── rate_limit.rs       # Per-key sliding window rate limiter
│   │   └── metering.rs         # Request logging & metrics
│   ├── routes/
│   │   ├── health.rs           # Health check endpoint
│   │   ├── embeddings.rs       # /v1/embeddings handler
│   │   └── admin.rs            # Admin key & stats endpoints
│   └── store/
│       ├── mod.rs              # SQLite pool & migrations
│       ├── keys.rs             # API key CRUD
│       └── logs.rs             # Request log storage & queries
├── migrations/
│   ├── 001_create_keys.sql     # API key schema
│   └── 002_create_logs.sql     # Request log schema
├── config.yaml                 # Configuration file
├── Dockerfile                  # CPU container build
├── Dockerfile.cuda             # CUDA container build
├── docker-compose.yml          # CPU composition
├── docker-compose.gpu.yml      # GPU composition
└── deploy.sh                   # Deployment script
```

## Tech Stack

- **Rust** — Systems programming language
- **Axum 0.7** — Async web framework
- **Candle 0.8** — ML inference framework (by Hugging Face)
- **Tokio** — Async runtime
- **SQLite** (rusqlite) — Embedded database with R2D2 connection pooling
- **HF Hub** — Hugging Face model downloads
- **Tokenizers** — BPE tokenization

## License

Private project. All rights reserved.
