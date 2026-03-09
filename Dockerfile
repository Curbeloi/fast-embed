# ── Stage 1: Build ────────────────────────────────────────────
FROM rust:1.82-bookworm AS builder

WORKDIR /build

# Cache dependencies first
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release 2>/dev/null || true && \
    rm -rf src

# Build real binary
COPY src/ src/
RUN cargo build --release && \
    strip target/release/fast-embed

# ── Stage 2: Runtime ──────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false fast-embed

WORKDIR /app

COPY --from=builder /build/target/release/fast-embed /app/fast-embed

# Data directories
RUN mkdir -p /app/data /app/models && chown -R fast-embed:fast-embed /app

USER fast-embed

EXPOSE 11435

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:11435/health || exit 1

ENTRYPOINT ["/app/fast-embed"]
CMD ["serve"]
