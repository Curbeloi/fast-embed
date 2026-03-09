#!/bin/bash
set -euo pipefail

# ── fast-embed production deploy script ──────────────────────
# Usage:
#   ./deploy.sh              # CPU (any VPS)
#   ./deploy.sh gpu          # CUDA GPU server

MODE="${1:-cpu}"
COMPOSE_FILE="docker-compose.yml"

if [ "$MODE" = "gpu" ]; then
    COMPOSE_FILE="docker-compose.gpu.yml"
    echo "→ Deploying with CUDA GPU support"
else
    echo "→ Deploying in CPU mode"
fi

# Build and start
docker compose -f "$COMPOSE_FILE" build
docker compose -f "$COMPOSE_FILE" up -d

# Wait for health
echo "→ Waiting for fast-embed to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:11435/health > /dev/null 2>&1; then
        echo "→ fast-embed is healthy!"
        break
    fi
    if [ "$i" = "30" ]; then
        echo "✗ fast-embed failed to start. Check logs:"
        echo "  docker compose -f $COMPOSE_FILE logs"
        exit 1
    fi
    sleep 2
done

# Create initial admin key if none exists
KEY_COUNT=$(docker compose -f "$COMPOSE_FILE" exec -T fast-embed /app/fast-embed keys list 2>/dev/null | grep -c "cg_" || true)
if [ "$KEY_COUNT" = "0" ]; then
    echo ""
    echo "→ No API keys found. Creating admin key..."
    docker compose -f "$COMPOSE_FILE" exec -T fast-embed /app/fast-embed keys create-admin
fi

echo ""
echo "→ fast-embed is running on http://localhost:11435"
echo ""
echo "  Useful commands:"
echo "    docker compose -f $COMPOSE_FILE logs -f     # View logs"
echo "    docker compose -f $COMPOSE_FILE restart      # Restart"
echo "    docker compose -f $COMPOSE_FILE down          # Stop"
echo "    docker compose -f $COMPOSE_FILE exec fast-embed /app/fast-embed keys list  # List keys"
