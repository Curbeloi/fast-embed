#!/bin/sh
# Fix ownership of mounted volumes (they may be created as root)
chown -R fast-embed:fast-embed /app/data /app/models 2>/dev/null || true

# Export env vars so they survive the su call
export HF_HOME="${HF_HOME:-/app/models}"

# Drop to non-root user and exec the binary, preserving environment
exec su -s /bin/sh -p fast-embed -c "HOME=/app HF_HOME=$HF_HOME /app/fast-embed $*"
