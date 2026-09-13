#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

./scripts/build.sh
podman compose down 2>/dev/null || true
podman compose up -d

echo "Waiting for http://127.0.0.1:8877/health..."
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8877/health >/dev/null 2>&1; then
    echo "Healthcheck OK."
    exit 0
  fi
  sleep 1
done

echo "Healthcheck timed out after 30s."
exit 1
