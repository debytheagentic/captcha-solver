#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

podman build --format docker -t localhost/captcha-solver:latest .
podman image prune -f --external 2>/dev/null || podman image prune -f || true
