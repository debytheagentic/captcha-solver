#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

REPO="debytheagentic/captcha-solver"

RUN_ID=$(deby gh run list --repo "$REPO" --status success --limit 1 --json databaseId -q '.[0].databaseId')

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

deby gh run download "$RUN_ID" --repo "$REPO" -n captcha-solver-image -D "$TMP_DIR"

podman load -i "$TMP_DIR/captcha-solver-image.tar.gz"
podman tag captcha-solver:latest localhost/captcha-solver:latest