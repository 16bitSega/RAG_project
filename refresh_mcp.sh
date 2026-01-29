#!/usr/bin/env bash
set -euo pipefail
# Wrapper for scripts/refresh_mcp.sh (keeps a single source of truth).
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$ROOT_DIR/scripts/refresh_mcp.sh"
