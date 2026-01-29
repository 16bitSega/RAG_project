#!/usr/bin/env bash
set -euo pipefail
# Refresh MCP snapshot (pulls latest llms-full.txt) and regenerates curated markdown files
OUT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
URL="https://modelcontextprotocol.io/llms-full.txt"
TMP="$OUT_DIR/.tmp"
mkdir -p "$TMP" "$OUT_DIR/mcp"
curl -fsSL "$URL" -o "$TMP/mcp_llms-full.txt"
python3 "$OUT_DIR/scripts/split_mcp.py" "$TMP/mcp_llms-full.txt" "$OUT_DIR/mcp"
echo "Refreshed MCP into $OUT_DIR/mcp"
