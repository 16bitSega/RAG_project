#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${RAG_OUT_DIR:-data/normalized}"

echo "[1/3] Normalize books + MCP -> $OUT_DIR"
RAG_OUT_DIR="$OUT_DIR" python scripts/normalize_all.py

echo "[2/3] Ingest articles -> $OUT_DIR"
RAG_OUT_DIR="$OUT_DIR" python scripts/ingest_articles.py

echo "[3/3] Merge chunks + manifests -> $OUT_DIR"
python scripts/merge_chunks.py \
  --out "$OUT_DIR/chunks.jsonl" \
  --out-manifest "$OUT_DIR/manifest.json" \
  --manifest "$OUT_DIR/manifest_books.json" \
  --manifest "$OUT_DIR/manifest_articles.json" \
  "$OUT_DIR/chunks_books.jsonl" \
  "$OUT_DIR/chunks_articles.jsonl"

echo "Done. To force FAISS rebuild: rm -f $OUT_DIR/index_*.faiss (or 'make clean-index')."
