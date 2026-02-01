---
title: Agentic RAG (Agentic DP + AIMA + MCP)
author: O.O
sdk: streamlit
app_file: app.py
---

# Agentic RAG (FAISS + SentenceTransformers + Local LLM)

A Streamlit UI that answers questions over a local RAG corpus with a retrieval-only baseline.
It indexes chunk files with FAISS and retrieves across:
- Agentic Design Patterns (doc_id: `agentic_design_patterns`)
- AIMA (doc_id: `aima`)
- MCP markdowns (doc_id prefix: `mcp::`)
- Articles (doc_id prefix: `article::`)


## Preview without local installation is available at the:
https://huggingface.co/spaces/16bitSega/Agentic_RAG


## Quick start (local)

### 0) Prerequisites

- Python 3.11+
- A local LLM server (Ollama recommended)
- Network access for article ingestion and MCP refresh scripts

### 1) Create venv and install deps

macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Windows (CMD):

```bat
python -m venv .venv
.\.venv\Scripts\activate.bat
pip install -r requirements.txt
```

### 2) Install Ollama and pull a model

macOS (Homebrew):

```bash
brew install ollama
ollama serve
ollama pull llama3.2:1b
```

Windows:

1. Download and install Ollama from `https://ollama.com/download`.
2. Start Ollama, then run:

```powershell
ollama pull llama3.2:1b
```

The app uses:
- `RAG_OLLAMA_URL` (default `http://localhost:11434`)
- `RAG_OLLAMA_MODEL` (default `llama3.2:1b`)

### 3) Prepare sources

- Books: drop PDFs into `data/raw_pdfs/` and add entries to `sources.json`
- Articles: edit `sources_articles.json` (list of `{id,type,url,publisher}`)
- MCP docs (optional): `bash scripts/refresh_mcp.sh` (downloads the latest snapshot)

### 4) Build datasets

Recommended one-command rebuild:

```bash
make rebuild
```

Outputs to `data/normalized/`:
- `chunks_books.jsonl` + `manifest_books.json`
- `chunks_articles.jsonl` + `manifest_articles.json`
- `chunks.jsonl` + `manifest.json` (merged)

### 5) Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` and keep `ollama serve` running. On first run, the app builds FAISS indexes:
- `data/normalized/index_books.faiss`
- `data/normalized/index_articles.faiss`

## Configuration

You can override defaults via environment variables:

```bash
export RAG_BOOK_CHUNKS_PATH=data/normalized/chunks_books.jsonl
export RAG_ARTICLE_CHUNKS_PATH=data/normalized/chunks_articles.jsonl
export RAG_BOOK_INDEX_PATH=data/normalized/index_books.faiss
export RAG_ARTICLE_INDEX_PATH=data/normalized/index_articles.faiss
export RAG_BOOK_MANIFEST_PATH=data/normalized/manifest_books.json
export RAG_ARTICLE_MANIFEST_PATH=data/normalized/manifest_articles.json
export RAG_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
export RAG_OLLAMA_URL=http://localhost:11434
export RAG_OLLAMA_MODEL=llama3.2:1b
export RAG_OUT_DIR=data/normalized
export RAG_ARTICLE_SOURCES=sources_articles.json
```

## Common maintenance tasks

### Add new books (PDFs)

1. Add PDFs to `data/raw_pdfs/`
2. Update `sources.json`
3. Run `make rebuild`
4. (Optional) `make clean-index`
5. `streamlit run app.py`

### Add new articles

1. Update `sources_articles.json`
2. Run `make rebuild`
3. (Optional) `make clean-index`
4. `streamlit run app.py`

### Rebuild indexes only

```bash
make clean-index
```

## Scripts and commands reference

- `app.py` - Streamlit UI; loads chunk files and builds/loads FAISS indexes.
- `scripts/normalize_all.py` - Parse PDFs and MCP markdowns into `chunks_books.jsonl` and `manifest_books.json`.
- `scripts/ingest_articles.py` - Fetch URLs from `sources_articles.json` and write `chunks_articles.jsonl` and `manifest_articles.json` plus `articles_ingest_report.json`.
- `scripts/merge_chunks.py` - Merge multiple chunk files and manifests; emits `chunks.jsonl`, `manifest.json`, and `merge_report.json`.
- `scripts/rebuild_all.sh` - Run normalize, ingest, and merge in order (same as `make rebuild`).
- `scripts/refresh_mcp.sh` - Download `llms-full.txt` and regenerate MCP markdowns in `mcp/`.
- `scripts/split_mcp.py` - Split a single MCP snapshot text file into topic markdown files.
- `refresh_mcp.sh` - Convenience wrapper for `scripts/refresh_mcp.sh`.
- `normalize_all.py`, `ingest_articles.py`, `merge_chunks.py`, `split_mcp.py` - Convenience wrappers for the `scripts/` versions.
- `Makefile` - `make install`, `make rebuild`, `make clean-index`, `make run`.
- `build_kb.py` - Legacy entry point referencing a removed `src/` package; not used by the current app.

## License

Apache License 2.0. See `LICENSE`.

## Troubleshooting

- If you see `No chunks loaded`, ensure `data/normalized/*.jsonl` exists and has content.
- If Ollama fails, verify `ollama serve` is running and the model name exists.
- If article ingestion skips sources, check `data/normalized/articles_ingest_report.json`.
