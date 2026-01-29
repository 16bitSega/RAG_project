# Build & Rebuild the RAG Dataset (Deterministic)

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/rebuild_all.sh
streamlit run app.py
```

## Inputs

- Books: `data/raw_pdfs/` and `sources.json`
- Articles: `sources_articles.json`
- MCP docs (optional): `mcp/`

## Outputs (default: `data/normalized/`)

- `chunks_books.jsonl`, `manifest_books.json`
- `chunks_articles.jsonl`, `manifest_articles.json`
- `chunks.jsonl`, `manifest.json` (merged)

## Clean re-index

FAISS indexes are built by the app. To force a rebuild:

```bash
make clean-index
```

## Adding sources

### Add a book
1) Add PDF to `data/raw_pdfs/`
2) Add entry to `sources.json`
3) Rebuild: `bash scripts/rebuild_all.sh`

### Add an article
1) Add entry to `sources_articles.json`
2) Rebuild: `bash scripts/rebuild_all.sh`
