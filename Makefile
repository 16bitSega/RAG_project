PYTHON ?= python
OUT_DIR ?= data/normalized

.PHONY: help install normalize articles merge rebuild clean-index run

help:
	@echo "Targets:"
	@echo "  make install      - install python deps"
	@echo "  make rebuild      - normalize books+mcp, ingest articles, merge manifests/chunks"
	@echo "  make clean-index  - remove FAISS indexes so they rebuild"
	@echo "  make run          - run the Streamlit app"

install:
	$(PYTHON) -m pip install -r requirements.txt

normalize:
	RAG_OUT_DIR=$(OUT_DIR) $(PYTHON) scripts/normalize_all.py

articles:
	RAG_OUT_DIR=$(OUT_DIR) $(PYTHON) scripts/ingest_articles.py

merge:
	$(PYTHON) scripts/merge_chunks.py \
	  --out $(OUT_DIR)/chunks.jsonl \
	  --out-manifest $(OUT_DIR)/manifest.json \
	  --manifest $(OUT_DIR)/manifest_books.json \
	  --manifest $(OUT_DIR)/manifest_articles.json \
	  $(OUT_DIR)/chunks_books.jsonl \
	  $(OUT_DIR)/chunks_articles.jsonl

rebuild: normalize articles merge

clean-index:
	rm -f $(OUT_DIR)/index_books.faiss $(OUT_DIR)/index_articles.faiss

run:
	streamlit run app.py
