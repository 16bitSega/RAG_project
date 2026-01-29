#!/usr/bin/env python3
"""
merge_chunks.py

Merge multiple JSONL chunk files into a single chunks.jsonl, with safety checks:
- validates required fields
- fails on duplicate chunk_id
- optional: merges manifests into one manifest.json
- outputs a merge_report.json (doc_id counts, totals, sources)

Usage examples:

  # simplest: merge two chunk files
  python scripts/merge_chunks.py \
    --out data/normalized_v1_1/chunks.jsonl \
    data/normalized_v1_1/chunks_books.jsonl \
    data/normalized_v1_1/chunks_articles.jsonl

  # also merge manifests (optional)
  python scripts/merge_chunks.py \
    --out data/normalized_v1_1/chunks.jsonl \
    --out-manifest data/normalized_v1_1/manifest.json \
    --manifest data/normalized_v1_1/manifest_books.json \
    --manifest data/normalized_v1_1/manifest_articles.json \
    data/normalized_v1_1/chunks_books.jsonl \
    data/normalized_v1_1/chunks_articles.jsonl
"""

import argparse
import json
import os
from collections import Counter
from datetime import datetime
from typing import Dict, Iterable, List, Optional


REQUIRED_FIELDS = ("chunk_id", "doc_id", "doc_title", "text")


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON in {path} at line {line_no}: {e}") from e
            yield obj


def validate_record(obj: Dict, path: str) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in obj or obj[k] in (None, "")]
    if missing:
        raise RuntimeError(f"Missing required fields {missing} in {path} for chunk_id={obj.get('chunk_id')!r}")


def load_manifest(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_manifests(manifest_paths: List[str]) -> Dict:
    """
    Assumes each manifest has:
      {
        "generated_at": "...",
        "documents": [ { "id": ..., "title": ..., "format": ..., "filename": ..., ... }, ... ]
      }
    We combine documents by `id` (first wins, duplicates skipped).
    """
    combined_docs = []
    seen_ids = set()

    for p in manifest_paths:
        mf = load_manifest(p)
        for doc in mf.get("documents", []):
            doc_id = doc.get("id")
            if not doc_id:
                continue
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            combined_docs.append(doc)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "documents": combined_docs,
        "sources": manifest_paths,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output merged chunks.jsonl path")
    ap.add_argument("--out-manifest", default=None, help="Output merged manifest.json path (optional)")
    ap.add_argument("--manifest", action="append", default=[], help="Input manifest.json path (repeatable, optional)")
    ap.add_argument("--out-report", default=None, help="Output merge report JSON path (default: <out_dir>/merge_report.json)")
    ap.add_argument("--allow-duplicate-chunk-id", action="store_true",
                    help="If set, duplicate chunk_id records are skipped instead of failing (NOT recommended).")
    ap.add_argument("inputs", nargs="+", help="Input chunk JSONL files to merge, in order")
    args = ap.parse_args()

    out_path = args.out
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    report_path = args.out_report or os.path.join(out_dir, "merge_report.json")

    seen_chunk_ids = set()
    doc_counts = Counter()
    total_written = 0
    inputs_summary = []

    with open(out_path, "w", encoding="utf-8") as out_f:
        for in_path in args.inputs:
            if not os.path.exists(in_path):
                raise FileNotFoundError(in_path)

            in_written = 0
            in_skipped_dupes = 0

            for obj in iter_jsonl(in_path):
                validate_record(obj, in_path)

                cid = str(obj["chunk_id"])
                if cid in seen_chunk_ids:
                    if args.allow_duplicate_chunk_id:
                        in_skipped_dupes += 1
                        continue
                    raise RuntimeError(f"Duplicate chunk_id detected: {cid} (from {in_path}). Aborting.")

                seen_chunk_ids.add(cid)

                out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                total_written += 1
                in_written += 1
                doc_counts[str(obj["doc_id"])] += 1

            inputs_summary.append({
                "input": in_path,
                "written": in_written,
                "skipped_duplicate_chunk_id": in_skipped_dupes,
            })

    # Optional: merged manifest
    if args.out_manifest:
        if not args.manifest:
            raise RuntimeError("--out-manifest provided but no --manifest inputs were provided.")
        merged = merge_manifests(args.manifest)
        with open(args.out_manifest, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, ensure_ascii=False)

    # Merge report
    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "out_chunks": out_path,
        "out_manifest": args.out_manifest,
        "total_chunks_written": total_written,
        "unique_chunk_ids": len(seen_chunk_ids),
        "doc_id_counts": dict(doc_counts),
        "inputs": inputs_summary,
        "notes": [
            "If total_chunks_written != unique_chunk_ids, you allowed duplicate chunk_id skipping.",
            "Pin this merged file as your dataset v1.1.0 before embedding/indexing.",
        ],
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[OK] Wrote merged chunks: {out_path}")
    if args.out_manifest:
        print(f"[OK] Wrote merged manifest: {args.out_manifest}")
    print(f"[OK] Wrote merge report: {report_path}")
    print(f"[OK] Total chunks: {total_written} | Unique chunk_ids: {len(seen_chunk_ids)}")


if __name__ == "__main__":
    main()
