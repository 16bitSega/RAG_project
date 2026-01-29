#!/usr/bin/env python3
"""Compatibility wrapper for scripts/ingest_articles.py."""
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "scripts" / "ingest_articles.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
