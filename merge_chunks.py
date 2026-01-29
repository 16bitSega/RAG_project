#!/usr/bin/env python3
"""Compatibility wrapper for scripts/merge_chunks.py."""
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "scripts" / "merge_chunks.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
