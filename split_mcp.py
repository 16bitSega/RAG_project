#!/usr/bin/env python3
"""Compatibility wrapper for scripts/split_mcp.py."""
from pathlib import Path
import runpy

SCRIPT = Path(__file__).resolve().parent / "scripts" / "split_mcp.py"
runpy.run_path(str(SCRIPT), run_name="__main__")
