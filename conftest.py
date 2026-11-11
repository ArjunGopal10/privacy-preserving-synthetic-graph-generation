"""Pytest root configuration: makes the ``src`` package importable from the repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
