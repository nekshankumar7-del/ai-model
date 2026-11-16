"""Canonical project paths. Import this instead of building paths by hand."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_FILE = ROOT / "knowledge" / "fact-find-knowledge.json"
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
SPLITS_FILE = DATA_DIR / "splits.json"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
ANGULAR_DIR = ROOT / "angular"

for _d in (DATA_DIR, DATASET_DIR, SYNTHETIC_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, ANGULAR_DIR):
    _d.mkdir(parents=True, exist_ok=True)
