"""Canonical paths for the public-dataset HTR pipeline.

Layout, as required by the brief:

    data/
      raw/        downloaded archives, one directory per dataset id
      extracted/  unpacked archives, one directory per dataset id
      processed/  normalised crops + per-dataset manifests
      metadata/   the dataset inventory and download state
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EXTRACTED_DIR = DATA_DIR / "extracted"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

INVENTORY_FILE = METADATA_DIR / "inventory.json"
MANIFEST_DIR = PROCESSED_DIR / "manifests"
SPLITS_FILE = METADATA_DIR / "htr_splits.json"

MODELS_DIR = ROOT / "models" / "htr"
REPORTS_DIR = ROOT / "reports"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, EXTRACTED_DIR, PROCESSED_DIR, METADATA_DIR,
              MANIFEST_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def raw_dir(dataset_id: str) -> Path:
    p = RAW_DIR / dataset_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def extracted_dir(dataset_id: str) -> Path:
    p = EXTRACTED_DIR / dataset_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def processed_dir(dataset_id: str) -> Path:
    p = PROCESSED_DIR / dataset_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def manifest_path(dataset_id: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    return MANIFEST_DIR / f"{dataset_id}.jsonl"


def ca_bundle() -> str | bool:
    """TLS verification target.

    This machine sits behind a TLS-intercepting corporate proxy, so the system
    trust store alone fails on every dataset host. Honour the standard
    requests/curl variables first, then fall back to the bundle that already
    ships in the virtualenv.
    """
    for var in ("HTR_CA_BUNDLE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE"):
        value = os.environ.get(var)
        if value and Path(value).exists():
            return value
    local = ROOT / ".venv" / "corp-ca-bundle.pem"
    if local.exists():
        return str(local)
    return True


ensure_dirs()
