"""Fetch an IAM line-level mirror from the Hugging Face hub, for reference use.

    pip install -U datasets pillow
    python scripts/download_iam_handwriting.py
    python scripts/download_iam_handwriting.py --limit 500      # a quick look
    python scripts/download_iam_handwriting.py --repo Teklia/IAM-line

Writes data/extracted/iam_hf/<split>/ as images plus a labels.txt, which the
generic folder-label adapter already reads, so the normal pipeline picks it up:

    python scripts/11_build_manifests.py --datasets iam_hf
    python scripts/12_train_htr.py --datasets iam_hf --level line

-------------------------------------------------------------------------
LICENCE - read this before using the result for anything that ships
-------------------------------------------------------------------------
The IAM Handwriting Database is **CC BY-NC 4.0: non-commercial research
only**, and its own distribution terms ask you to register at
https://fki.tic.heia-fr.ch/databases/iam-handwriting-database.

This script pulls a third-party mirror instead, which is why the registry
entry for `iam` still says MANUAL and still points at the original. Use what
this produces for benchmarking and comparison against published IAM numbers.
A model trained on it is not something a commercial firm can ship.

The script records the mirror, the date and the licence in
data/extracted/iam_hf/PROVENANCE.json so that the origin of any model trained
on it stays recoverable.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from htr.paths import ROOT, extracted_dir

DEFAULT_REPO = "Teklia/IAM-line"
LICENCE = "CC BY-NC 4.0 - non-commercial research only"
ORIGINAL = "https://fki.tic.heia-fr.ch/databases/iam-handwriting-database"

IMAGE_KEYS = ("image", "img", "picture", "line_image")
TEXT_KEYS = ("text", "transcription", "label", "sentence", "ground_truth")


def _pick(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per split")
    ap.add_argument("--dataset-id", default="iam_hf",
                    help="directory under data/extracted/ to write into")
    ap.add_argument("--ca-bundle", default=None,
                    help="TLS bundle; defaults to the one paths.ca_bundle finds")
    args = ap.parse_args()

    # The corporate proxy intercepts TLS, and huggingface_hub reads these two
    # variables rather than anything we can pass it.
    import os
    from htr.paths import ca_bundle
    bundle = args.ca_bundle or ca_bundle()
    if isinstance(bundle, str):
        os.environ.setdefault("REQUESTS_CA_BUNDLE", bundle)
        os.environ.setdefault("SSL_CERT_FILE", bundle)
        os.environ.setdefault("CURL_CA_BUNDLE", bundle)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        detail = str(exc)
        if "numpy" in detail or "Application Control" in detail:
            # Distinguish the two causes: "install it" is useless advice when
            # the package is installed and the OS is refusing to load its DLL.
            print("STOPPING. A dependency is installed but cannot be loaded:")
            print(f"  {detail.splitlines()[0]}")
            print("\nIf this mentions an Application Control policy, Windows "
                  "Smart App Control is blocking the unsigned native "
                  "extensions in .venv (numpy, torch, pyarrow). Turn it off "
                  "under Windows Security > App & browser control, or have IT "
                  "exempt this directory.")
        else:
            print("STOPPING. The datasets library is not installed.")
            print("  .venv/Scripts/python.exe -m pip install -U datasets pillow")
        return 2

    print(f"repo    : {args.repo}")
    print(f"licence : {LICENCE}")
    print(f"original: {ORIGINAL}")
    print("\nThis is a third-party mirror. For anything beyond benchmarking, "
          "register with the original provider.\n")

    dataset = load_dataset(args.repo)
    root = extracted_dir(args.dataset_id)

    written: dict[str, int] = {}
    columns_used: dict[str, str] = {}

    for split, rows in dataset.items():
        columns = list(rows.column_names)
        image_key = _pick(columns, IMAGE_KEYS)
        text_key = _pick(columns, TEXT_KEYS)
        if image_key is None or text_key is None:
            print(f"  {split}: skipped - no image/text pair in {columns}")
            continue
        columns_used = {"image": image_key, "text": text_key}

        # IAM's own split is writer-disjoint, so keep its names: the pipeline
        # honours an official split rather than re-cutting one.
        name = {"val": "validation", "valid": "validation",
                "dev": "validation"}.get(split, split)
        out = root / name
        out.mkdir(parents=True, exist_ok=True)
        labels = out / "labels.txt"

        count = 0
        with labels.open("w", encoding="utf-8", newline="\n") as handle:
            total = len(rows) if args.limit is None else min(args.limit, len(rows))
            for index in range(total):
                row = rows[index]
                text = str(row[text_key]).strip()
                if not text:
                    continue
                image = row[image_key]
                if not hasattr(image, "save"):
                    continue
                filename = f"{name}_{index:06d}.png"
                image.convert("L").save(out / filename)
                # Tab-separated: IAM transcriptions contain spaces.
                handle.write(f"{filename}\t{text}\n")
                count += 1
                if count % 500 == 0:
                    print(f"  {name}: {count:,}/{total:,}", flush=True)
        written[name] = count
        print(f"  {name}: {count:,} lines -> {out}")

    provenance = {
        "dataset_id": args.dataset_id,
        "source": "huggingface",
        "repo": args.repo,
        "url": f"https://huggingface.co/datasets/{args.repo}",
        "original_provider": ORIGINAL,
        "licence": LICENCE,
        "commercial_use": False,
        "note": ("Third-party mirror of the IAM Handwriting Database. The "
                 "original requires registration; this copy does not. Use for "
                 "benchmarking only."),
        "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "columns": columns_used,
        "splits": written,
    }
    (root / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8")

    total = sum(written.values())
    print(f"\n{total:,} lines written to {root}")
    print(f"provenance: {root / 'PROVENANCE.json'}")
    print("\nnext:")
    print(f"  .venv/Scripts/python.exe scripts/11_build_manifests.py "
          f"--datasets {args.dataset_id}")
    print(f"  .venv/Scripts/python.exe scripts/12_train_htr.py "
          f"--datasets {args.dataset_id} --level line")
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
