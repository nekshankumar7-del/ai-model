"""Step 2 - convert extracted datasets into validated manifests.

    python scripts/11_build_manifests.py
    python scripts/11_build_manifests.py --datasets emnist --limit 50000
    python scripts/11_build_manifests.py --datasets all --no-dedupe

Writes data/processed/manifests/<id>.jsonl and data/metadata/build_report.json.
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from htr import registry
from htr.build import BUILD_REPORT, build
from htr.download import load_inventory


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset ids, or 'all'. Default: everything already "
                         "extracted on disk.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap samples per dataset (useful for a smoke run)")
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--no-image-check", action="store_true",
                    help="skip decoding every image; much faster, much less safe")
    ap.add_argument("--max-text-len", type=int, default=96)
    args = ap.parse_args()

    if args.datasets:
        ids = [s.id for s in registry.select(args.datasets)]
    else:
        inventory = load_inventory()["datasets"]
        ids = [i for i, entry in inventory.items()
               if entry.get("extraction", {}).get("status") in
               ("extracted", "cached")]
        if not ids:
            ids = [s.id for s in registry.select(defaults_only=True)]

    print("building: " + ", ".join(ids))
    payload = build(ids, limit=args.limit, dedupe=not args.no_dedupe,
                    check_images=not args.no_image_check,
                    max_text_len=args.max_text_len)

    print(f"\ntotal: {payload['total_samples']:,} samples")
    print(BUILD_REPORT)
    return 0 if payload["total_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
