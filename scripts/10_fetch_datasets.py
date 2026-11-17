"""Step 1 of the public-dataset HTR pipeline - discover, download, extract.

    # what is registered, on what terms, and how big
    python scripts/10_fetch_datasets.py --list

    # the default English set (EMNIST + NIST SD19, ~1.1 GB)
    python scripts/10_fetch_datasets.py

    # named datasets, or everything that may legitimately be downloaded
    python scripts/10_fetch_datasets.py --datasets uci_dhcd iiit_hw_dev
    python scripts/10_fetch_datasets.py --datasets all --max-gb 40

Downloads resume, are size-checked, and are hashed into
``data/metadata/inventory.json``. Datasets that require registration are
reported with instructions and skipped - never fetched from a mirror.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from htr import registry
from htr.download import fetch, load_inventory, save_inventory
from htr.extraction import extract_all
from htr.paths import INVENTORY_FILE


def print_table() -> None:
    header = f"{'id':<18}{'access':<13}{'level':<6}{'size':>9}  {'scripts':<22}licence"
    print(header)
    print("-" * len(header))
    for spec in registry.ALL:
        size = f"{spec.total_bytes / 1e9:.2f} GB" if spec.total_bytes else "-"
        star = "*" if spec.default else " "
        print(f"{star}{spec.id:<17}{spec.access:<13}{spec.level:<6}{size:>9}  "
              f"{','.join(spec.scripts):<22}{spec.licence[:48]}")
    print("\n* = downloaded by default")
    manual = [s for s in registry.ALL if s.access != "auto"]
    if manual:
        print("\nNot downloadable automatically:")
        for spec in manual:
            print(f"\n  {spec.id} - {spec.name}")
            print(f"    licence: {spec.licence}")
            print(f"    {spec.notes}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset ids, or 'all'. Default: the starred set.")
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--max-gb", type=float, default=None,
                    help="stop before exceeding this total download size")
    ap.add_argument("--include-optional", action="store_true",
                    help="also fetch resources marked optional (e.g. SD19 by_class)")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--no-checksum", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    args = ap.parse_args()

    if args.list:
        print_table()
        return 0

    specs = registry.select(args.datasets, defaults_only=args.datasets is None)
    if not specs:
        print("nothing selected")
        return 1

    print("selected: " + ", ".join(s.id for s in specs))
    total = sum(s.total_bytes for s in specs if s.access == "auto")
    print(f"download budget: {total / 1e9:.2f} GB\n")

    inventory = fetch(specs, include_optional=args.include_optional,
                      checksum=not args.no_checksum,
                      max_bytes=int(args.max_gb * 1e9) if args.max_gb else None)

    if not args.no_extract:
        downloadable = [s.id for s in specs
                        if s.extract
                        and inventory["datasets"].get(s.id, {}).get("status")
                        in ("downloaded", "partial")]
        for spec in specs:
            if not spec.extract:
                print(f"\n[{spec.id}] not extracted - the adapter reads "
                      f"the archive directly")
        # Unpack only what was actually fetched this run. An optional archive
        # left over from an earlier run should not be unpacked by accident.
        only = {
            dataset_id: [Path(f["file"]).name
                         for f in inventory["datasets"][dataset_id].get("files", [])
                         if f.get("file")]
            for dataset_id in downloadable
        }
        results = extract_all(downloadable, force=args.force_extract, only=only)
        for result in results:
            entry = inventory["datasets"].setdefault(result["dataset"], {})
            entry["extraction"] = result
        save_inventory(inventory)

    print("\nsummary")
    for dataset_id, entry in inventory["datasets"].items():
        print(f"  {dataset_id:<18} {entry.get('status', '?')}")
    print(f"\n{INVENTORY_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
