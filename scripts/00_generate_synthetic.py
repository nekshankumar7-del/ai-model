"""Generate synthetic fact-find pages from the knowledge file.

These exist so the pipeline can be built, tested and demonstrated before real
client notes are available, and so the models have something to pre-train on.
Per the brief, synthetic data is for pre-training and development only: final
fine-tuning and all model selection must happen on real pages.

    python scripts/00_generate_synthetic.py --pages 400 --writers 14
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from factfind.knowledge import load_knowledge
from factfind.paths import SYNTHETIC_DIR
from factfind.synth import generate_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pages", type=int, default=400)
    ap.add_argument("--writers", type=int, default=14)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(SYNTHETIC_DIR))
    ap.add_argument("--augment-fraction", type=float, default=0.65)
    args = ap.parse_args()

    kn = load_knowledge()
    print(f"knowledge: v{kn.version}, {len(kn.sections)} sections, {len(kn.fields)} fields")
    for problem in kn.problems:
        print("  ", problem)

    print(f"generating {args.pages} pages from {args.writers} writers -> {args.out}")
    manifest = generate_dataset(
        kn, args.out, n_pages=args.pages, n_writers=args.writers,
        seed=args.seed, augment_fraction=args.augment_fraction,
    )

    from pathlib import Path
    Path(args.out, "_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    layouts: dict[str, int] = {}
    for row in manifest:
        layouts[row["layout"]] = layouts.get(row["layout"], 0) + 1
    total_fields = sum(row["n_fields"] for row in manifest)
    print(f"wrote {len(manifest)} pages, {total_fields} labelled field values")
    print(f"layouts: {layouts}")
    print(f"writers: {len({row['writer'] for row in manifest})}  "
          f"clients: {len({row['client'] for row in manifest})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
