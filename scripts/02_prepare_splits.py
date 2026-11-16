"""Step 2 - leakage-free splits: 70 train / 15 validation / 15 test.

Groups are connected components over (writer, client), so neither a hand nor a
person can appear on both sides of a split. The test split is written once and
is not to be tuned on.

    python scripts/02_prepare_splits.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from factfind.dataset import discover_samples
from factfind.paths import DATASET_DIR, SPLITS_FILE
from factfind.splits import (group_folds, make_splits, save_splits,
                             verify_no_leakage)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_DIR))
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    samples = discover_samples(args.data)
    if not samples:
        print(f"STOPPING. No labelled pages found in {args.data}.")
        print("Expected image files each with a JSON label file of the same name.")
        return 2

    splits = make_splits(samples, train=args.train, val=args.val,
                         test=args.test, seed=args.seed)
    leaks = verify_no_leakage(splits, samples)
    if leaks:
        print("STOPPING. The splits leak:")
        for leak in leaks[:20]:
            print("  -", leak)
        return 2

    path = save_splits(splits, samples, SPLITS_FILE)
    total = sum(len(v) for v in splits.values())

    print(f"{len(samples)} pages, "
          f"{len({s.writer for s in samples})} writers, "
          f"{len({s.client for s in samples})} clients")
    print()
    for name in ("train", "val", "test"):
        stems = set(splits[name])
        writers = {s.writer for s in samples if s.stem in stems}
        clients = {s.client for s in samples if s.stem in stems}
        print(f"  {name:<6} {len(stems):>4} pages ({len(stems)/total:5.1%})  "
              f"{len(writers):>3} writers  {len(clients):>4} clients")
    print("\nno writer or client appears in more than one split")

    folds = group_folds(samples, splits["train"] + splits["val"],
                        n_folds=args.folds, seed=args.seed)
    print(f"\n{len(folds)} grouped folds prepared over train+validation "
          f"for model selection")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
