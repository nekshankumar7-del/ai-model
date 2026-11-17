"""Step 4 - score a saved checkpoint on the held-out test split, or read images.

    python scripts/13_evaluate_htr.py --name htr-char-crnn
    python scripts/13_evaluate_htr.py --name htr-char-crnn --images some/*.png

The split is reloaded from data/metadata/htr_splits.json, so this scores the
same test set the training run held out - it does not re-split, which would
quietly move the goalposts.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from htr import splits as split_module
from htr.build import available_manifests, load_samples
from htr.evaluate import evaluate, read_crops
from htr.models import TORCH_AVAILABLE
from htr.paths import REPORTS_DIR, SPLITS_FILE
from htr.train import checkpoint_path, load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="htr-char-crnn")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--level", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--images", nargs="*", default=None,
                    help="read these image files instead of scoring a split")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("STOPPING. PyTorch is not installed.")
        return 2

    path = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.name)
    if not path.exists():
        print(f"no checkpoint at {path}")
        return 1
    model, charset, config, metrics = load_checkpoint(path, args.device)
    print(f"loaded {path}")
    print(f"  {config.architecture}, {charset.n_classes} classes, "
          f"trained at level {config.level}")

    if args.images:
        import cv2
        import numpy as np
        paths = [Path(p) for pattern in args.images for p in glob.glob(pattern)]
        crops = []
        kept = []
        for image_path in paths:
            data = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            if data is None:
                print(f"  unreadable: {image_path}")
                continue
            crops.append(data)
            kept.append(image_path)
        for image_path, prediction in zip(
                kept, read_crops(model, crops, charset,
                                 preprocess=config.preprocess(),
                                 device=args.device)):
            print(f"  {image_path.name:<40} {prediction.text!r} "
                  f"(confidence {prediction.confidence:.2f})")
        return 0

    dataset_ids = args.datasets or available_manifests()
    samples = load_samples(dataset_ids, level=args.level or config.level)
    if not SPLITS_FILE.exists():
        print(f"no split file at {SPLITS_FILE}; run scripts/12_train_htr.py")
        return 1
    parts = split_module.load(samples, SPLITS_FILE)
    test = [s for s in parts.get(split_module.TEST, []) if charset.filter(s.text)]
    print(f"  test split: {len(test):,} samples")

    result = evaluate(model, test, charset, preprocess=config.preprocess(),
                      batch_size=args.batch_size, device=args.device)
    print(f"\n  CER          {result['cer']:.4f}")
    print(f"  WER          {result['wer']:.4f}")
    print(f"  exact match  {result['exact_match']:.4f}   (n={result['n']:,})")
    for dataset, row in result.get("per_dataset", {}).items():
        print(f"    {dataset:<16} cer={row['cer']:.4f} "
              f"exact={row['exact_match']:.4f} n={row['n']:,}")
    for script, row in result.get("per_script", {}).items():
        print(f"    script {script:<10} cer={row['cer']:.4f} n={row['n']:,}")

    out = REPORTS_DIR / f"{args.name}-eval.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\n{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
