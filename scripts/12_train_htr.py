"""Step 3 - split, build the charset, train, evaluate, checkpoint.

    # character model on everything character-level that is built
    python scripts/12_train_htr.py --level char

    # word model, with character corpora composed into word images
    python scripts/12_train_htr.py --level word --compose-words

    # a quick end-to-end smoke run
    python scripts/12_train_htr.py --level char --epochs 2 --max-samples 4000

Splits are writer-independent wherever the corpus carries writer ids, and the
split report says per dataset whether that was possible. The charset is built
from the training split only - deriving it from the test split would leak.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import _bootstrap  # noqa: F401

from htr import splits as split_module
from htr.build import available_manifests, load_samples
from htr.charset import build_charset
from htr.export import export_bundle
from htr.evaluate import evaluate
from htr.manifest import corpus_stats
from htr.models import TORCH_AVAILABLE
from htr.paths import METADATA_DIR, MODELS_DIR, REPORTS_DIR
from htr.train import TrainConfig, build_and_train, checkpoint_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="manifest ids to train on. Default: all built.")
    ap.add_argument("--level", choices=["char", "word", "line"], default=None,
                    help="only use samples at this level. Required when the "
                         "built manifests mix levels.")
    ap.add_argument("--arch", default="crnn",
                    choices=["crnn", "crnn-large", "transformer"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--augment", type=float, default=1.0,
                    help="augmentation strength; 0 disables it")
    ap.add_argument("--preprocess", default="light",
                    choices=["none", "light", "full"])
    ap.add_argument("--max-samples", type=int, default=None,
                    help="cap the training split (for quick runs)")
    ap.add_argument("--max-validation", type=int, default=4000)
    ap.add_argument("--min-char-count", type=int, default=5,
                    help="drop characters rarer than this from the charset")
    ap.add_argument("--compose-words", action="store_true",
                    help="also train on words stitched from character crops")
    ap.add_argument("--composed-per-epoch", type=int, default=20000)
    ap.add_argument("--name", default=None, help="checkpoint / export name")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("STOPPING. PyTorch is not installed.")
        print("  .venv/Scripts/python.exe -m pip install torch "
              "--index-url https://download.pytorch.org/whl/cpu")
        return 2

    dataset_ids = args.datasets or available_manifests()
    if not dataset_ids:
        print("no manifests built. Run scripts/11_build_manifests.py first.")
        return 1

    name = args.name or f"htr-{args.level or 'mixed'}-{args.arch}"
    print(f"datasets: {', '.join(dataset_ids)}")

    try:
        samples = load_samples(dataset_ids, level=args.level)
    except ValueError as exc:
        print(f"\n{exc}")
        return 1
    if not samples:
        print("no samples at that level")
        return 1
    print(f"samples: {len(samples):,}")

    # ---------------------------------------------------------- splits
    parts, report = split_module.assign(samples)
    leakage = split_module.check_leakage(parts)
    split_module.save(parts, report)
    print("\nsplit:")
    for side, rows in parts.items():
        print(f"  {side:<11} {len(rows):>8,}")
    for dataset, info in report.per_dataset.items():
        flag = "writer-independent" if info["writer_independent"] else \
            "NOT writer-independent (no writer ids)"
        print(f"  {dataset:<16} {info['writers']:>5} writers  {flag}")
    print(f"  leakage check: {'clean' if leakage['clean'] else leakage['overlaps']}")

    train_samples = parts[split_module.TRAIN]
    validation = parts[split_module.VALIDATION]
    test = parts[split_module.TEST]
    if args.max_samples:
        random.Random(args.seed).shuffle(train_samples)
        train_samples = train_samples[: args.max_samples]

    # --------------------------------------------------------- charset
    charset, charset_report = build_charset([s.text for s in train_samples],
                                            min_count=args.min_char_count)
    print(f"\ncharset: {charset.n_classes} classes "
          f"(blank + {len(charset)} characters)")
    for script, count in charset.scripts().items():
        print(f"  {script:<14} {count}")
    if charset_report["dropped_characters"]:
        print(f"  dropped {charset_report['dropped_characters']} characters "
              f"seen fewer than {args.min_char_count} times")

    train_samples = [s for s in train_samples if charset.filter(s.text)]
    validation = [s for s in validation if charset.filter(s.text)]
    test = [s for s in test if charset.filter(s.text)]

    config = TrainConfig(
        architecture=args.arch, level=args.level or "mixed", epochs=args.epochs,
        batch_size=args.batch_size, learning_rate=args.lr, hidden=args.hidden,
        workers=args.workers, seed=args.seed, device=args.device,
        patience=args.patience, augment_strength=args.augment,
        preprocess_level=args.preprocess, max_validation=args.max_validation,
    )

    # ------------------------------------------------- composed words
    composed = None
    if args.compose_words:
        from htr.augment import AugmentConfig
        from htr.data import ComposedWordDataset
        characters = [s for s in train_samples if len(s.text) == 1]
        vocabulary = sorted({s.text for s in train_samples if len(s.text) > 1})
        if not vocabulary:
            vocabulary = _fallback_vocabulary()
        if characters:
            composed = ComposedWordDataset(
                characters, charset, vocabulary,
                length=args.composed_per_epoch, preprocess=config.preprocess(),
                augment=AugmentConfig(strength=args.augment), seed=args.seed)
            print(f"\ncomposed words: {args.composed_per_epoch:,} per epoch "
                  f"from {len(characters):,} glyphs, "
                  f"{len(vocabulary):,} word forms")
        else:
            print("\n--compose-words asked for, but no single-character "
                  "samples are in the training split; skipping")

    # ---------------------------------------------------------- train
    print(f"\n=== training {name} ===")
    print(f"  {len(train_samples):,} train / {len(validation):,} validation "
          f"/ {len(test):,} test")
    checkpoint = checkpoint_path(name)
    model, result = build_and_train(train_samples, validation, charset, config,
                                    composed=composed, checkpoint=checkpoint)

    # ------------------------------------------------------- evaluate
    print("\n=== held-out test split ===")
    metrics = evaluate(model, test, charset, preprocess=config.preprocess(),
                       batch_size=max(16, args.batch_size), device=args.device)
    print(f"  CER          {metrics['cer']:.4f}")
    print(f"  WER          {metrics['wer']:.4f}")
    print(f"  exact match  {metrics['exact_match']:.4f}   (n={metrics['n']:,})")
    for dataset, row in metrics.get("per_dataset", {}).items():
        print(f"    {dataset:<16} cer={row['cer']:.4f} "
              f"exact={row['exact_match']:.4f} n={row['n']:,}")

    charset.save(MODELS_DIR / f"{name}-charset.json")

    payload = {
        "name": name,
        "config": config.__dict__,
        "datasets": dataset_ids,
        "split": report.as_dict(),
        "leakage": leakage,
        "charset": charset_report,
        "corpus": corpus_stats(train_samples),
        "training": result.as_dict(),
        "test": metrics,
    }

    if not args.no_export:
        print("\n=== export ===")
        try:
            bundle = export_bundle(model, charset, MODELS_DIR, name=name,
                                   metrics={"test": {k: metrics[k] for k in
                                                     ("cer", "wer",
                                                      "exact_match", "n")}})
            payload["export"] = bundle
            print(f"  {bundle['onnx']['path']} "
                  f"({bundle['onnx']['megabytes']} MB), parity "
                  f"max diff {bundle['parity']['max_abs_difference']:.2e}")
        except (AssertionError, RuntimeError) as exc:
            payload["export"] = {"error": str(exc)}
            print(f"  export FAILED: {exc}")

    out = REPORTS_DIR / f"{name}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                   encoding="utf-8")
    print(f"\ncheckpoint {checkpoint}")
    print(f"report     {out}")
    return 0


def _fallback_vocabulary() -> list[str]:
    """Words to compose when the corpus has no multi-character labels.

    Drawn from the fact-find knowledge file, so the composed words are the
    strings this project actually has to read.
    """
    try:
        from factfind.knowledge import load_knowledge
        kn = load_knowledge()
    except Exception:                                   # noqa: BLE001
        return ["Yes", "No", "Name", "Address", "Salary", "Date"]
    words: set[str] = set()
    for fld in kn.fields:
        words.add(fld.label)
        words.update(fld.aliases)
        for choice in fld.resolved_choices():
            words.add(choice.value)
            words.add(choice.label)
        words.update(fld.vocabulary)
    return sorted(w for w in words if w and len(w) <= 24)


if __name__ == "__main__":
    raise SystemExit(main())
