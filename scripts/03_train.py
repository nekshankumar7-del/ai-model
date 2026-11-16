"""Steps 3 and 4 - baseline, then the models, measured on validation.

  1. Baseline: preprocessing + OCR + rule/fuzzy matcher with per-kind parsers
     and lexicon snapping. This is the number to beat.
  2. Linker: re-ranks the rule matcher's field candidates. Model family chosen
     by grouped cross-validation over train+validation.
  3. Confidence: calibrated P(correct), with per-kind fill thresholds chosen on
     validation to meet the wrong-fill targets.

Each model is kept only if it beats the baseline on validation. If it does
not, it is dropped and the report says so.

    python scripts/03_train.py --data data/synthetic
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from factfind.confidence import ConfidenceModel
from factfind.dataset import (collect_many, discover_samples, evaluate_samples,
                              flatten_confidence, flatten_linker)
from factfind.evaluate import (per_kind_table, summarise, targets_report,
                               error_breakdown)
from factfind.extract import Extractor
from factfind.knowledge import KnowledgeError, load_knowledge
from factfind.linker import FieldLinker
from factfind.paths import DATASET_DIR, MODELS_DIR, REPORTS_DIR
from factfind.splits import load_splits


def banner(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def show(stats: dict, title: str) -> None:
    print(f"  {title}: filled={stats['n_filled']} "
          f"exact={stats['exact_match_rate']:.3f} "
          f"wrong-fill={stats['wrong_fill_rate']:.4f} "
          f"coverage={stats['coverage']:.3f} "
          f"unsure={stats['unsure_rate']:.3f} "
          f"missing={stats['missing_rate']:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_DIR))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--blend", type=float, default=0.65,
                    help="weight of the linker probability against the rule score")
    args = ap.parse_args()

    started = time.time()
    try:
        kn = load_knowledge()
    except KnowledgeError as exc:
        print("STOPPING. The knowledge file is not usable:\n")
        print(exc)
        return 2

    samples = discover_samples(args.data)
    if not samples:
        print(f"STOPPING. No labelled pages found in {args.data}.")
        return 2

    splits = load_splits()["splits"]
    by_stem = {s.stem: s for s in samples}
    train = [by_stem[s] for s in splits["train"] if s in by_stem]
    val = [by_stem[s] for s in splits["val"] if s in by_stem]
    print(f"train {len(train)} pages | validation {len(val)} pages "
          f"| test {len(splits['test'])} pages (locked)")

    report: dict = {"data": args.data, "n_train": len(train), "n_val": len(val)}

    # ---------------------------------------------------------------- 1
    banner("Step 3 - baseline (rules only, no trained model)")
    baseline = Extractor(kn)
    baseline_outcomes = evaluate_samples(val, kn, baseline)
    baseline_stats = summarise(baseline_outcomes)
    show(baseline_stats, "baseline")
    report["baseline"] = {
        "summary": baseline_stats,
        "per_kind": per_kind_table(baseline_outcomes),
        "targets": targets_report(baseline_outcomes, kn),
    }

    # ---------------------------------------------------------------- 2
    banner("Step 4.1 - field linker")
    print("collecting training examples (OCR is cached after the first pass)")
    train_pages = collect_many(train, kn, baseline)
    val_pages = collect_many(val, kn, baseline)

    linker_examples = flatten_linker(train_pages) + flatten_linker(val_pages)
    pos = sum(e.label for e in linker_examples)
    print(f"  {len(linker_examples)} (line, field) hypotheses, "
          f"{pos} positive ({pos / max(1, len(linker_examples)):.1%})")

    linker = None
    try:
        linker = FieldLinker.select_and_fit(linker_examples, folds=args.folds,
                                            seed=args.seed)
        linker.blend = args.blend
    except ValueError as exc:
        print(f"  linker could not be trained: {exc}")

    linker_stats = None
    if linker is not None:
        with_linker = Extractor(kn, linker=linker)
        linker_outcomes = evaluate_samples(val, kn, with_linker)
        linker_stats = summarise(linker_outcomes)
        show(baseline_stats, "baseline ")
        show(linker_stats, "+ linker ")
        better = (linker_stats["wrong_fill_rate"] <= baseline_stats["wrong_fill_rate"]
                  and linker_stats["exact_match_rate"] >= baseline_stats["exact_match_rate"] - 1e-9)
        if not better:
            print("  linker does NOT beat the baseline on validation - dropping it")
            report["linker"] = {"kept": False, "reason": "did not beat baseline",
                                "metrics": linker.metrics, "summary": linker_stats}
            linker = None
        else:
            print("  linker kept")
            report["linker"] = {"kept": True, "model": linker.model_name,
                                "blend": linker.blend, "metrics": linker.metrics,
                                "summary": linker_stats}
    else:
        report["linker"] = {"kept": False, "reason": "training failed"}

    # ---------------------------------------------------------------- 3
    banner("Step 4.3 - confidence model and fill thresholds")
    feature_source = Extractor(kn, linker=linker)
    print("re-collecting examples with the chosen linker in place")
    train_pages = collect_many(train, kn, feature_source)
    val_pages = collect_many(val, kn, feature_source)

    conf_train = flatten_confidence(train_pages)
    conf_val = flatten_confidence(val_pages)
    print(f"  {len(conf_train)} train / {len(conf_val)} validation assigned values")
    if conf_train:
        rate = np.mean([e.correct for e in conf_train])
        print(f"  raw correctness before thresholding: {rate:.3f}")

    confidence = None
    try:
        confidence = ConfidenceModel.fit(conf_train, conf_val, seed=args.seed)
    except ValueError as exc:
        print(f"  confidence model could not be trained: {exc}")
        report["confidence"] = {"kept": False, "reason": str(exc)}

    final_stats = linker_stats or baseline_stats
    final_outcomes = None
    if confidence is not None:
        final = Extractor(kn, linker=linker, confidence_model=confidence,
                          thresholds=confidence.thresholds)
        final_outcomes = evaluate_samples(val, kn, final)
        final_stats = summarise(final_outcomes)
        print()
        show(baseline_stats, "baseline   ")
        if linker_stats:
            show(linker_stats, "+ linker   ")
        show(final_stats, "+ confidence")

        improved = final_stats["wrong_fill_rate"] <= baseline_stats["wrong_fill_rate"]
        if not improved:
            print("  confidence model does NOT reduce wrong-fill - dropping it")
            report["confidence"] = {"kept": False,
                                    "reason": "did not reduce wrong-fill",
                                    "metrics": confidence.metrics}
            confidence = None
            final_stats = linker_stats or baseline_stats
            final_outcomes = None
        else:
            report["confidence"] = {
                "kept": True, "metrics": confidence.metrics,
                "thresholds": confidence.thresholds,
                "summary": final_stats,
            }

    # ---------------------------------------------------------------- 4
    banner("Validation result")
    show(baseline_stats, "baseline")
    show(final_stats, "final   ")

    outcomes = final_outcomes if final_outcomes is not None else baseline_outcomes
    tr = targets_report(outcomes, kn)
    report["final"] = {
        "summary": final_stats,
        "per_kind": per_kind_table(outcomes),
        "targets": tr,
        "errors": error_breakdown(outcomes),
    }

    print("\nwrong-fill targets on validation:")
    for row in tr["per_kind"]:
        mark = "ok  " if row["target_met"] else "MISS"
        print(f"  {mark} {row['kind']:<10} wrong-fill={row['wrong_fill']:.4f} "
              f"(target <= {row['target_wrong_fill']:.3f})  "
              f"filled={row['n_filled']:<5} coverage={row['coverage']:.3f}")
    if not tr["all_met"]:
        print("\n  targets missed on validation for: "
              + ", ".join(m["kind"] for m in tr["misses"]))
        print("  the targets are not being lowered; see reports/final.md for what "
              "would close the gap")

    # ---------------------------------------------------------------- 5
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if linker is not None:
        linker.save(MODELS_DIR / "linker.joblib")
        print(f"\nsaved {MODELS_DIR / 'linker.joblib'}")
    if confidence is not None:
        confidence.save(MODELS_DIR / "confidence.joblib")
        (MODELS_DIR / "thresholds.json").write_text(
            json.dumps(confidence.thresholds, indent=2), encoding="utf-8")
        print(f"saved {MODELS_DIR / 'confidence.joblib'}")
        print(f"saved {MODELS_DIR / 'thresholds.json'}")

    report["elapsed_seconds"] = round(time.time() - started, 1)
    out = REPORTS_DIR / "train.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}  ({report['elapsed_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
