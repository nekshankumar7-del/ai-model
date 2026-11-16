"""Step 5/7 - run the locked test split ONCE and write the final report.

Only run this when the validation numbers have stopped improving. The test
split must not be used to choose anything.

    python scripts/04_evaluate_test.py --data data/synthetic

Writes reports/test.json, reports/final.md and the coverage-vs-wrong-fill
figure. If a target is missed it is reported with the data, and the report
says what would close the gap. Targets are never lowered.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from factfind.dataset import discover_samples, evaluate_samples
from factfind.evaluate import (coverage_curve, error_breakdown, per_field_table,
                               per_kind_table, summarise, targets_report)
from factfind.extract import Extractor, load_extractor
from factfind.knowledge import load_knowledge
from factfind.paths import DATASET_DIR, FIGURES_DIR, REPORTS_DIR
from factfind.splits import load_splits
from factfind.report import write_final_report, plot_coverage_curve


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(DATASET_DIR))
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--allow-rerun", action="store_true",
                    help="acknowledge that the test split has already been used")
    args = ap.parse_args()

    kn = load_knowledge()
    samples = discover_samples(args.data)
    if not samples:
        print(f"STOPPING. No labelled pages found in {args.data}.")
        return 2

    splits = load_splits()["splits"]
    by_stem = {s.stem: s for s in samples}
    chosen = [by_stem[s] for s in splits[args.split] if s in by_stem]
    if not chosen:
        print(f"STOPPING. The {args.split} split is empty.")
        return 2

    marker = REPORTS_DIR / f".{args.split}-used"
    if args.split == "test" and marker.exists() and not args.allow_rerun:
        print(f"The test split has already been scored (see {marker}).")
        print("Scoring it again after changing anything turns it into a "
              "validation set. Re-run with --allow-rerun if you are sure.")
        return 3

    print(f"scoring the {args.split} split: {len(chosen)} pages")

    baseline = Extractor(kn)
    print("\nbaseline (rules only):")
    baseline_outcomes = evaluate_samples(chosen, kn, baseline)
    baseline_stats = summarise(baseline_outcomes)

    final = load_extractor(use_models=True)
    has_models = final.linker is not None or final.confidence_model is not None
    if has_models:
        print("\nfinal (with trained models):")
        final_outcomes = evaluate_samples(chosen, kn, final)
    else:
        print("\nno trained models found - the final system IS the baseline")
        final_outcomes = baseline_outcomes
    final_stats = summarise(final_outcomes)

    print("\n" + "=" * 72)
    for title, stats in (("baseline", baseline_stats), ("final", final_stats)):
        print(f"{title:<9} filled={stats['n_filled']:<5} "
              f"exact={stats['exact_match_rate']:.4f} "
              f"wrong-fill={stats['wrong_fill_rate']:.4f} "
              f"coverage={stats['coverage']:.4f} "
              f"unsure={stats['unsure_rate']:.4f}")

    tr = targets_report(final_outcomes, kn)
    print("\nwrong-fill targets:")
    for row in tr["per_kind"]:
        mark = "ok  " if row["target_met"] else "MISS"
        print(f"  {mark} {row['kind']:<10} {row['wrong_fill']:.4f} "
              f"(target <= {row['target_wrong_fill']:.3f})  "
              f"n_filled={row['n_filled']}  n_wrong={row['n_wrong']}")

    payload = {
        "split": args.split,
        "n_pages": len(chosen),
        "baseline": {
            "summary": baseline_stats,
            "per_kind": per_kind_table(baseline_outcomes),
            "per_field": per_field_table(baseline_outcomes, kn),
        },
        "final": {
            "summary": final_stats,
            "per_kind": per_kind_table(final_outcomes),
            "per_field": per_field_table(final_outcomes, kn),
            "targets": tr,
            "errors": error_breakdown(final_outcomes, top=25),
            "curve": coverage_curve(final_outcomes),
        },
        "has_models": has_models,
    }

    # Carry the validation numbers through so the report can show the
    # generalisation gap rather than only the test figures.
    train_report = REPORTS_DIR / "train.json"
    if train_report.exists():
        try:
            train_payload = json.loads(train_report.read_text(encoding="utf-8"))
            payload["validation_per_kind"] = train_payload["final"]["per_kind"]
        except (json.JSONDecodeError, KeyError):
            pass
    out = REPORTS_DIR / f"{args.split}.json"
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")

    figure = plot_coverage_curve(payload["final"]["curve"],
                                 FIGURES_DIR / f"coverage-{args.split}.png")
    if figure:
        print(f"wrote {figure}")

    if args.split == "test":
        report_path = write_final_report(payload, kn, REPORTS_DIR / "final.md",
                                         figure=figure, data_dir=args.data)
        print(f"wrote {report_path}")
        marker.write_text(json.dumps({"scored": True, "n_pages": len(chosen)}),
                          encoding="utf-8")

    if not tr["all_met"]:
        print("\nTARGETS MISSED. They have not been lowered. "
              "reports/final.md lists what would close each gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
