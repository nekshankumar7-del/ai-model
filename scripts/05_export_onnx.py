"""Step 6 - export everything the Angular app needs.

    python scripts/05_export_onnx.py

Writes to angular/assets/factfind/:
  det.onnx, rec.onnx, cls.onnx   OCR models (PP-OCRv4, run under onnxruntime-web)
  linker.onnx, confidence.onnx   the two trained models, dynamically quantised
  factfind-spec.json             preprocessing, feature order, scaling,
                                 calibration, thresholds and every field rule

Also writes a model card per model into models/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from factfind.export_onnx import export_all
from factfind.knowledge import load_knowledge
from factfind.paths import ANGULAR_DIR, MODELS_DIR, REPORTS_DIR
from factfind.report import write_model_card

BUDGET_MB = 30.0


def load_metrics() -> dict:
    out = {}
    for name in ("train", "test"):
        path = REPORTS_DIR / f"{name}.json"
        if path.exists():
            try:
                out[name] = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ANGULAR_DIR / "assets" / "factfind"))
    ap.add_argument("--no-ocr", action="store_true",
                    help="skip copying the OCR models (smaller bundle, but the "
                         "browser must get them elsewhere)")
    args = ap.parse_args()

    kn = load_knowledge()
    metrics = load_metrics()

    summary = {}
    if "test" in metrics:
        summary = {
            "split": "test",
            "wrong_fill_rate": metrics["test"]["final"]["summary"]["wrong_fill_rate"],
            "exact_match_rate": metrics["test"]["final"]["summary"]["exact_match_rate"],
            "coverage": metrics["test"]["final"]["summary"]["coverage"],
            "n_pages": metrics["test"]["n_pages"],
        }
    elif "train" in metrics:
        summary = {"split": "validation", **metrics["train"].get("final", {}).get("summary", {})}

    result = export_all(kn, out_dir=args.out, include_ocr=not args.no_ocr,
                        metrics=summary)

    print(f"exported to {result.out_dir}\n")
    for f in sorted(result.out_dir.glob("*")):
        if f.is_file():
            print(f"  {f.name:<24} {f.stat().st_size / 1024:>9.1f} KB")
    print(f"\n  {'total':<24} {result.total_mb:>9.2f} MB  "
          f"(budget {BUDGET_MB:.0f} MB)")

    if result.total_mb > BUDGET_MB:
        print(f"\nOVER BUDGET by {result.total_mb - BUDGET_MB:.2f} MB.")
        print("Options: drop cls.onnx (orientation classifier, optional), or "
              "use the PP-OCRv4 mobile recogniser in place of the server one.")
    else:
        print(f"  within budget, {BUDGET_MB - result.total_mb:.2f} MB to spare")

    # ---- model cards
    linker_metrics = {}
    conf_metrics = {}
    train = metrics.get("train", {})
    if train.get("linker", {}).get("kept"):
        cv = train["linker"].get("metrics", {}).get("cv", {})
        best = train["linker"].get("model")
        linker_metrics = {f"cv_{k}": v for k, v in (cv.get(best, {}) or {}).items()}
    if train.get("confidence", {}).get("kept"):
        conf_metrics = {k: v for k, v in train["confidence"]["metrics"].items()
                        if isinstance(v, (int, float))}

    write_model_card(MODELS_DIR / "model_card_linker.md", "field linker", {
        "version": "0.1.0",
        "purpose": (
            "Scores one (text line, field) hypothesis: given a line read off a "
            "page and a candidate field from the knowledge file, how likely is "
            "it that this line is that field? It re-ranks the candidates the "
            "rule matcher produces, it does not replace them.\n\n"
            "It is a re-ranker rather than a 107-way classifier on purpose: the "
            "knowledge file is designed so you can add a field without "
            "retraining, and a flat softmax over field keys would break that "
            "every time you edited the file."
        ),
        "data": (
            "Candidate hypotheses generated from the labelled pages in the "
            "train and validation splits. A hypothesis is positive when the "
            "value parsed under that field matches the label file. Splits are "
            "grouped by (writer, client) so no hand or person crosses a split."
        ),
        "io": (
            "Input `features`: float32 [batch, N] in the exact order given by "
            "`models.linker.featureNames` in factfind-spec.json, with the "
            "scaling in `features.scaling` already applied.\n\n"
            "Output `probabilities`: float32 [batch, 2]; column 1 is "
            "P(this is the right field)."
        ),
        "metrics": linker_metrics,
        "failures": (
            "- Fields whose labels collide (several money fields share the "
            "alias 'savings' or 'other') are genuinely ambiguous from the "
            "label alone; the model leans on the section heading, so an "
            "unheaded note is harder.\n"
            "- Narrative notes with several facts per sentence are the weakest "
            "layout.\n"
            "- It cannot recover a field whose label the knowledge file does "
            "not list: the candidate never gets generated."
        ),
        "limits": (
            "Not a document classifier and not an OCR model. It assumes the "
            "text it is given was read from the page; it has no way to tell "
            "that the recogniser misread a word."
        ),
    })

    write_model_card(MODELS_DIR / "model_card_confidence.md", "confidence", {
        "version": "0.1.0",
        "purpose": (
            "Predicts whether an extracted value is actually correct, from "
            "recogniser confidence, the margin between the top two candidates, "
            "fuzzy label score, parser validity, cross-field checks and image "
            "quality. Its calibrated output is compared with a per-kind "
            "threshold to decide fill versus unsure.\n\n"
            "This is the model that controls the wrong-fill rate, which the "
            "brief makes priority 1."
        ),
        "data": (
            "One example per assigned value on the train split, labelled by "
            "whether it matched the label file. A value filled for a field the "
            "page never mentioned counts as incorrect. Calibration and all "
            "thresholds are fitted on the validation split, never on test."
        ),
        "io": (
            "Input `features`: float32 [batch, N] in the order given by "
            "`models.confidence.featureNames`.\n\n"
            "Output `probabilities`: float32 [batch, 2]; column 1 is the raw "
            "P(correct). Apply the isotonic knots in "
            "`models.confidence.calibration` by linear interpolation to get "
            "the calibrated probability, then compare with "
            "`decision.thresholds[kind]`."
        ),
        "metrics": conf_metrics,
        "failures": (
            "- A confidently misread value that still parses cleanly is the "
            "hard case: a date read as 03/04 instead of 08/04 is valid, "
            "plausible and wrong. Cross-field checks catch some of these, not "
            "all.\n"
            "- Calibration is only as good as the validation split; with few "
            "writers it will be optimistic on a new hand.\n"
            "- Fields with very few validation examples get a threshold fitted "
            "on thin evidence."
        ),
        "limits": (
            "The probability is calibrated for the data it was fitted on. "
            "Refit the calibration and thresholds whenever the document mix, "
            "the writers or the OCR engine change."
        ),
    })
    print(f"\nwrote {MODELS_DIR / 'model_card_linker.md'}")
    print(f"wrote {MODELS_DIR / 'model_card_confidence.md'}")
    print(f"wrote {result.spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
