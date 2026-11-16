"""Metrics.

The four outcomes for a field that is present on a page:

    exact    filled, and the value equals the label
    wrong    filled, and the value does NOT equal the label  <-- priority 1
    unsure   read but not filled; the adviser is asked
    missing  not found at all

Plus one that is not on the page at all:

    spurious filled a field the page never mentioned          <-- also a
                                                                  wrong-fill

wrong_fill_rate is computed over everything the pipeline chose to fill,
because that is what the user experiences: of the values it put in my form,
how many were wrong. Spurious fills count against it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .extract import ExtractionResult, FieldPrediction, STATUS_FILLED, STATUS_UNSURE
from .knowledge import Knowledge, Field
from .labels import values_match, qualified_key

# Targets from the brief, by field kind.
WRONG_FILL_TARGETS: dict[str, float] = {
    "choice": 0.01, "yesNo": 0.01, "date": 0.01, "number": 0.01, "money": 0.01,
    "niNumber": 0.0, "email": 0.0, "phone": 0.0,
    "name": 0.05, "fullName": 0.05, "text": 0.05,
}


def target_for(fld: Field) -> float:
    return WRONG_FILL_TARGETS.get(fld.kind, 0.01)


@dataclass
class Outcome:
    """One field on one page, scored."""
    stem: str
    flat_key: str
    qualified_key: str
    kind: str
    status: str                 # filled | unsure | missing
    correct: bool
    present_in_truth: bool
    predicted: Any = None
    truth: Any = None
    confidence: float = 0.0
    in_top3: bool = False

    @property
    def is_wrong_fill(self) -> bool:
        return self.status == STATUS_FILLED and not self.correct

    @property
    def is_exact(self) -> bool:
        return self.status == STATUS_FILLED and self.correct

    @property
    def is_spurious(self) -> bool:
        return self.status == STATUS_FILLED and not self.present_in_truth


def score_document(result: ExtractionResult, truth: dict[str, Any],
                   kn: Knowledge, stem: str = "") -> list[Outcome]:
    """Compare one document's predictions with its label file."""
    outcomes: list[Outcome] = []
    by_key = {p.flat_key: p for p in result.predictions}
    all_keys = set(by_key) | set(truth)

    for flat_key in sorted(all_keys):
        pred = by_key.get(flat_key)
        has_truth = flat_key in truth
        truth_value = truth.get(flat_key)

        try:
            qk = qualified_key(flat_key)
            fld = kn.field(qk)
        except (KeyError, ValueError):
            # A label key the knowledge file does not define: the audit
            # reports it; it cannot be scored.
            continue

        if pred is None:
            outcomes.append(Outcome(
                stem=stem, flat_key=flat_key, qualified_key=qk, kind=fld.kind,
                status="missing", correct=False, present_in_truth=has_truth,
                truth=truth_value,
            ))
            continue

        correct = has_truth and values_match(pred.value, truth_value, fld)
        in_top3 = has_truth and any(
            values_match(c, truth_value, fld) for c in pred.candidates[:3]
        )
        outcomes.append(Outcome(
            stem=stem, flat_key=flat_key, qualified_key=qk, kind=fld.kind,
            status=pred.status, correct=bool(correct), present_in_truth=has_truth,
            predicted=pred.value, truth=truth_value, confidence=pred.confidence,
            in_top3=bool(in_top3),
        ))
    return outcomes


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _rate(numerator: int, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def summarise(outcomes: Sequence[Outcome]) -> dict[str, Any]:
    """Headline metrics over a set of outcomes."""
    n_filled = sum(1 for o in outcomes if o.status == STATUS_FILLED)
    n_exact = sum(1 for o in outcomes if o.is_exact)
    n_wrong = sum(1 for o in outcomes if o.is_wrong_fill)
    n_unsure = sum(1 for o in outcomes if o.status == STATUS_UNSURE)
    n_missing = sum(1 for o in outcomes if o.status == "missing")
    n_present = sum(1 for o in outcomes if o.present_in_truth)
    n_spurious = sum(1 for o in outcomes if o.is_spurious)
    n_unsure_top3 = sum(1 for o in outcomes if o.status == STATUS_UNSURE and o.in_top3)

    return {
        "n_outcomes": len(outcomes),
        "n_present_in_truth": n_present,
        "n_filled": n_filled,
        "n_exact": n_exact,
        "n_wrong": n_wrong,
        "n_unsure": n_unsure,
        "n_missing": n_missing,
        "n_spurious": n_spurious,
        # Of everything we filled, how much was wrong. The priority-1 number.
        "wrong_fill_rate": _rate(n_wrong, n_filled),
        # Of everything truly on the page, how much did we fill correctly.
        "exact_match_rate": _rate(n_exact, n_present),
        # Of everything truly on the page, how much did we fill at all.
        "coverage": _rate(sum(1 for o in outcomes
                              if o.status == STATUS_FILLED and o.present_in_truth),
                          n_present),
        "unsure_rate": _rate(n_unsure, n_present),
        "missing_rate": _rate(n_missing, n_present),
        "unsure_top3_recall": _rate(n_unsure_top3, n_unsure),
    }


def per_field_table(outcomes: Sequence[Outcome], kn: Knowledge) -> list[dict[str, Any]]:
    """One row per field, with its target and whether it is met."""
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        grouped[o.qualified_key].append(o)

    rows = []
    for qk, items in sorted(grouped.items()):
        try:
            fld = kn.field(qk)
        except KeyError:
            continue
        stats = summarise(items)
        target = target_for(fld)
        rows.append({
            "field": qk,
            "label": fld.label,
            "kind": fld.kind,
            "n": stats["n_present_in_truth"],
            "n_filled": stats["n_filled"],
            "exact_match": stats["exact_match_rate"],
            "wrong_fill": stats["wrong_fill_rate"],
            "coverage": stats["coverage"],
            "unsure": stats["unsure_rate"],
            "missing": stats["missing_rate"],
            "n_wrong": stats["n_wrong"],
            "target_wrong_fill": target,
            "target_met": stats["wrong_fill_rate"] <= target + 1e-9,
        })
    return rows


def per_kind_table(outcomes: Sequence[Outcome]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        grouped[o.kind].append(o)
    rows = []
    for kind, items in sorted(grouped.items()):
        stats = summarise(items)
        target = WRONG_FILL_TARGETS.get(kind, 0.01)
        rows.append({
            "kind": kind, "n": stats["n_present_in_truth"],
            "n_filled": stats["n_filled"], "n_wrong": stats["n_wrong"],
            "exact_match": stats["exact_match_rate"],
            "wrong_fill": stats["wrong_fill_rate"],
            "coverage": stats["coverage"], "unsure": stats["unsure_rate"],
            "target_wrong_fill": target,
            "target_met": stats["wrong_fill_rate"] <= target + 1e-9,
        })
    return rows


def coverage_curve(outcomes: Sequence[Outcome],
                   points: int = 60) -> list[dict[str, float]]:
    """Coverage against wrong-fill as the fill threshold sweeps 0..1.

    Uses every outcome that produced a value (filled or unsure), re-deciding
    at each threshold. This is the curve the final report plots.
    """
    scored = [o for o in outcomes if o.status in (STATUS_FILLED, STATUS_UNSURE)]
    n_present = sum(1 for o in outcomes if o.present_in_truth) or 1
    curve = []
    for i in range(points + 1):
        threshold = i / points
        filled = [o for o in scored if o.confidence >= threshold]
        n_wrong = sum(1 for o in filled if not o.correct)
        n_right = sum(1 for o in filled if o.correct)
        curve.append({
            "threshold": threshold,
            "coverage": n_right / n_present,
            "wrong_fill_rate": _rate(n_wrong, len(filled)),
            "n_filled": len(filled),
        })
    return curve


def error_breakdown(outcomes: Sequence[Outcome], top: int = 15) -> dict[str, Any]:
    """What is going wrong, ranked. Drives the error-analysis loop."""
    wrong = [o for o in outcomes if o.is_wrong_fill]
    by_field: dict[str, int] = defaultdict(int)
    by_kind: dict[str, int] = defaultdict(int)
    for o in wrong:
        by_field[o.qualified_key] += 1
        by_kind[o.kind] += 1

    missing = [o for o in outcomes if o.status == "missing" and o.present_in_truth]
    by_field_missing: dict[str, int] = defaultdict(int)
    for o in missing:
        by_field_missing[o.qualified_key] += 1

    return {
        "wrong_by_field": sorted(by_field.items(), key=lambda kv: -kv[1])[:top],
        "wrong_by_kind": sorted(by_kind.items(), key=lambda kv: -kv[1])[:top],
        "missing_by_field": sorted(by_field_missing.items(), key=lambda kv: -kv[1])[:top],
        "examples": [
            {"stem": o.stem, "field": o.flat_key, "predicted": o.predicted,
             "truth": o.truth, "confidence": round(o.confidence, 3)}
            for o in sorted(wrong, key=lambda o: -o.confidence)[:top]
        ],
    }


def targets_report(outcomes: Sequence[Outcome], kn: Knowledge) -> dict[str, Any]:
    """Did we hit the brief's targets? Per kind, with the miss spelled out."""
    rows = per_kind_table(outcomes)
    misses = [r for r in rows if not r["target_met"] and r["n_filled"] > 0]
    return {
        "per_kind": rows,
        "all_met": not misses,
        "misses": [
            {
                "kind": r["kind"],
                "wrong_fill": r["wrong_fill"],
                "target": r["target_wrong_fill"],
                "n_wrong": r["n_wrong"],
                "n_filled": r["n_filled"],
                "shortfall": r["wrong_fill"] - r["target_wrong_fill"],
            }
            for r in misses
        ],
    }


def to_json(outcomes: Sequence[Outcome], kn: Knowledge,
            path: Path | str | None = None) -> dict[str, Any]:
    payload = {
        "summary": summarise(outcomes),
        "per_kind": per_kind_table(outcomes),
        "per_field": per_field_table(outcomes, kn),
        "targets": targets_report(outcomes, kn),
        "errors": error_breakdown(outcomes),
        "curve": coverage_curve(outcomes),
    }
    if path:
        Path(path).write_text(json.dumps(payload, indent=2, default=str),
                              encoding="utf-8")
    return payload


__all__ = ["Outcome", "score_document", "summarise", "per_field_table",
           "per_kind_table", "coverage_curve", "error_breakdown",
           "targets_report", "to_json", "WRONG_FILL_TARGETS", "target_for"]
