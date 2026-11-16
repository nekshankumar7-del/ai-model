"""Dataset audit -- run before training anything.

Answers the questions in Step 1 of the brief: how much data is there, who
wrote it, what layouts, how often does each field and each value appear, which
labels look wrong, what is duplicated, what is unreadable, and what is the
minimum extra data needed per field to reach the targets.

It also screens for personal data so it can be flagged for anonymisation, and
it never writes a client value into the report.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from rapidfuzz import fuzz

from .dataset import discover_samples, unlabelled_images
from .knowledge import Knowledge
from .labels import load_label_file, qualified_key, validate_against_knowledge
from .ocr import read_lines
from .parsers import parse_value
from .preprocess import load_pages, prepare_page
from .splits import Sample

# How many labelled examples of a field we want before trusting its numbers.
# Below this the measured wrong-fill rate has no resolution: you cannot
# demonstrate a 1% error rate with 20 samples.
MIN_EXAMPLES_PER_FIELD = 120
# To show a rate of r with reasonable confidence you need roughly 3/r samples.
def min_examples_for_target(target: float) -> int:
    if target <= 0:
        return 400          # "zero errors" needs a few hundred clean examples
    return int(max(MIN_EXAMPLES_PER_FIELD, round(3.0 / target)))


# --------------------------------------------------------------------------
# Personal data screening
# --------------------------------------------------------------------------

PII_PATTERNS = {
    "ni_number": re.compile(r"\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "uk_phone": re.compile(r"\b0\d{4}\s?\d{6}\b|\b0\d{3}\s?\d{3}\s?\d{4}\b"),
    "passport": re.compile(r"\b\d{9}\b"),
    "postcode": re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b", re.I),
}


def scan_pii(samples: Sequence[Sample]) -> dict[str, Any]:
    """Count personal data by type across the label files.

    Counts only -- no values are copied into the report.
    """
    counts: Counter[str] = Counter()
    files_with_pii: set[str] = set()
    for sample in samples:
        try:
            flat, _meta = load_label_file(sample.label_path)
        except Exception:
            continue
        blob = " ".join(str(v) for v in flat.values())
        for name, pattern in PII_PATTERNS.items():
            hits = pattern.findall(blob)
            if hits:
                counts[name] += len(hits)
                files_with_pii.add(sample.stem)
    return {
        "by_type": dict(counts),
        "n_files_with_personal_data": len(files_with_pii),
        "total_files": len(samples),
    }


# --------------------------------------------------------------------------
# Duplicates
# --------------------------------------------------------------------------

def _phash(image: np.ndarray, size: int = 16) -> str:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(small)[:8, :8]
    med = np.median(dct[1:, 1:])
    bits = (dct > med).flatten()
    return "".join("1" if b else "0" for b in bits)


def _hamming(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def find_duplicates(samples: Sequence[Sample], threshold: int = 6) -> dict[str, Any]:
    """Exact and near-duplicate images -- a leakage risk across splits."""
    exact: dict[str, list[str]] = defaultdict(list)
    hashes: dict[str, str] = {}
    unreadable: list[str] = []

    for sample in samples:
        try:
            pages = load_pages(sample.image_path)
            if not pages:
                unreadable.append(sample.stem)
                continue
            image = pages[0]
            digest = hashlib.md5(Path(sample.image_path).read_bytes()).hexdigest()
            exact[digest].append(sample.stem)
            hashes[sample.stem] = _phash(image)
        except Exception:
            unreadable.append(sample.stem)

    exact_groups = [stems for stems in exact.values() if len(stems) > 1]

    near: list[tuple[str, str, int]] = []
    stems = sorted(hashes)
    for i, a in enumerate(stems):
        for b in stems[i + 1:]:
            d = _hamming(hashes[a], hashes[b])
            if d <= threshold:
                near.append((a, b, d))
    return {
        "exact_duplicate_groups": exact_groups,
        "n_exact_duplicates": sum(len(g) - 1 for g in exact_groups),
        "near_duplicate_pairs": near[:200],
        "n_near_duplicate_pairs": len(near),
        "unreadable": unreadable,
    }


# --------------------------------------------------------------------------
# Label / OCR disagreement
# --------------------------------------------------------------------------

def find_label_disagreements(samples: Sequence[Sample], kn: Knowledge,
                             limit: int | None = None,
                             min_ocr_confidence: float = 0.80,
                             similarity_floor: int = 55) -> list[dict[str, Any]]:
    """Labels that confidently-read OCR text contradicts.

    A label is flagged when the page's OCR contains nothing resembling the
    labelled value, and the OCR for that page was confident. These are the
    candidates for relabelling -- the report names the field and the file, not
    the value.
    """
    flagged: list[dict[str, Any]] = []
    for sample in (samples[:limit] if limit else samples):
        try:
            flat, _meta = load_label_file(sample.label_path)
            pages = load_pages(sample.image_path)
            text_parts, confidences = [], []
            for img in pages:
                prepared = prepare_page(img)
                lines = read_lines(prepared.gray)
                text_parts.extend(ln.text for ln in lines)
                confidences.extend(ln.confidence for ln in lines)
        except Exception as exc:
            flagged.append({"stem": sample.stem, "field": "*", "issue": "unreadable",
                            "detail": f"{type(exc).__name__}: {exc}"})
            continue

        if not text_parts:
            flagged.append({"stem": sample.stem, "field": "*", "issue": "no_text_read",
                            "detail": "OCR returned nothing"})
            continue

        page_text = " ".join(text_parts)
        page_confidence = float(np.mean(confidences)) if confidences else 0.0
        if page_confidence < min_ocr_confidence:
            continue        # the page is hard to read; not evidence against the label

        for key, value in flat.items():
            try:
                fld = kn.field(qualified_key(key))
            except (KeyError, ValueError):
                continue
            needle = str(value)
            if len(needle) < 3:
                continue    # too short to search for reliably
            score = fuzz.partial_ratio(needle.lower(), page_text.lower())
            if score < similarity_floor:
                flagged.append({
                    "stem": sample.stem, "field": key, "kind": fld.kind,
                    "issue": "label_not_in_ocr",
                    "detail": f"best partial match {score}% at page confidence "
                              f"{page_confidence:.2f}",
                })
    return flagged


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def field_frequencies(samples: Sequence[Sample], kn: Knowledge) -> dict[str, Any]:
    """How often each field appears, and the spread of its values."""
    field_counts: Counter[str] = Counter()
    value_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_keys: Counter[str] = Counter()

    for sample in samples:
        try:
            flat, _meta = load_label_file(sample.label_path)
        except Exception:
            continue
        for key, value in flat.items():
            try:
                qk = qualified_key(key)
            except ValueError:
                unknown_keys[key] += 1
                continue
            try:
                fld = kn.field(qk)
            except KeyError:
                unknown_keys[key] += 1
                continue
            field_counts[qk] += 1
            # Only record value distributions for closed sets -- never for
            # names, emails or anything identifying.
            if fld.kind in {"choice", "yesNo"}:
                value_counts[qk][str(value)] += 1

    rows = []
    for fld in kn.fields:
        qk = fld.qualified_key
        n = field_counts.get(qk, 0)
        needed = min_examples_for_target(_target_for_kind(fld.kind))
        rows.append({
            "field": qk, "label": fld.label, "kind": fld.kind,
            "n_examples": n,
            "n_needed": needed,
            "shortfall": max(0, needed - n),
            "value_spread": dict(value_counts.get(qk, {})) or None,
            "n_distinct_values": len(value_counts.get(qk, {})) or None,
        })
    return {
        "fields": rows,
        "unknown_label_keys": dict(unknown_keys),
        "n_fields_with_no_examples": sum(1 for r in rows if r["n_examples"] == 0),
    }


def _target_for_kind(kind: str) -> float:
    from .evaluate import WRONG_FILL_TARGETS
    return WRONG_FILL_TARGETS.get(kind, 0.01)


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------

def audit(directory: Path | str, kn: Knowledge,
          check_labels: bool = True,
          label_check_limit: int | None = 60) -> dict[str, Any]:
    """Full audit of a dataset directory."""
    directory = Path(directory)
    samples = discover_samples(directory)
    orphans = unlabelled_images(directory)

    report: dict[str, Any] = {
        "directory": str(directory),
        "n_images_with_labels": len(samples),
        "n_images_without_labels": len(orphans),
        "images_without_labels": [p.name for p in orphans[:50]],
    }

    if not samples:
        report["fatal"] = (
            f"no labelled pages found in {directory}. Expected image files "
            f"(jpg/png/pdf) each with a JSON label file of the same name."
        )
        return report

    writers = Counter(s.writer for s in samples)
    clients = Counter(s.client for s in samples)
    layouts = Counter(s.layout or "unknown" for s in samples)

    report.update({
        "n_writers": len(writers),
        "n_clients": len(clients),
        "writers": dict(writers.most_common()),
        "layouts": dict(layouts.most_common()),
        "pages_per_writer": {
            "min": min(writers.values()), "max": max(writers.values()),
            "mean": round(sum(writers.values()) / len(writers), 2),
        },
        "fields_per_page": {
            "min": min(s.n_fields for s in samples),
            "max": max(s.n_fields for s in samples),
            "mean": round(sum(s.n_fields for s in samples) / len(samples), 2),
        },
    })

    # Label-key sanity against the knowledge file.
    key_problems: list[str] = []
    for sample in samples:
        try:
            flat, _ = load_label_file(sample.label_path)
        except Exception as exc:
            key_problems.append(f"{sample.stem}: unreadable label file ({exc})")
            continue
        for problem in validate_against_knowledge(flat, kn):
            key_problems.append(f"{sample.stem}: {problem}")
    report["label_key_problems"] = key_problems[:100]
    report["n_label_key_problems"] = len(key_problems)

    report["coverage"] = field_frequencies(samples, kn)
    report["duplicates"] = find_duplicates(samples)
    report["personal_data"] = scan_pii(samples)

    if check_labels:
        report["label_disagreements"] = find_label_disagreements(
            samples, kn, limit=label_check_limit
        )
        report["n_label_disagreements"] = len(report["label_disagreements"])

    report["gaps"] = summarise_gaps(report, kn)
    return report


def summarise_gaps(report: dict[str, Any], kn: Knowledge) -> dict[str, Any]:
    """The 'what do I need to add' list the brief asks for."""
    rows = report.get("coverage", {}).get("fields", [])
    short = [r for r in rows if r["shortfall"] > 0]
    short.sort(key=lambda r: -r["shortfall"])

    n_writers = report.get("n_writers", 0)
    advice: list[str] = []
    if n_writers < 8:
        advice.append(
            f"only {n_writers} writer(s): with fewer than ~8 the test split "
            f"cannot show the model generalises to a new hand. Add pages from "
            f"more advisers."
        )
    if report.get("duplicates", {}).get("n_near_duplicate_pairs", 0):
        advice.append(
            f"{report['duplicates']['n_near_duplicate_pairs']} near-duplicate "
            f"pairs found: these inflate accuracy and risk leaking across splits."
        )
    layouts = report.get("layouts", {})
    if len(layouts) < 3:
        advice.append(
            f"only {len(layouts)} layout(s) present ({', '.join(layouts)}): add "
            f"label-value notes, narrative notes and printed forms so the "
            f"linker sees all three shapes."
        )
    return {
        "fields_needing_more_data": short[:60],
        "n_fields_needing_more_data": len(short),
        "total_extra_pages_implied": _implied_pages(short, report),
        "advice": advice,
    }


def _implied_pages(short: list[dict[str, Any]], report: dict[str, Any]) -> int:
    """Rough number of extra pages implied by the per-field shortfalls."""
    fields_per_page = report.get("fields_per_page", {}).get("mean", 1) or 1
    total_shortfall = sum(r["shortfall"] for r in short)
    return int(round(total_shortfall / max(1.0, float(fields_per_page))))


__all__ = ["audit", "find_duplicates", "find_label_disagreements",
           "field_frequencies", "scan_pii", "min_examples_for_target"]
