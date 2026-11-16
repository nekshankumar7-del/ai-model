"""Discovering labelled pages and turning them into training examples.

A dataset directory holds, for each page, an image (jpg/png/pdf) and a JSON
label file with the same stem. Writer and client ids come from the label
file's _meta; when they are absent we fall back to the file stem, which makes
every page its own group -- safe, if wasteful of data.

`collect_examples` runs the pipeline over a page and emits:

  linker examples      one per (line, candidate field), labelled by whether
                       that field is what the line actually says
  confidence examples  one per assigned field, labelled by whether the value
                       matches the label file

Both carry the group id, so every downstream split stays leakage-free.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import cache
from .confidence import ConfidenceExample
from .extract import Extractor, assign_matches, cross_field_flags
from .ocr import default_engine
from .features import confidence_features, linker_features
from .knowledge import Knowledge
from .labels import load_label_file, qualified_key, values_match
from .linker import LinkerExample
from .matcher import FieldMatch
from .preprocess import load_pages
from .splits import Sample

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".pdf", ".tif", ".tiff", ".bmp", ".webp"}


def discover_samples(directory: Path | str) -> list[Sample]:
    """Find every image that has a matching JSON label file."""
    directory = Path(directory)
    if not directory.exists():
        return []
    samples: list[Sample] = []
    for image_path in sorted(directory.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = image_path.with_suffix(".json")
        if not label_path.exists():
            continue
        try:
            flat, meta = load_label_file(label_path)
        except Exception:
            continue
        samples.append(Sample(
            image_path=str(image_path), label_path=str(label_path),
            writer=str(meta.get("writer") or f"unknown:{image_path.stem}"),
            client=str(meta.get("client") or f"unknown:{image_path.stem}"),
            layout=str(meta.get("layout", "")),
            n_fields=len(flat), meta=meta,
        ))
    return samples


def unlabelled_images(directory: Path | str) -> list[Path]:
    """Images with no label file -- reported by the audit."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [p for p in sorted(directory.rglob("*"))
            if p.suffix.lower() in IMAGE_SUFFIXES and not p.with_suffix(".json").exists()]


# --------------------------------------------------------------------------
# Example collection
# --------------------------------------------------------------------------

def _truth_values_for(qk: str, truth: dict[str, Any]) -> list[Any]:
    """Every truth value recorded for this field, across repeating rows."""
    out = []
    for key, value in truth.items():
        try:
            if qualified_key(key) == qk:
                out.append(value)
        except ValueError:
            continue
    return out


@dataclass
class PageExamples:
    stem: str
    group: str
    linker: list[LinkerExample]
    confidence: list[ConfidenceExample]
    n_truth_fields: int
    n_lines: int
    ocr_text: str = ""


def collect_examples(sample: Sample, kn: Knowledge, extractor: Extractor,
                     group: str | None = None,
                     use_linker: bool = False) -> PageExamples:
    """Run the pipeline over one labelled page and emit training examples."""
    truth, _meta = load_label_file(sample.label_path)

    engine = extractor.ocr_engine or default_engine()
    cached = cache.load(sample.image_path, engine)
    pages = []
    all_matches: list[FieldMatch] = []

    if cached is not None:
        for i, entry in enumerate(cached):
            page = extractor.read_page(None, page_index=i, cached=entry)
            pages.append(page)
            all_matches.extend(page.matches)
    else:
        images = load_pages(sample.image_path)
        to_store = []
        for i, img in enumerate(images):
            page = extractor.read_page(img, page_index=i)
            pages.append(page)
            all_matches.extend(page.matches)
            to_store.append((page.lines, page.prepared.quality))
        cache.store(sample.image_path, to_store, engine)

    default_section = kn.default_section.id
    group = group or sample.writer

    # --- linker examples: is this candidate field the right one? ---------
    linker_examples: list[LinkerExample] = []
    for match in all_matches:
        qk = match.qualified_key
        truth_values = _truth_values_for(qk, truth)
        # Positive when the parsed value matches one of this field's true
        # values. That ties "right field" to "right answer", which is what we
        # actually want the linker to prefer.
        positive = bool(truth_values) and any(
            values_match(match.parse.value, tv, match.field) for tv in truth_values
        )
        linker_examples.append(LinkerExample(
            features=linker_features(match, default_section),
            label=int(positive), group=group, stem=sample.stem,
            qualified_key=qk,
        ))

    # --- confidence examples: is the assigned value correct? -------------
    chosen = assign_matches(all_matches, kn)
    flags = cross_field_flags(chosen, kn)
    quality_by_page = {p.page_index: p.prepared.quality.as_dict() for p in pages}

    confidence_examples: list[ConfidenceExample] = []
    for flat_key, match in chosen.items():
        truth_value = truth.get(flat_key)
        if truth_value is None:
            # The field was filled but the page never recorded it: a spurious
            # fill, and a crucial negative example.
            correct = 0
        else:
            correct = int(values_match(match.parse.value, truth_value, match.field))

        linker_score = float(getattr(match, "linker_score", match.score))
        feats = confidence_features(
            match, default_section, linker_score,
            flags=flags.get(flat_key, {}),
            image=quality_by_page.get(match.line.line.page_index, {}),
        )
        confidence_examples.append(ConfidenceExample(
            features=feats, correct=correct, kind=match.field.kind,
            qualified_key=match.qualified_key, group=group, stem=sample.stem,
        ))

    ocr_text = "\n".join(ln.text for p in pages for ln in p.lines)
    return PageExamples(
        stem=sample.stem, group=group, linker=linker_examples,
        confidence=confidence_examples, n_truth_fields=len(truth),
        n_lines=sum(len(p.lines) for p in pages), ocr_text=ocr_text,
    )


def collect_many(samples: Sequence[Sample], kn: Knowledge, extractor: Extractor,
                 groups: dict[str, str] | None = None,
                 progress: bool = True) -> list[PageExamples]:
    out: list[PageExamples] = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        group = (groups or {}).get(sample.stem, sample.writer)
        try:
            out.append(collect_examples(sample, kn, extractor, group=group))
        except Exception as exc:
            print(f"  ! {sample.stem}: {type(exc).__name__}: {exc}")
            continue
        if progress and (i % 10 == 0 or i == total):
            print(f"  processed {i}/{total} pages", flush=True)
    return out


def extract_sample(sample: Sample, extractor: Extractor):
    """Run the full pipeline over one labelled page, using the OCR cache."""
    from .extract import ExtractionResult

    engine = extractor.ocr_engine or default_engine()
    cached = cache.load(sample.image_path, engine)
    pages = []
    all_matches: list[FieldMatch] = []

    if cached is not None:
        for i, entry in enumerate(cached):
            page = extractor.read_page(None, page_index=i, cached=entry)
            pages.append(page)
            all_matches.extend(page.matches)
    else:
        images = load_pages(sample.image_path)
        to_store = []
        for i, img in enumerate(images):
            page = extractor.read_page(img, page_index=i)
            pages.append(page)
            all_matches.extend(page.matches)
            to_store.append((page.lines, page.prepared.quality))
        cache.store(sample.image_path, to_store, engine)

    chosen = assign_matches(all_matches, extractor.kn)
    flags = cross_field_flags(chosen, extractor.kn)
    predictions = extractor._decide(chosen, flags, pages)
    return ExtractionResult(source=sample.image_path, predictions=predictions,
                            pages=pages)


def evaluate_samples(samples: Sequence[Sample], kn: Knowledge,
                     extractor: Extractor, progress: bool = True) -> list:
    """Score a set of labelled pages end to end. Returns evaluate.Outcome list."""
    from .evaluate import score_document

    outcomes = []
    total = len(samples)
    for i, sample in enumerate(samples, 1):
        try:
            truth, _meta = load_label_file(sample.label_path)
            result = extract_sample(sample, extractor)
            outcomes.extend(score_document(result, truth, kn, stem=sample.stem))
        except Exception as exc:
            print(f"  ! {sample.stem}: {type(exc).__name__}: {exc}")
        if progress and (i % 25 == 0 or i == total):
            print(f"  evaluated {i}/{total} pages", flush=True)
    return outcomes


def flatten_linker(pages: Iterable[PageExamples]) -> list[LinkerExample]:
    return [e for p in pages for e in p.linker]


def flatten_confidence(pages: Iterable[PageExamples]) -> list[ConfidenceExample]:
    return [e for p in pages for e in p.confidence]


__all__ = ["discover_samples", "unlabelled_images", "collect_examples",
           "extract_sample", "evaluate_samples",
           "collect_many", "PageExamples", "flatten_linker", "flatten_confidence"]
