"""End-to-end extraction: file in, decided fields out.

The pipeline:

    image/pdf -> preprocess -> OCR lines -> label/value candidates
              -> field matching (rules, optionally rescored by the linker)
              -> assignment (one value per field, rows for repeating sections)
              -> confidence model -> threshold -> filled | unsure | missing

Status is the point of the whole exercise:

    filled  - we are confident enough to put this in the form
    unsure  - we read something but will not fill it; top-3 candidates shown
    missing - nothing on the page looked like this field

A value only ever becomes `filled` when the calibrated probability that it is
correct clears the threshold chosen for that field's kind on the validation
set. Everything else degrades to `unsure`, never to a guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .features import confidence_features
from .knowledge import Knowledge, Field, load_knowledge
from .labels import canonical
from .matcher import FieldMatch, match_line
from .ocr import TextLine, read_lines
from .parsers import ParseResult
from .preprocess import PreparedPage, prepare_page, load_pages, load_pages_from_bytes
from .segment import build_candidates

STATUS_FILLED = "filled"
STATUS_UNSURE = "unsure"
STATUS_MISSING = "missing"


class _QualityStub:
    """Stands in for PageQuality when the values came from the OCR cache."""

    def __init__(self, values: dict[str, Any]):
        self._values = dict(values)

    def as_dict(self) -> dict[str, Any]:
        return dict(self._values)


class _PreparedStub:
    """A PreparedPage without the images, for cache-backed runs."""

    def __init__(self, quality: dict[str, Any]):
        self.quality = _QualityStub(quality)
        self.colour = None
        self.gray = None
        self.binary = None


def _copy_line(line: TextLine) -> TextLine:
    """Lines from the cache are reused across runs; copy before mutating."""
    return TextLine(
        text=line.text, box=line.box, confidence=line.confidence,
        words=list(line.words or []), page_width=line.page_width,
        page_height=line.page_height, line_index=line.line_index,
        page_index=line.page_index,
    )


@dataclass
class FieldPrediction:
    """What the pipeline decided about one field (one row of one section)."""
    flat_key: str                    # 'dependants[0].name'
    qualified_key: str               # 'dependants.name'
    kind: str
    label: str
    value: Any = None
    status: str = STATUS_MISSING
    confidence: float = 0.0          # calibrated P(correct), 0..1
    candidates: list[Any] = dc_field(default_factory=list)
    raw_text: str = ""
    source_line: int = -1
    source_box: tuple[int, int, int, int] | None = None
    page_index: int = 0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.flat_key,
            "field": self.qualified_key,
            "kind": self.kind,
            "label": self.label,
            # `value` stays null unless the field was actually filled: that is
            # the contract the Angular side relies on, and it is what stops an
            # unverified value reaching the form by accident.
            "value": self.value if self.status == STATUS_FILLED else None,
            # What the pipeline read regardless of whether it trusts it, so a
            # review UI can offer it for correction. Never write this to a
            # form without a human confirming it.
            "bestGuess": self.value,
            "status": self.status,
            "confidence": round(float(self.confidence), 4),
            "candidates": self.candidates[:3],
            "raw": self.raw_text,
            "page": self.page_index,
            "box": list(self.source_box) if self.source_box else None,
            "reason": self.reason,
        }


@dataclass
class PageResult:
    page_index: int
    lines: list[TextLine]
    prepared: PreparedPage
    matches: list[FieldMatch]


@dataclass
class ExtractionResult:
    source: str
    predictions: list[FieldPrediction]
    pages: list[PageResult] = dc_field(default_factory=list)
    warnings: list[str] = dc_field(default_factory=list)

    def filled(self) -> list[FieldPrediction]:
        return [p for p in self.predictions if p.status == STATUS_FILLED]

    def unsure(self) -> list[FieldPrediction]:
        return [p for p in self.predictions if p.status == STATUS_UNSURE]

    def as_sections(self) -> dict[str, Any]:
        """Group into the per-section JSON the Angular side saves."""
        out: dict[str, Any] = {}
        for pred in self.predictions:
            if pred.status != STATUS_FILLED:
                continue
            section, _, rest = pred.flat_key.partition(".")
            if section.endswith("]"):
                base, _, idx = section.rpartition("[")
                idx = int(idx.rstrip("]"))
                rows = out.setdefault(base, [])
                while len(rows) <= idx:
                    rows.append({})
                rows[idx][rest] = pred.value
            else:
                out.setdefault(section, {})[rest] = pred.value
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sections": self.as_sections(),
            "fields": [p.to_dict() for p in self.predictions],
            "summary": {
                "filled": len(self.filled()),
                "unsure": len(self.unsure()),
                "pages": len(self.pages),
            },
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------------
# Assignment
# --------------------------------------------------------------------------

def _row_index_for(match: FieldMatch, assigned_rows: dict[str, list[dict]],
                   kn: Knowledge) -> int:
    """Pick which row of a repeating section this match belongs to.

    Rows are separated vertically. A field goes into the first existing row
    that does not already hold it and that is vertically close; otherwise it
    starts a new row.
    """
    section = kn.section(match.field.section_id)
    if not section.is_repeating:
        return 0
    rows = assigned_rows.setdefault(section.id, [])
    y = match.line.line.cy
    line_h = max(1.0, float(match.line.line.h))
    for i, row in enumerate(rows):
        if match.field.key in row["keys"]:
            continue
        # Same row if within a few text lines of the rows seen so far.
        if abs(y - row["y"]) <= line_h * 9:
            row["keys"].add(match.field.key)
            row["y"] = (row["y"] * row["n"] + y) / (row["n"] + 1)
            row["n"] += 1
            return i
    rows.append({"keys": {match.field.key}, "y": y, "n": 1})
    return len(rows) - 1


def assign_matches(all_matches: list[FieldMatch], kn: Knowledge,
                   min_score: float = 0.15) -> dict[str, FieldMatch]:
    """Resolve competing hypotheses into at most one match per flat key.

    Greedy by score: the strongest hypothesis claims its field first, and a
    line can only supply one field so that a single line does not fill three
    fields with the same text.

    `min_score` is deliberately low. It is a sanity floor, not the fill
    decision -- that belongs to the confidence model and its per-kind
    thresholds. Setting it high silently deletes candidates, so a page the
    recogniser struggled with returns nothing at all rather than a list of
    "unsure" fields with their top-3 candidates, which is what the adviser
    actually needs to see.
    """
    ordered = sorted(all_matches, key=lambda m: -m.score)
    chosen: dict[str, FieldMatch] = {}
    used_lines: dict[int, str] = {}
    assigned_rows: dict[str, list[dict]] = {}

    for match in ordered:
        if match.score < min_score:
            continue
        line_id = (match.line.line.page_index if hasattr(match.line.line, "page_index")
                   else 0, match.line.line.line_index)
        line_key = hash(line_id)
        if line_key in used_lines:
            continue
        row = _row_index_for(match, assigned_rows, kn)
        section = kn.section(match.field.section_id)
        flat_key = (f"{section.id}[{row}].{match.field.key}" if section.is_repeating
                    else match.qualified_key)
        if flat_key in chosen:
            continue
        chosen[flat_key] = match
        used_lines[line_key] = flat_key
    return chosen


# --------------------------------------------------------------------------
# Cross-field checks -- cheap, high-value wrong-fill catches
# --------------------------------------------------------------------------

def cross_field_flags(chosen: dict[str, FieldMatch], kn: Knowledge) -> dict[str, dict[str, float]]:
    """Per-field consistency signals, consumed by the confidence model.

    These catch the errors a single-field view cannot see: a dependant older
    than the client, a retirement age already passed, a surname that does not
    match any other surname on the page.
    """
    flags: dict[str, dict[str, float]] = {k: {} for k in chosen}
    values = {k: m.parse.value for k, m in chosen.items()}

    def get(key: str):
        return values.get(key)

    today = date.today()

    def age_of(iso: Any) -> float | None:
        try:
            y, m, d = (int(p) for p in str(iso).split("-"))
            return today.year - y - ((today.month, today.day) < (m, d))
        except Exception:
            return None

    client_dob = get("personal-details.dateOfBirth")
    client_age = age_of(client_dob) if client_dob else None

    for key, match in chosen.items():
        f = flags[key]
        fld = match.field

        # Retirement age vs current age.
        if fld.key == "retirementAge" and client_age is not None:
            try:
                ra = float(match.parse.value)
                f["retirement_before_age"] = float(ra < client_age - 1)
                f["retirement_gap"] = float(ra - client_age)
            except (TypeError, ValueError):
                pass

        # Dependant date of birth vs client's.
        if fld.section_id == "dependants" and fld.key == "dateOfBirth" and client_age is not None:
            dep_age = age_of(match.parse.value)
            if dep_age is not None:
                f["dependant_older_than_client"] = float(dep_age > client_age)
                f["dependant_age"] = float(dep_age)
                # A "Son"/"Daughter" who is older than the client is wrong.
                rel_key = key.replace(".dateOfBirth", ".relationshipType")
                rel = get(rel_key)
                if rel in {"Son", "Daughter"} and dep_age >= client_age:
                    f["relationship_age_conflict"] = 1.0

        # Dependency term beyond a plausible horizon.
        if fld.key == "dependencyTermYears":
            try:
                f["term_implausible"] = float(float(match.parse.value) > 30)
            except (TypeError, ValueError):
                pass

        # Surname agreement: a last name that matches nothing else on the page
        # is more likely to be a misread.
        if fld.kind in {"name", "fullName"}:
            surnames = []
            for other_key, other in chosen.items():
                if other_key == key or other.field.kind not in {"name", "fullName"}:
                    continue
                parts = str(other.parse.value or "").split()
                if parts:
                    surnames.append(parts[-1].lower())
            mine = str(match.parse.value or "").split()
            if mine and surnames:
                f["surname_shared"] = float(mine[-1].lower() in surnames)

        # Amounts: an outlier against the other amounts on the page.
        if fld.kind == "money":
            others = [float(m.parse.value) for k2, m in chosen.items()
                      if m.field.kind == "money" and k2 != key
                      and isinstance(m.parse.value, (int, float))]
            if len(others) >= 3 and isinstance(match.parse.value, (int, float)):
                med = float(np.median(others))
                if med > 0:
                    f["money_ratio_to_median"] = float(min(100.0, match.parse.value / med))

        # onlyIf: a field whose condition is not met should not be filled.
        if fld.only_if:
            dep_key = f"{fld.section_id}.{fld.only_if['key']}"
            if key.endswith("]." + fld.key) or "[" in key:
                prefix = key.rsplit(".", 1)[0]
                dep_key = f"{prefix}.{fld.only_if['key']}"
            dep_val = values.get(dep_key)
            if "filled" in fld.only_if:
                satisfied = (dep_val not in (None, "")) == bool(fld.only_if["filled"])
            else:
                satisfied = dep_val == fld.only_if.get("equals")
            # Unknown dependency is not a violation; only a contradiction is.
            f["only_if_violated"] = float(dep_key in values and not satisfied)
    return flags


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class Extractor:
    """Runs the pipeline. Models are optional: without them it is the baseline."""

    def __init__(self, kn: Knowledge | None = None, linker=None,
                 confidence_model=None, thresholds: dict[str, float] | None = None,
                 ocr_engine: str | None = None):
        self.kn = kn or load_knowledge()
        self.linker = linker
        self.confidence_model = confidence_model
        self.thresholds = thresholds or {}
        self.ocr_engine = ocr_engine

    # -- per page -------------------------------------------------------
    def read_page(self, image: np.ndarray, page_index: int = 0,
                  cached: tuple[list[TextLine], dict[str, Any]] | None = None) -> PageResult:
        """Read one page. `cached` supplies pre-computed OCR lines and image
        quality, skipping the expensive half of the pipeline."""
        if cached is not None:
            lines, quality = cached
            lines = [_copy_line(ln) for ln in lines]
            prepared = _PreparedStub(quality)
        else:
            prepared = prepare_page(image)
            lines = read_lines(prepared.gray, engine=self.ocr_engine)
        for ln in lines:
            ln.page_index = page_index
        candidates = build_candidates(lines, self.kn)

        matches: list[FieldMatch] = []
        for cand in candidates:
            top = match_line(cand, self.kn, top_k=6)
            if self.linker is not None and top:
                top = self.linker.rescore(cand, top, self.kn.default_section.id)
            matches.extend(top)
        return PageResult(page_index=page_index, lines=lines,
                          prepared=prepared, matches=matches)

    # -- whole document --------------------------------------------------
    def extract_images(self, images: Iterable[np.ndarray], source: str = "") -> ExtractionResult:
        pages: list[PageResult] = []
        all_matches: list[FieldMatch] = []
        for i, img in enumerate(images):
            page = self.read_page(img, page_index=i)
            pages.append(page)
            all_matches.extend(page.matches)

        chosen = assign_matches(all_matches, self.kn)
        flags = cross_field_flags(chosen, self.kn)
        predictions = self._decide(chosen, flags, pages)

        warnings = self._diagnose(pages, all_matches, predictions)
        return ExtractionResult(source=source, predictions=predictions,
                                pages=pages, warnings=warnings)

    def _diagnose(self, pages: list[PageResult], all_matches: list[FieldMatch],
                  predictions: list[FieldPrediction]) -> list[str]:
        """Explain an empty or weak result instead of returning silence.

        A page that yields nothing is almost always a recognition problem, not
        a matching one, and saying so is far more useful than an empty table.
        """
        warnings: list[str] = []
        lines = [ln for p in pages for ln in p.lines]
        if not lines:
            warnings.append(
                "No text was read from this document. Check it is a photo of a "
                "page rather than a blank or very dark image."
            )
            return warnings

        mean_confidence = float(np.mean([ln.confidence for ln in lines]))
        n_filled = sum(1 for p in predictions if p.status == STATUS_FILLED)

        if not predictions:
            warnings.append(
                f"Read {len(lines)} line(s), but none of them matched a field "
                f"well enough to report. Mean recogniser confidence was "
                f"{mean_confidence:.0%}."
            )
        elif n_filled == 0:
            warnings.append(
                f"Read {len(lines)} line(s) and found {len(predictions)} "
                f"possible field(s), but none passed its confidence threshold, "
                f"so nothing was filled. Mean recogniser confidence was "
                f"{mean_confidence:.0%} — see the candidates on each unsure "
                f"field."
            )

        if mean_confidence < 0.80 and n_filled == 0:
            warnings.append(
                "The limiting factor here is handwriting recognition, not "
                "field matching. A recogniser trained on your own notes is "
                "what moves this number."
            )
        return warnings

    def extract_file(self, path: Path | str) -> ExtractionResult:
        path = Path(path)
        return self.extract_images(load_pages(path), source=str(path))

    def extract_bytes(self, data: bytes, filename: str) -> ExtractionResult:
        return self.extract_images(load_pages_from_bytes(data, filename),
                                   source=filename)

    # -- decision --------------------------------------------------------
    def _decide(self, chosen: dict[str, FieldMatch],
                flags: dict[str, dict[str, float]],
                pages: list[PageResult]) -> list[FieldPrediction]:
        preds: list[FieldPrediction] = []

        quality_by_page = {p.page_index: p.prepared.quality.as_dict() for p in pages}

        default_section = self.kn.default_section.id

        for flat_key, match in sorted(chosen.items()):
            fld = match.field
            page_index = getattr(match.line.line, "page_index", 0)
            # This must build exactly the vector the model was trained on --
            # the same helper dataset.py uses when collecting examples.
            # Building it by hand here silently zeroes the one-hot and
            # linker features and destroys the model's calibration.
            feats = confidence_features(
                match, default_section,
                linker_score=float(getattr(match, "linker_score", match.score)),
                flags=flags.get(flat_key, {}),
                image=quality_by_page.get(page_index, {}),
            )

            if self.confidence_model is not None:
                confidence = float(self.confidence_model.predict_proba_one(feats, fld))
            else:
                # Baseline: the rule score stands in for a calibrated
                # probability. It is not calibrated, and the report says so.
                confidence = float(match.score)

            threshold = self.threshold_for(fld)
            hard_block = flags.get(flat_key, {}).get("only_if_violated", 0.0) > 0
            status = (STATUS_FILLED
                      if confidence >= threshold and match.parse.valid and not hard_block
                      else STATUS_UNSURE)

            reason = match.parse.reason
            if hard_block:
                reason = (reason + " | onlyIf condition not met").strip(" |")

            preds.append(FieldPrediction(
                flat_key=flat_key,
                qualified_key=match.qualified_key,
                kind=fld.kind,
                label=fld.label,
                value=match.parse.value,
                status=status,
                confidence=confidence,
                candidates=[c for c, _ in match.parse.candidates[:3]],
                raw_text=match.parse.raw,
                source_line=match.line.line.line_index,
                source_box=match.line.line.box,
                page_index=page_index,
                reason=reason,
            ))
        return preds

    def threshold_for(self, fld: Field) -> float:
        """Threshold for this field, by qualified key then by kind."""
        if fld.qualified_key in self.thresholds:
            return self.thresholds[fld.qualified_key]
        if fld.kind in self.thresholds:
            return self.thresholds[fld.kind]
        return float(self.thresholds.get("default", 0.5))


def load_extractor(models_dir: Path | str | None = None,
                   use_models: bool = True, **kwargs) -> Extractor:
    """Build an Extractor with whatever trained models are on disk."""
    from .paths import MODELS_DIR
    models_dir = Path(models_dir or MODELS_DIR)
    linker = confidence = None
    thresholds: dict[str, float] = {}

    if use_models:
        try:
            from .linker import FieldLinker
            linker_path = models_dir / "linker.joblib"
            if linker_path.exists():
                linker = FieldLinker.load(linker_path)
        except Exception:
            linker = None
        try:
            from .confidence import ConfidenceModel
            conf_path = models_dir / "confidence.joblib"
            if conf_path.exists():
                confidence = ConfidenceModel.load(conf_path)
                thresholds = confidence.thresholds
        except Exception:
            confidence = None

    threshold_file = models_dir / "thresholds.json"
    if threshold_file.exists():
        thresholds = json.loads(threshold_file.read_text(encoding="utf-8"))

    return Extractor(linker=linker, confidence_model=confidence,
                     thresholds=thresholds, **kwargs)


__all__ = ["Extractor", "ExtractionResult", "FieldPrediction", "PageResult",
           "load_extractor", "assign_matches", "cross_field_flags",
           "STATUS_FILLED", "STATUS_UNSURE", "STATUS_MISSING"]
