"""Rule + fuzzy field matching.

Given a line's candidate (label, value) splits, score every field in the
knowledge file and return the best few. Scoring combines:

  * how well the label text matches the field's label/aliases
  * whether the value text actually parses under the field's kind
  * whether the field belongs to the section in force on the page
  * how strong the label/value split itself was

This is the candidate generator used by the baseline directly, and used by the
model path to narrow 107 fields down to a handful before the classifier
re-ranks them.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from rapidfuzz import fuzz

from .knowledge import Knowledge, Field, normalise_label
from .parsers import ParseResult, parse_value, is_blank
from .segment import LineCandidate, Pair

# OCR sometimes loses spaces entirely ("Fullname:JohnSmith"). Comparing the
# de-spaced forms recovers those without weakening normal matching.
def _despace(s: str) -> str:
    return s.replace(" ", "")


@dataclass
class FieldMatch:
    """One (line, field) hypothesis with everything needed to judge it."""
    qualified_key: str
    field: Field
    pair: Pair
    label_score: float          # 0..1 label/alias similarity
    label_margin: float         # gap to the next best *field's* label score
    parse: ParseResult
    section_bonus: float
    score: float                # combined 0..1
    line: LineCandidate

    @property
    def value(self):
        return self.parse.value

    def as_features(self) -> dict[str, float]:
        """Numeric view used by the linker and the confidence model."""
        line = self.line.line
        rx, ry, rw, rh = line.rel_box()
        return {
            "label_score": self.label_score,
            "label_margin": self.label_margin,
            "parse_score": self.parse.parse_score,
            "parse_valid": float(self.parse.valid),
            "split_confidence": self.pair.confidence,
            "ocr_confidence": float(line.confidence),
            "section_bonus": self.section_bonus,
            "combined_score": self.score,
            "rel_x": rx, "rel_y": ry, "rel_w": rw, "rel_h": rh,
            "label_len": float(len(self.pair.label_text)),
            "value_len": float(len(self.pair.value_text)),
            "line_len": float(len(line.text)),
            "n_candidates": float(len(self.parse.candidates)),
            "candidate_margin": _candidate_margin(self.parse),
            "is_sensitive": float(self.field.is_sensitive),
            "value_has_digit": float(bool(re.search(r"\d", self.pair.value_text))),
            "value_has_alpha": float(bool(re.search(r"[A-Za-z]", self.pair.value_text))),
        }


def _candidate_margin(parse: ParseResult) -> float:
    if len(parse.candidates) < 2:
        return 1.0 if parse.candidates else 0.0
    return float(parse.candidates[0][1] - parse.candidates[1][1])


def label_similarity(text: str, fld: Field) -> float:
    """Best similarity between a label fragment and the field's label/aliases."""
    norm = normalise_label(text)
    if not norm:
        return 0.0
    best = 0.0
    norm_nospace = _despace(norm)
    for form in fld.label_forms():
        if not form:
            continue
        if norm == form:
            return 1.0
        # token_set_ratio handles "date of birth" vs "birth date".
        score = max(
            fuzz.ratio(norm, form),
            fuzz.token_set_ratio(norm, form),
            fuzz.ratio(norm_nospace, _despace(form)),
        ) / 100.0
        # A label fragment that merely contains the alias ("client dob today")
        # is weaker evidence than one that is the alias.
        if form in norm and norm != form:
            score = max(score, 0.9 - 0.02 * (len(norm) - len(form)))
        # Very short aliases ("ni", "db", "m") match far too much.
        if len(form) <= 2:
            score *= 0.6 if len(norm) > 3 else 1.0
        best = max(best, score)

    # A very short label fragment is weak evidence however well it scores.
    # token_set_ratio gives "age" a perfect 100 against "retirement age"
    # because one is a subset of the other, and OCR debris like "o", "xm" or
    # "mam" lands on real field names the same way. Length is the signal that
    # separates a genuine abbreviation from noise.
    length_penalty = {1: 0.25, 2: 0.35, 3: 0.55}.get(len(norm), 1.0)
    return min(1.0, best * length_penalty)


def _section_bonus(fld: Field, line: LineCandidate, kn: Knowledge) -> float:
    """Reward a field that belongs to the section heading in force."""
    if line.section_id == "__other__":
        # Under 'partner', 'applicant 2', 'notes' etc. we must not fill.
        return -1.0
    if fld.section_id == line.section_id:
        return 0.12
    # The default section's fields can appear anywhere on an unheaded note.
    if fld.section_id == kn.default_section.id:
        return 0.0
    return -0.10


def match_pair(pair: Pair, line: LineCandidate, kn: Knowledge,
               fields: Iterable[Field] | None = None) -> list[FieldMatch]:
    """Score every field against one (label, value) split."""
    fields = list(fields if fields is not None else kn.fields)
    if is_blank(pair.value_text):
        return []

    # A pair built from a neighbouring line ("the value is on the next line")
    # is a guess about layout, not evidence from the line itself. On a clean
    # page it rescues a split label; on a badly-read page it attaches whatever
    # text happens to sit nearby to a field, which is how a page footer ends
    # up as someone's surname. Demand a near-exact label before trusting one.
    floor = 0.78 if pair.separator in {"above", "below"} else 0.45

    scored: list[tuple[Field, float, ParseResult, float]] = []
    for fld in fields:
        lab = label_similarity(pair.label_text, fld)
        if lab < floor:
            continue                      # nowhere near this field's name
        parse = parse_value(pair.value_text, fld)
        bonus = _section_bonus(fld, line, kn)
        scored.append((fld, lab, parse, bonus))

    if not scored:
        return []

    label_scores = sorted((s[1] for s in scored), reverse=True)
    out: list[FieldMatch] = []
    for fld, lab, parse, bonus in scored:
        margin = lab - next((s for s in label_scores if s < lab), 0.0)
        # A value that does not parse under this kind is strong evidence the
        # line is not this field, however well the label matched.
        parse_term = parse.parse_score if parse.valid else 0.0
        score = (0.56 * lab + 0.34 * parse_term + 0.10 * pair.confidence) + bonus
        if not parse.valid:
            score *= 0.45
        score = max(0.0, min(1.0, score))
        out.append(FieldMatch(
            qualified_key=fld.qualified_key, field=fld, pair=pair,
            label_score=lab, label_margin=margin, parse=parse,
            section_bonus=bonus, score=score, line=line,
        ))
    out.sort(key=lambda m: -m.score)
    return out


def match_line(line: LineCandidate, kn: Knowledge,
               top_k: int = 6) -> list[FieldMatch]:
    """Best field hypotheses across all of a line's candidate splits."""
    if line.is_heading:
        return []
    all_matches: list[FieldMatch] = []
    for pair in line.pairs:
        all_matches.extend(match_pair(pair, line, kn))
    if not all_matches:
        return []
    # Keep the best hypothesis per field, then the best fields overall.
    best_per_field: dict[str, FieldMatch] = {}
    for m in all_matches:
        prev = best_per_field.get(m.qualified_key)
        if prev is None or m.score > prev.score:
            best_per_field[m.qualified_key] = m
    ranked = sorted(best_per_field.values(), key=lambda m: -m.score)
    return ranked[:top_k]


def match_page(lines: Iterable[LineCandidate], kn: Knowledge,
               top_k: int = 6) -> list[list[FieldMatch]]:
    return [match_line(line, kn, top_k=top_k) for line in lines]


__all__ = ["FieldMatch", "match_line", "match_page", "match_pair",
           "label_similarity"]
