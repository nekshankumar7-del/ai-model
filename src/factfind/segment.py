"""Turn OCR lines into (label, value) candidates and track section context.

A fact-find note says things in a handful of shapes:

    Date of birth: 14/03/1982      <- label and value on one line
    Date of birth                  <- label alone, value underneath
       14/03/1982
    Name ......... John Smith      <- dot leader
    DOB    14/03/1982              <- separated by a wide gap only
    John Smith, dob 14/03/1982     <- narrative, several pairs per line

This module produces LineCandidates: for each OCR line, the plausible ways to
read it as one or more label/value pairs. Field identity is decided later by
the matcher or the model, not here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Iterable

from .knowledge import Knowledge, normalise_label
from .ocr import TextLine

# Separators that mark the end of a label, best first.
SEP_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Arrows first: advisers write "Relationship -> Son" constantly, and the
    # arrow glyphs OCR emits (→ ⟶ ⇒) are unambiguous separators, unlike a
    # dash which can also sit inside a label ("Anti- Retirement age").
    (re.compile(r"\s*(?:-{1,2}>|=>|[→➔⟶⇒➜⮕])\s*"), "arrow"),
    (re.compile(r"\s*[:：]\s*-?\s*"), "colon"),
    (re.compile(r"\s*\.{3,}\s*"), "leader"),
    (re.compile(r"\s*[-–—]\s+"), "dash"),
    (re.compile(r"\s{3,}"), "gap"),
    # Last resort: a dash with no surrounding space at all. Handwriting OCR
    # routinely loses the spaces ("Client-Jane Smith"), and without this
    # the line has no separator, so the label and value are never split and
    # the field is missed entirely. Tried last so a properly spaced separator
    # always wins, and scored lower by the matcher.
    (re.compile(r"[-–—]"), "tight-dash"),
]

HEADING_RE = re.compile(r"^[^a-z]*$")       # ALL CAPS / no lower case


@dataclass
class Pair:
    """One label/value reading of part of a line."""
    label_text: str
    value_text: str
    separator: str                 # which rule produced the split
    # Where the value sits, if we can tell (used for crops and position feats).
    value_x_frac: float = 0.0
    confidence: float = 1.0        # how trustworthy the split itself is


@dataclass
class LineCandidate:
    """An OCR line plus the ways it can be read."""
    line: TextLine
    section_id: str                # section in force at this point on the page
    is_heading: bool
    pairs: list[Pair] = dc_field(default_factory=list)
    prev_text: str = ""
    next_text: str = ""
    # Set when the value was found on the following line rather than this one.
    value_from_next_line: bool = False

    @property
    def text(self) -> str:
        return self.line.text


def _split_once(text: str, max_occurrences: int = 4) -> list[Pair]:
    """Split a line at its strongest separator type, at every occurrence.

    Splitting only at the *first* separator breaks the common handwritten
    form "Anti- Retirement age - 55": the first dash is inside the label, and
    the real split is the second one. So every occurrence of the winning
    separator type becomes a candidate and the matcher decides which reading
    scores best against the knowledge file.
    """
    out: list[Pair] = []
    for pattern, name in SEP_PATTERNS:
        matches = list(pattern.finditer(text))[:max_occurrences]
        if not matches:
            continue
        for m in matches:
            label = text[: m.start()].strip()
            value = text[m.end():].strip()
            if not label:
                continue
            # A separator with nothing after it means the value is elsewhere.
            conf = 1.0 if value else 0.4
            if name == "gap":
                conf *= 0.75      # a wide gap is weaker evidence than a colon
            if name == "tight-dash":
                conf *= 0.7
            if name == "dash":
                conf *= 0.9
            out.append(Pair(label_text=label, value_text=value, separator=name,
                            confidence=conf))
        if out:
            break                 # a colon beats a dash; do not mix types
    return out


# Narrative: "name John Smith, dob 14/03/1982, married"
NARRATIVE_SPLIT = re.compile(r"\s*[,;]\s+|\s+and\s+|\s*\.\s+(?=[a-zA-Z])")


def _split_narrative(text: str) -> list[Pair]:
    """Break prose into clauses, each read as 'label-ish words then value'."""
    chunks = [c.strip() for c in NARRATIVE_SPLIT.split(text) if c.strip()]
    if len(chunks) < 2:
        return []
    pairs: list[Pair] = []
    for chunk in chunks:
        sub = _split_once(chunk)
        if sub:
            pairs.extend(Pair(p.label_text, p.value_text, "narrative",
                              confidence=p.confidence * 0.85) for p in sub)
            continue
        # No separator: assume leading words are the label and the tail the
        # value. Try every split point; the matcher decides which is right.
        words = chunk.split()
        if len(words) < 2:
            continue
        for cut in range(1, min(len(words), 6)):
            label = " ".join(words[:cut])
            value = " ".join(words[cut:])
            if not value:
                continue
            pairs.append(Pair(label_text=label, value_text=value,
                              separator="narrative-guess",
                              confidence=0.55 - 0.05 * cut))
    return pairs


def _split_by_prefix(text: str, min_label: int = 3,
                     max_label: int = 22) -> list[Pair]:
    """Split inside a word, at every character position.

    Handwriting OCR regularly loses the space between a label and its value:
    "Occupation - Accountant" comes back as "ccupabomAccountant". Splitting
    only on whitespace cannot recover that, so every cut point becomes a
    candidate and the matcher picks the one whose left side actually matches
    a field name in the knowledge file.

    Only used for lines that have no separator and barely any spaces, because
    it is the most speculative split there is.
    """
    text = text.strip()
    if len(text) < min_label + 2:
        return []
    out: list[Pair] = []
    limit = min(max_label, len(text) - 1)
    for cut in range(min_label, limit + 1):
        label = text[:cut].strip()
        value = text[cut:].strip()
        if not label or not value:
            continue
        # Prefer cuts at a case change ("ccupabomAccountant"), which is
        # usually where the lost space was.
        boundary = text[cut - 1].islower() and text[cut].isupper()
        out.append(Pair(label_text=label, value_text=value,
                        separator="prefix",
                        confidence=(0.55 if boundary else 0.38)))
    return out


def _split_by_words(text: str, max_label_words: int = 6) -> list[Pair]:
    """Fallback for 'DOB 14/03/1982' with no separator the regexes caught."""
    words = text.split()
    if len(words) < 2:
        return []
    pairs = []
    for cut in range(1, min(len(words), max_label_words + 1)):
        label = " ".join(words[:cut])
        value = " ".join(words[cut:])
        if not label or not value:
            continue
        pairs.append(Pair(label_text=label, value_text=value, separator="words",
                          confidence=0.6 - 0.06 * (cut - 1)))
    return pairs


def _fuzzy_heading(norm: str, stem: str, floor: int = 74) -> bool:
    """Is this line a badly-recognised version of a known heading?

    Whole-string similarity only, and only for short lines. `partial_ratio`
    is deliberately NOT used: it matches any line that merely *contains*
    something heading-like, which turned "Client-Jane Smith" into a
    heading because it starts with "Client" and the knowledge file lists
    "client details". A data line misread as a heading loses its value
    outright, so this errs towards missing a heading rather than inventing one.
    """
    from rapidfuzz import fuzz

    if len(norm.split()) > 3:
        return False
    # Require the stem to be long enough for a meaningful fuzzy match.
    # Short single-word stems ("wealth", "employ") are already handled
    # by the word-starts branch; letting them through the fuzzy path
    # turns company headers ("WealthMaxS") into section headings.
    if len(stem) < 8:
        return False
    return fuzz.ratio(norm, stem) >= floor


def carries_value(text: str) -> bool:
    """True when the line has a separator with something after it.

    This is what separates "Occupation" (a heading) from
    "Occupation - accountant" (a field line). Getting it wrong is expensive:
    a data line misread as a heading loses its value entirely, and every
    following line inherits the wrong section.
    """
    for pattern, _name in SEP_PATTERNS:
        for m in pattern.finditer(text):
            if text[m.end():].strip():
                return True
    return False


def detect_section(text: str, kn: Knowledge, current: str) -> tuple[str, bool]:
    """Return (section_id_in_force, is_heading).

    A heading line switches the section for everything below it. Headings
    listed under otherHeadings switch to '__other__', which suppresses filling
    -- the knowledge file says answers under those are listed, never filled.
    """
    norm = normalise_label(text)
    if not norm or len(norm) > 60:
        return current, False

    words = norm.split()
    if len(words) > 6:
        return current, False

    # A line that answers something is data, not a heading. "Dependent
    # Details ->" has a separator but nothing after it, so it stays a heading.
    if carries_value(text):
        return current, False

    # Score each candidate heading by how much of the line it accounts for,
    # so "primary income" picks Employment rather than Income & Expenditure
    # merely because that section owns a longer heading string.
    best: tuple[int, int, str] | None = None
    padded = f" {norm} "
    for stem, section_id in kn.heading_index():
        if not stem:
            continue
        if norm == stem:
            score = 3
        elif norm.startswith(stem) and (
            len(norm) == len(stem) or norm[len(stem)] == " "
        ):
            score = 2
        elif f" {stem} " in padded:
            score = 1
        elif len(words) <= 2 and any(
            w == stem or (w.startswith(stem) and len(w) - len(stem) <= 3)
            for w in words
        ):
            # Stem headings: the knowledge file says "employ" matches
            # "Employment". Only trusted on a genuinely short line, and
            # the word must not extend far past the stem.
            score = 1
        elif len(stem) >= 6 and _fuzzy_heading(norm, stem):
            # Handwritten headings get mangled too: "Dependent Details"
            # comes back as "Dependmtetal". Without this they are read as
            # data lines, which both loses the heading and hands the next
            # field a nonsense value.
            score = 1
        else:
            continue
        candidate = (score, len(stem), section_id)
        if best is None or candidate > best:
            best = candidate

    if best is not None:
        return best[2], True
    return current, False


def build_candidates(lines: Iterable[TextLine], kn: Knowledge) -> list[LineCandidate]:
    """Attach section context and candidate label/value splits to every line."""
    lines = list(lines)
    current_section = kn.default_section.id
    out: list[LineCandidate] = []

    for i, line in enumerate(lines):
        text = line.text.strip()
        section_id, is_heading = detect_section(text, kn, current_section)
        if is_heading:
            current_section = section_id

        prev_text = lines[i - 1].text if i > 0 else ""
        next_text = lines[i + 1].text if i + 1 < len(lines) else ""

        cand = LineCandidate(line=line, section_id=current_section,
                             is_heading=is_heading, prev_text=prev_text,
                             next_text=next_text)
        if not is_heading and text:
            pairs = _split_once(text)
            if not pairs or (pairs and not pairs[0].value_text):
                pairs = pairs + _split_by_words(text)
                # No separator and essentially one run of characters: the
                # space between label and value was probably lost.
                if not pairs and len(text.split()) <= 2:
                    pairs = pairs + _split_by_prefix(text)
            pairs += _split_narrative(text)
            # A whole line can also be a bare value under a label line above.
            pairs.append(Pair(label_text=prev_text.strip(), value_text=text,
                              separator="above", confidence=0.45))
            # Or a bare label whose value is the next line.
            if next_text:
                pairs.append(Pair(label_text=text, value_text=next_text.strip(),
                                  separator="below", confidence=0.4))
            cand.pairs = _dedupe(pairs)
        out.append(cand)
    return out


def _dedupe(pairs: list[Pair]) -> list[Pair]:
    seen: dict[tuple[str, str], Pair] = {}
    for p in pairs:
        if not p.label_text and not p.value_text:
            continue
        key = (normalise_label(p.label_text), normalise_label(p.value_text))
        if key not in seen or seen[key].confidence < p.confidence:
            seen[key] = p
    return sorted(seen.values(), key=lambda p: -p.confidence)


__all__ = ["Pair", "LineCandidate", "build_candidates", "detect_section"]
