"""Indic corpora: the UCI Devanagari character set and the CVIT word sets."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from ..manifest import Sample, normalise_text
from ..registry import DatasetSpec
from .generic import IMAGE_SUFFIXES

# --------------------------------------------------------------------------
# UCI Devanagari Handwritten Character Dataset
# --------------------------------------------------------------------------

# The folders are named in transliteration ("character_10_yna"), so the actual
# Unicode characters have to be supplied. Index order follows the dataset's
# own numbering, which is the stable key; the names are a fallback.
DEVANAGARI_CONSONANTS = [
    "क", "ख", "ग", "घ", "ङ", "च", "छ", "ज", "झ", "ञ",
    "ट", "ठ", "ड", "ढ", "ण", "त", "थ", "द", "ध", "न",
    "प", "फ", "ब", "भ", "म", "य", "र", "ल", "व", "श",
    "ष", "स", "ह", "क्ष", "त्र", "ज्ञ",
]
DEVANAGARI_DIGITS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"]


def dhcd_label(folder: str) -> str | None:
    """'character_10_yna' -> 'ञ'; 'digit_7' -> '७'."""
    folder = folder.strip().lower()
    match = re.match(r"character_(\d+)", folder)
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(DEVANAGARI_CONSONANTS):
            return DEVANAGARI_CONSONANTS[index]
        return None
    match = re.match(r"digit_(\d+)", folder)
    if match:
        index = int(match.group(1))
        if 0 <= index < 10:
            return DEVANAGARI_DIGITS[index]
    return None


def uci_dhcd_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                     ) -> Iterable[Sample]:
    produced = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        text = dhcd_label(path.parent.name)
        if text is None:
            continue
        if limit is not None and produced >= limit:
            return
        parts = {p.lower() for p in path.parts}
        hint = "train" if "train" in parts else ("test" if "test" in parts else None)
        yield Sample(
            id=f"{spec.id}/{produced:07d}",
            text=text, dataset=spec.id, level="char",
            script="Devanagari", language=spec.languages[0],
            writer="unknown", image=str(path), split_hint=hint,
        )
        produced += 1


# --------------------------------------------------------------------------
# CVIT IIIT-HW-* / IIIT-INDIC-HW-WORDS
# --------------------------------------------------------------------------

_WRITER_PATTERNS = (
    re.compile(r"(?:writer|wr|w)[-_]?(\d{1,4})", re.I),
    re.compile(r"^(\d{1,4})$"),
)


def _writer_from(relative: str) -> str:
    """CVIT encodes the writer in the path; pull it out where it is there."""
    for part in Path(relative).parts:
        for pattern in _WRITER_PATTERNS:
            match = pattern.match(part)
            if match:
                return match.group(1)
    return "unknown"


def _index_files(root: Path) -> list[tuple[Path, str | None]]:
    out: list[tuple[Path, str | None]] = []
    for path in root.rglob("*.txt"):
        name = path.name.lower()
        if name.startswith("train"):
            out.append((path, "train"))
        elif name.startswith(("val", "dev")):
            out.append((path, "validation"))
        elif name.startswith("test"):
            out.append((path, "test"))
    return out


def iiit_words_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                       ) -> Iterable[Sample]:
    """CVIT word releases: a train/val/test list of ``relative/path label``.

    Writer identity, where the path carries it, is kept and prefixed with the
    dataset id so two corpora cannot collide on 'writer 3'.
    """
    index_files = _index_files(root)
    if not index_files:
        raise FileNotFoundError(
            "no train/val/test index files found under " + str(root)
            + " - the CVIT releases ship them beside the image folders")

    # One pass to locate images, so a path in the index that does not resolve
    # relative to the index file can still be found by its basename.
    by_name: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            by_name.setdefault(path.name, path)

    produced = 0
    for index_file, hint in index_files:
        try:
            lines = index_file.read_text(encoding="utf-8",
                                         errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(" ", 1)
            if len(parts) != 2:
                continue
            relative, text = parts[0].strip(), normalise_text(parts[1])
            if not text:
                continue
            candidate = (index_file.parent / relative)
            if not candidate.exists():
                candidate = by_name.get(Path(relative).name)
            if candidate is None or not candidate.exists():
                continue
            if limit is not None and produced >= limit:
                return
            yield Sample(
                id=f"{spec.id}/{produced:07d}",
                text=text, dataset=spec.id, level="word",
                script=spec.scripts[0], language=spec.languages[0],
                writer=f"{spec.id}-{_writer_from(relative)}",
                image=str(candidate), split_hint=hint,
            )
            produced += 1


__all__ = ["uci_dhcd_adapter", "iiit_words_adapter", "dhcd_label",
           "DEVANAGARI_CONSONANTS", "DEVANAGARI_DIGITS"]
