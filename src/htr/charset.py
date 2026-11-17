"""A Unicode charset built from the corpus rather than hard-coded.

The existing fact-find recogniser uses an ASCII charset whose index order is
baked into shipped ONNX models, so it must not be touched. This one is
derived from whatever manifests are being trained on, which is the only way
to guarantee that every label is encodable - and the only way Devanagari,
Bengali, Tamil, Gurmukhi or Arabic script work at all.

Two decisions worth stating:

* **One index per Unicode codepoint, after NFC.** A Devanagari syllable is a
  consonant plus a matra plus sometimes a virama, and CTC is perfectly happy
  to emit them as a sequence. Treating a rendered cluster as one class would
  blow the class count out to thousands and starve most of them.
* **Rare characters are dropped, and so are the samples containing them.** A
  class with four examples is a class the model cannot learn and a label it
  will never get right; leaving it in only inflates the class count and the
  reported error.
"""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

BLANK = 0


def script_of(ch: str) -> str:
    """A coarse script name, from the Unicode block the codepoint sits in."""
    code = ord(ch)
    blocks = [
        (0x0000, 0x007F, "Latin"), (0x0080, 0x024F, "Latin"),
        (0x0370, 0x03FF, "Greek"), (0x0400, 0x04FF, "Cyrillic"),
        (0x0590, 0x05FF, "Hebrew"), (0x0600, 0x06FF, "Arabic"),
        (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
        (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
        (0x0B00, 0x0B7F, "Odia"), (0x0B80, 0x0BFF, "Tamil"),
        (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
        (0x0D00, 0x0D7F, "Malayalam"), (0x2000, 0x206F, "Punctuation"),
        (0x20A0, 0x20CF, "Currency"),
    ]
    for low, high, name in blocks:
        if low <= code <= high:
            if name == "Latin" and ch.isdigit():
                return "Digits"
            return name
    return unicodedata.category(ch)


@dataclass
class UnicodeCharset:
    """Index 0 is the CTC blank; every other index is one Unicode codepoint."""
    characters: tuple[str, ...]

    def __post_init__(self) -> None:
        self.characters = tuple(self.characters)
        self._to_index = {c: i + 1 for i, c in enumerate(self.characters)}
        self._to_char = {i + 1: c for i, c in enumerate(self.characters)}

    # -- construction ---------------------------------------------------
    @staticmethod
    def from_texts(texts: Iterable[str], *, min_count: int = 1,
                   always_include: str = "") -> "UnicodeCharset":
        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(unicodedata.normalize("NFC", text))
        kept = {c for c, n in counts.items() if n >= min_count}
        kept.update(always_include)
        kept.discard("")
        # Sort by (script, codepoint) so the index order is stable and
        # inspectable rather than dependent on iteration order.
        ordered = sorted(kept, key=lambda c: (script_of(c), ord(c)))
        return UnicodeCharset(tuple(ordered))

    # -- properties -----------------------------------------------------
    @property
    def n_classes(self) -> int:
        return len(self.characters) + 1

    def __len__(self) -> int:
        return len(self.characters)

    def __contains__(self, ch: str) -> bool:
        return ch in self._to_index

    # -- coding ---------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        text = unicodedata.normalize("NFC", text)
        return [self._to_index[c] for c in text if c in self._to_index]

    def decode_indices(self, indices: Iterable[int]) -> str:
        return "".join(self._to_char.get(int(i), "") for i in indices)

    def supports(self, text: str) -> bool:
        return all(c in self._to_index
                   for c in unicodedata.normalize("NFC", text))

    def filter(self, text: str) -> str:
        return "".join(c for c in unicodedata.normalize("NFC", text)
                       if c in self._to_index)

    # -- reporting / persistence ---------------------------------------
    def scripts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for ch in self.characters:
            name = script_of(ch)
            out[name] = out.get(name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "characters": list(self.characters),
            "blankIndex": BLANK,
            "nClasses": self.n_classes,
            "scripts": self.scripts(),
            "encoding": "NFC, one index per Unicode codepoint",
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                        encoding="utf-8")
        return path

    @staticmethod
    def load(path: Path | str) -> "UnicodeCharset":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        return UnicodeCharset(tuple(blob["characters"]))


def build_charset(texts: Sequence[str], *, min_count: int = 5
                  ) -> tuple[UnicodeCharset, dict[str, Any]]:
    """Build a charset and report what it cost to drop the rare characters."""
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(unicodedata.normalize("NFC", text))
    charset = UnicodeCharset.from_texts(texts, min_count=min_count)
    dropped = sorted((c for c, n in counts.items() if n < min_count),
                     key=lambda c: -counts[c])
    unusable = sum(1 for text in texts if not charset.supports(text))
    report = {
        "n_classes": charset.n_classes,
        "min_count": min_count,
        "scripts": charset.scripts(),
        "dropped_characters": len(dropped),
        "dropped_examples": [{"char": c, "codepoint": f"U+{ord(c):04X}",
                              "count": counts[c]} for c in dropped[:30]],
        "samples_made_unencodable": unusable,
        "samples_total": len(texts),
    }
    return charset, report


__all__ = ["UnicodeCharset", "build_charset", "script_of", "BLANK"]
