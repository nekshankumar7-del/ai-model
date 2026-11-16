"""OCR layer.

Produces `TextLine`s: a string, a box, a per-line confidence and the word
boxes inside it. Two engines are supported behind one interface:

  rapidocr  - ONNX detection+recognition, pip-installable, no system binary,
              and the same model family can run in the browser via
              onnxruntime-web. This is the default.
  tesseract - used only if the binary is on PATH; kept because the brief names
              it as the baseline engine.

The rest of the pipeline only ever sees TextLine, so swapping engines (or
adding a fine-tuned handwriting recogniser later) touches nothing downstream.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field as dc_field
from functools import lru_cache
from typing import Any, Literal

import cv2
import numpy as np

EngineName = Literal["rapidocr", "tesseract"]


@dataclass
class Word:
    text: str
    box: tuple[int, int, int, int]      # x, y, w, h
    confidence: float


@dataclass
class TextLine:
    """One line of text read off the page."""
    text: str
    box: tuple[int, int, int, int]      # x, y, w, h in prepared-page pixels
    confidence: float                   # 0..1 from the engine
    words: list[Word] = dc_field(default_factory=list)
    page_width: int = 0
    page_height: int = 0
    line_index: int = 0
    page_index: int = 0

    @property
    def x(self) -> int: return self.box[0]

    @property
    def y(self) -> int: return self.box[1]

    @property
    def w(self) -> int: return self.box[2]

    @property
    def h(self) -> int: return self.box[3]

    @property
    def cx(self) -> float: return self.box[0] + self.box[2] / 2

    @property
    def cy(self) -> float: return self.box[1] + self.box[3] / 2

    @property
    def right(self) -> int: return self.box[0] + self.box[2]

    @property
    def bottom(self) -> int: return self.box[1] + self.box[3]

    def rel_box(self) -> tuple[float, float, float, float]:
        """Box as fractions of the page, so features are resolution-free."""
        pw = max(1, self.page_width)
        ph = max(1, self.page_height)
        return (self.x / pw, self.y / ph, self.w / pw, self.h / ph)

    def crop(self, image: np.ndarray, pad: int = 3) -> np.ndarray:
        h, w = image.shape[:2]
        x0 = max(0, self.x - pad)
        y0 = max(0, self.y - pad)
        x1 = min(w, self.right + pad)
        y1 = min(h, self.bottom + pad)
        return image[y0:y1, x0:x1]


class OcrError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# Engines
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _rapidocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:                      # pragma: no cover
        raise OcrError(
            "rapidocr-onnxruntime is not installed; pip install -r requirements.txt"
        ) from exc
    # text_score low: we want the raw score and decide ourselves what is too
    # weak, rather than letting the engine drop lines silently.
    return RapidOCR(text_score=0.15)


def tesseract_available() -> bool:
    if shutil.which("tesseract"):
        return True
    for candidate in (
        os.environ.get("TESSERACT_PATH"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ):
        if candidate and os.path.exists(candidate):
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = candidate
                return True
            except ImportError:
                return False
    return False


def default_engine() -> EngineName:
    return "tesseract" if os.environ.get("FACTFIND_OCR") == "tesseract" else "rapidocr"


def _run_rapidocr(image: np.ndarray) -> list[TextLine]:
    engine = _rapidocr()
    # RapidOCR wants 3-channel; feed the flattened greyscale as BGR so the
    # detector sees the same pixels the crops will come from.
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        rgb = image
    result, _ = engine(rgb)
    lines: list[TextLine] = []
    if not result:
        return lines
    h, w = image.shape[:2]
    for idx, item in enumerate(result):
        quad, text, score = item[0], item[1], item[2]
        pts = np.array(quad, dtype=np.float32).reshape(-1, 2)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
        box = (int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0)))
        if not str(text).strip():
            continue
        lines.append(TextLine(text=str(text), box=box, confidence=float(score),
                              page_width=w, page_height=h, line_index=idx))
    return lines


def _run_tesseract(image: np.ndarray, psm: int = 6) -> list[TextLine]:
    import pytesseract
    from pytesseract import Output

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    config = f"--oem 1 --psm {psm}"
    data = pytesseract.image_to_data(gray, config=config, output_type=Output.DICT)

    h, w = gray.shape[:2]
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for i, txt in enumerate(data["text"]):
        if not str(txt).strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        grouped.setdefault(key, []).append(i)

    lines: list[TextLine] = []
    for idx, (_key, members) in enumerate(sorted(grouped.items())):
        words = []
        confs = []
        for i in members:
            conf = float(data["conf"][i])
            conf = max(0.0, conf) / 100.0
            words.append(Word(text=str(data["text"][i]),
                              box=(data["left"][i], data["top"][i],
                                   data["width"][i], data["height"][i]),
                              confidence=conf))
            confs.append(conf)
        if not words:
            continue
        x0 = min(wd.box[0] for wd in words)
        y0 = min(wd.box[1] for wd in words)
        x1 = max(wd.box[0] + wd.box[2] for wd in words)
        y1 = max(wd.box[1] + wd.box[3] for wd in words)
        text = " ".join(wd.text for wd in words)
        lines.append(TextLine(text=text, box=(x0, y0, x1 - x0, y1 - y0),
                              confidence=float(np.mean(confs)) if confs else 0.0,
                              words=words, page_width=w, page_height=h,
                              line_index=idx))
    return lines


# --------------------------------------------------------------------------
# Public interface
# --------------------------------------------------------------------------

def read_lines(image: np.ndarray, engine: EngineName | None = None) -> list[TextLine]:
    """Read every text line on a prepared page, top-to-bottom, left-to-right."""
    engine = engine or default_engine()
    if engine == "tesseract":
        if not tesseract_available():
            raise OcrError(
                "engine='tesseract' requested but the tesseract binary was not found. "
                "Install it, or set FACTFIND_OCR=rapidocr."
            )
        lines = _run_tesseract(image)
    elif engine == "rapidocr":
        lines = _run_rapidocr(image)
    else:
        raise OcrError(f"unknown OCR engine {engine!r}")

    # merge_split_lines already returns whole lines in reading order.
    lines = merge_split_lines(lines)
    for i, ln in enumerate(lines):
        ln.line_index = i
    return lines


def _median_height(lines: list[TextLine]) -> float:
    if not lines:
        return 1.0
    return float(np.median([ln.h for ln in lines])) or 1.0


def merge_split_lines(lines: list[TextLine], overlap_ratio: float = 0.4,
                      x_gap: float = 8.0) -> list[TextLine]:
    """Group detected boxes into whole written lines, left to right.

    The detector routinely splits one handwritten line into several boxes --
    "Country of Birth" and "USA" come back separately because the writer left
    a wide gap before the value. Two things then go wrong: the recogniser sees
    fragments with no context, and the boxes can come back out of order, so
    the value appears before its own label.

    So rows are formed by *vertical overlap* rather than centre distance,
    which is robust to a tall descender or a value written slightly high, and
    every box in a row is joined in left-to-right order. The horizontal gap
    allowance is generous, because the gap between a label and its value on a
    hand-written note is often several times the text height.
    """
    if not lines:
        return []
    med_h = _median_height(lines)

    def vertical_overlap(a: TextLine, b: TextLine) -> float:
        top = max(a.y, b.y)
        bottom = min(a.bottom, b.bottom)
        if bottom <= top:
            return 0.0
        return (bottom - top) / max(1.0, min(a.h, b.h))

    def row_centre(row: list[TextLine]) -> float:
        return sum(l.y + l.h / 2 for l in row) / len(row)

    rows: list[list[TextLine]] = []
    for ln in sorted(lines, key=lambda l: (l.y, l.x)):
        # Score every existing row and take the BEST, not the first that
        # passes. On tightly spaced narrative handwriting, ascenders and
        # descenders make a box overlap two rows at once, and taking the
        # first match puts the text on the wrong line -- which is how
        # "Indian nationality" ended up above "He is".
        best_row = None
        best_score = 0.0
        for row in rows:
            reachable = any(
                (ln.x - other.right if ln.x >= other.right
                 else other.x - ln.right) <= x_gap * med_h
                for other in row
            )
            if not reachable:
                continue        # a separate column, not the same line
            centre_gap = abs((ln.y + ln.h / 2) - row_centre(row))
            # Baseline distance, not overlap, decides the row. On closely
            # written narrative notes the ascenders of one line overlap the
            # descenders of the line above, so an overlap test happily merges
            # two different sentences into one. Vertical centres stay
            # separated by roughly a line height even then.
            if centre_gap > 0.45 * med_h:
                continue
            overlap = max(vertical_overlap(ln, other) for other in row)
            score = overlap + max(0.0, 1.0 - centre_gap / max(1.0, med_h))
            if score > best_score:
                best_score, best_row = score, row

        if best_row is None:
            rows.append([ln])
        else:
            best_row.append(ln)

    merged: list[TextLine] = []
    for row in rows:
        row.sort(key=lambda l: l.x)          # reading order within the line
        first = row[0]
        x0 = min(l.x for l in row)
        y0 = min(l.y for l in row)
        x1 = max(l.right for l in row)
        y1 = max(l.bottom for l in row)
        merged.append(TextLine(
            text=" ".join(l.text.strip() for l in row if l.text.strip()),
            box=(x0, y0, x1 - x0, y1 - y0),
            confidence=min(l.confidence for l in row),
            words=[w for l in row for w in (l.words or [])],
            page_width=first.page_width, page_height=first.page_height,
            page_index=first.page_index,
        ))
    merged.sort(key=lambda l: (l.y + l.h / 2, l.x))
    return merged


def read_text(image: np.ndarray, engine: EngineName | None = None) -> str:
    return "\n".join(ln.text for ln in read_lines(image, engine))


def engine_report() -> dict[str, Any]:
    """What is actually available on this machine, for the audit report."""
    report: dict[str, Any] = {"default": default_engine(), "tesseract": tesseract_available()}
    try:
        _rapidocr()
        report["rapidocr"] = True
    except Exception as exc:
        report["rapidocr"] = False
        report["rapidocr_error"] = str(exc)
    return report


__all__ = ["TextLine", "Word", "read_lines", "read_text", "OcrError",
           "engine_report", "tesseract_available", "default_engine",
           "merge_split_lines"]
