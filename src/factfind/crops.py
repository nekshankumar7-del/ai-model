"""Cutting labelled line crops out of real pages.

The detector finds the lines on your notepad pages perfectly well -- it is
only recognition that fails on cursive. So we reuse the detector's boxes to
cut crops, show each crop to a human with the current (wrong) reading
pre-filled, and collect the corrections. Those corrections are the training
set for the recogniser.

Crops and their transcriptions live in `data/crops/`:

    data/crops/
      images/<page-stem>-l003.png
      labels.jsonl          one {"image", "text", "writer", "page"} per line

labels.jsonl is append-friendly, so labelling can stop and resume.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .ocr import read_lines
from .paths import DATA_DIR
from .preprocess import load_pages, prepare_page

CROPS_DIR = DATA_DIR / "crops"
CROP_IMAGES = CROPS_DIR / "images"
LABELS_FILE = CROPS_DIR / "labels.jsonl"

# Crops below this are detector noise, not writing.
MIN_CROP_WIDTH = 12
MIN_CROP_HEIGHT = 8
# A line of text is always wider than it is tall. Anything squarer is either
# a detection on a rotated page, a stray mark, or a column of a table -- none
# of which belong in a handwriting training set, and all of which waste the
# labeller's time.
MIN_ASPECT_RATIO = 1.5


@dataclass
class CropRecord:
    image: str                 # path relative to CROPS_DIR
    text: str                  # the human-corrected transcription
    page: str                  # source page stem
    writer: str = "unknown"
    ocr_guess: str = ""
    ocr_confidence: float = 0.0
    line_index: int = 0

    def resolved(self) -> Path:
        return CROPS_DIR / self.image


def cut_page(image_path: Path | str, writer: str = "unknown",
             pad: int = 1, out_dir: Path | None = None) -> list[CropRecord]:
    """Detect lines on a page and write one crop image per line.

    Returns records with the OCR's guess filled in but `text` left empty --
    nothing is a label until a human confirms it.
    """
    image_path = Path(image_path)
    out_dir = out_dir or CROP_IMAGES
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[CropRecord] = []
    for page_index, page in enumerate(load_pages(image_path)):
        prepared = prepare_page(page)
        lines = read_lines(prepared.gray)
        for line in lines:
            crop = line.crop(prepared.gray, pad=pad)
            if crop.size == 0:
                continue
            h, w = crop.shape[:2]
            if w < MIN_CROP_WIDTH or h < MIN_CROP_HEIGHT:
                continue
            if w / max(1, h) < MIN_ASPECT_RATIO:
                continue
            stem = f"{image_path.stem}-p{page_index}-l{line.line_index:03d}"
            target = out_dir / f"{stem}.png"
            cv2.imwrite(str(target), crop)
            records.append(CropRecord(
                image=str(target.relative_to(CROPS_DIR)).replace("\\", "/"),
                text="", page=image_path.stem, writer=writer,
                ocr_guess=line.text, ocr_confidence=float(line.confidence),
                line_index=line.line_index,
            ))
    return records


def cut_directory(directory: Path | str, writer_from_meta: bool = True
                  ) -> list[CropRecord]:
    """Cut crops from every image in a directory."""
    directory = Path(directory)
    from .preprocess import PAGE_SUFFIXES as suffixes
    out: list[CropRecord] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in suffixes:
            continue
        writer = "unknown"
        if writer_from_meta:
            label = path.with_suffix(".json")
            if label.exists():
                try:
                    meta = json.loads(label.read_text(encoding="utf-8")).get("_meta", {})
                    writer = str(meta.get("writer", "unknown"))
                except (json.JSONDecodeError, OSError):
                    pass
        out.extend(cut_page(path, writer=writer))
    return out


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def load_labels(path: Path | str | None = None) -> list[CropRecord]:
    path = Path(path or LABELS_FILE)
    if not path.exists():
        return []
    out: list[CropRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(CropRecord(**json.loads(line)))
        except (json.JSONDecodeError, TypeError):
            continue
    return out


def save_labels(records: Iterable[CropRecord], path: Path | str | None = None) -> Path:
    path = Path(path or LABELS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return path


def merge_records(existing: list[CropRecord],
                  incoming: list[CropRecord]) -> list[CropRecord]:
    """Keep human transcriptions; add newly-cut crops that are not yet known."""
    by_image = {r.image: r for r in existing}
    for record in incoming:
        if record.image in by_image:
            continue                       # never clobber a confirmed label
        by_image[record.image] = record
    return sorted(by_image.values(), key=lambda r: (r.page, r.line_index))


def transcribed(records: Iterable[CropRecord]) -> list[CropRecord]:
    """Only the crops a human has actually transcribed."""
    return [r for r in records if r.text.strip()]


def to_samples(records: Iterable[CropRecord]):
    """Convert to the CropSample type the trainer consumes."""
    from .recognizer_train import CropSample
    return [
        CropSample(text=r.text, image_path=str(r.resolved()),
                   writer=r.writer, source="real")
        for r in records if r.text.strip()
    ]


def split_by_writer(records: list[CropRecord], validation_fraction: float = 0.2,
                    seed: int = 0) -> tuple[list[CropRecord], list[CropRecord]]:
    """Hold out whole writers, so validation measures a new hand.

    Falls back to holding out whole pages when there is only one writer --
    weaker, and the training script says so rather than pretending otherwise.
    """
    import random

    rng = random.Random(seed)
    writers = sorted({r.writer for r in records})
    if len(writers) > 1:
        rng.shuffle(writers)
        n_val = max(1, int(round(len(writers) * validation_fraction)))
        held = set(writers[:n_val])
        key = lambda r: r.writer in held          # noqa: E731
    else:
        pages = sorted({r.page for r in records})
        rng.shuffle(pages)
        n_val = max(1, int(round(len(pages) * validation_fraction)))
        held = set(pages[:n_val])
        key = lambda r: r.page in held            # noqa: E731

    validation = [r for r in records if key(r)]
    train = [r for r in records if not key(r)]
    return train, validation


def stats(records: list[CropRecord]) -> dict[str, Any]:
    done = transcribed(records)
    chars = sum(len(r.text) for r in done)
    return {
        "n_crops": len(records),
        "n_transcribed": len(done),
        "n_remaining": len(records) - len(done),
        "n_characters": chars,
        "n_writers": len({r.writer for r in done}) if done else 0,
        "n_pages": len({r.page for r in records}),
        "mean_length": round(chars / len(done), 1) if done else 0.0,
    }


__all__ = ["CropRecord", "cut_page", "cut_directory", "load_labels",
           "save_labels", "merge_records", "transcribed", "to_samples",
           "split_by_writer", "stats", "CROPS_DIR", "CROP_IMAGES", "LABELS_FILE"]
