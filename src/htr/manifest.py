"""The unified sample record every dataset is converted into.

One JSONL file per dataset under ``data/processed/manifests/``. A line is one
labelled image:

    {"id": "emnist/train/000001", "text": "A", "dataset": "emnist",
     "level": "char", "script": "Latin", "language": "en",
     "writer": "emnist-unknown", "shard": "emnist/train_000.npy",
     "index": 1, "height": 28, "width": 28, "split_hint": "train"}

Images live either on disk (``image``, a path relative to the project root) or
inside a compressed shard (``shard`` + ``index``). Shards exist because
EMNIST alone is 814k 28x28 images: as loose PNGs that is hundreds of thousands
of files and a filesystem that crawls; as a handful of .npy shards it is a few
hundred megabytes that memory-map cleanly.

Text is stored NFC-normalised so that visually identical Devanagari or Tamil
strings encode to one sequence rather than two.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .paths import PROCESSED_DIR, ROOT, manifest_path


def normalise_text(text: str) -> str:
    """NFC, collapse whitespace, strip. Never changes the characters read."""
    text = unicodedata.normalize("NFC", str(text))
    # Zero-width joiners carry meaning in Indic scripts; keep them, but drop
    # the formatting marks that only ever come from bad exports.
    text = text.replace("﻿", "").replace("‎", "").replace("‏", "")
    return " ".join(text.split())


@dataclass
class Sample:
    id: str
    text: str
    dataset: str
    level: str = "word"
    script: str = "Latin"
    language: str = "en"
    writer: str = "unknown"
    image: str | None = None
    shard: str | None = None
    index: int | None = None
    height: int = 0
    width: int = 0
    split_hint: str | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None},
                          ensure_ascii=False)

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Sample":
        return Sample(**{k: row.get(k) for k in Sample.__dataclass_fields__
                         if k in row})

    @property
    def group(self) -> str:
        """The unit a writer-independent split must keep together."""
        return f"{self.dataset}:{self.writer}"


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------

class ShardWriter:
    """Accumulates fixed-size greyscale images into shards.

    Uncompressed ``.npy``, deliberately. Compressed ``.npz`` is about half the
    size on disk but has to be decompressed into RAM in full, and EMNIST
    ByClass is 814k 28x28 glyphs - 640 MB resident before a single batch is
    assembled. Plain ``.npy`` memory-maps, so the operating system pages in
    only the glyphs actually being read.
    """

    def __init__(self, dataset: str, prefix: str, shard_size: int = 200_000):
        self.dir = PROCESSED_DIR / dataset
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.prefix = prefix
        self.shard_size = shard_size
        self._buffer: list[np.ndarray] = []
        self._shard = 0
        self.written: list[str] = []

    def add(self, image: np.ndarray) -> tuple[str, int]:
        name = f"{self.dataset}/{self.prefix}_{self._shard:03d}.npy"
        index = len(self._buffer)
        self._buffer.append(image)
        if len(self._buffer) >= self.shard_size:
            self.flush()
        return name, index

    def flush(self) -> None:
        if not self._buffer:
            return
        name = f"{self.prefix}_{self._shard:03d}.npy"
        np.save(self.dir / name, np.stack(self._buffer))
        self.written.append(f"{self.dataset}/{name}")
        self._buffer = []
        self._shard += 1

    def close(self) -> None:
        self.flush()


_SHARD_CACHE: dict[str, np.ndarray] = {}


def load_shard(name: str) -> np.ndarray:
    """Memory-mapped, so a 200k-glyph shard costs no resident memory."""
    cached = _SHARD_CACHE.get(name)
    if cached is None:
        cached = np.load(PROCESSED_DIR / name, mmap_mode="r")
        _SHARD_CACHE[name] = cached
    return cached


def load_image(sample: Sample) -> np.ndarray:
    """Greyscale uint8 array for one sample, from a shard or from disk."""
    if sample.shard is not None:
        # A copy, because the shard is a read-only memory map and OpenCV is
        # entitled to want a writable, contiguous buffer. 784 bytes a glyph.
        return np.array(load_shard(sample.shard)[sample.index])
    import cv2
    path = Path(sample.image)
    if not path.is_absolute():
        path = ROOT / path
    data = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8),
                        cv2.IMREAD_GRAYSCALE)
    if data is None:
        raise ValueError(f"cannot read image {path}")
    return data


# --------------------------------------------------------------------------
# Manifest IO
# --------------------------------------------------------------------------

def write_manifest(dataset_id: str, samples: Iterable[Sample]) -> dict[str, Any]:
    path = manifest_path(dataset_id)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in samples:
            handle.write(sample.to_json() + "\n")
            count += 1
    return {"dataset": dataset_id, "manifest": str(path), "samples": count}


def read_manifest(dataset_id: str) -> Iterator[Sample]:
    path = manifest_path(dataset_id)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Sample.from_dict(json.loads(line))


def read_manifests(dataset_ids: Iterable[str]) -> list[Sample]:
    out: list[Sample] = []
    for dataset_id in dataset_ids:
        out.extend(read_manifest(dataset_id))
    return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    total: int = 0
    kept: int = 0
    empty_label: int = 0
    too_long: int = 0
    unreadable: int = 0
    degenerate_image: int = 0
    duplicates: int = 0
    scripts: dict[str, int] = field(default_factory=dict)
    datasets: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _exact_fingerprint(image: np.ndarray) -> str:
    """Content hash. Two samples collide only if the pixels are identical."""
    return hashlib.blake2b(np.ascontiguousarray(image).tobytes()
                           + repr(image.shape).encode(),
                           digest_size=12).hexdigest()


def _near_fingerprint(image: np.ndarray) -> str:
    """A 16x16 mean-threshold perceptual hash.

    Deliberately NOT the default. At character level it is actively harmful:
    two different writers' 28x28 'A' collapse to the same bits, so perceptual
    dedupe would throw away most of EMNIST and most of the writer diversity
    that makes the corpus worth having. Use it only on page or line corpora,
    where the real risk is the same scan appearing twice.
    """
    import cv2
    small = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    bits = (small > small.mean()).astype(np.uint8).flatten()
    return hashlib.blake2b(bits.tobytes(), digest_size=8).hexdigest()


def validate(samples: Iterable[Sample], *, max_text_len: int = 96,
             check_images: bool = True, dedupe: bool = True,
             near_duplicates: bool = False,
             sample_limit_for_dedupe: int | None = None
             ) -> tuple[list[Sample], ValidationReport]:
    """Drop empty, over-long, unreadable, blank and duplicate samples."""
    fingerprint = _near_fingerprint if near_duplicates else _exact_fingerprint
    report = ValidationReport()
    kept: list[Sample] = []
    seen: set[str] = set()

    for sample in samples:
        report.total += 1
        text = normalise_text(sample.text)
        if not text:
            report.empty_label += 1
            continue
        if len(text) > max_text_len:
            report.too_long += 1
            continue
        sample.text = text

        if check_images:
            try:
                image = load_image(sample)
            except (ValueError, OSError, KeyError, IndexError):
                report.unreadable += 1
                continue
            if image is None or image.size == 0 or min(image.shape[:2]) < 4:
                report.degenerate_image += 1
                continue
            if float(image.std()) < 1.0:
                # A uniform crop carries no handwriting at all.
                report.degenerate_image += 1
                continue
            sample.height, sample.width = int(image.shape[0]), int(image.shape[1])

            if dedupe and (sample_limit_for_dedupe is None
                           or report.total <= sample_limit_for_dedupe):
                key = fingerprint(image) + "|" + text
                if key in seen:
                    report.duplicates += 1
                    continue
                seen.add(key)

        report.kept += 1
        report.datasets[sample.dataset] = report.datasets.get(sample.dataset, 0) + 1
        report.scripts[sample.script] = report.scripts.get(sample.script, 0) + 1
        kept.append(sample)

    return kept, report


def corpus_stats(samples: Iterable[Sample]) -> dict[str, Any]:
    samples = list(samples)
    characters: Counter[str] = Counter()
    lengths: list[int] = []
    writers: set[str] = set()
    for sample in samples:
        characters.update(sample.text)
        lengths.append(len(sample.text))
        writers.add(sample.group)
    return {
        "samples": len(samples),
        "writers": len(writers),
        "distinct_characters": len(characters),
        "mean_text_length": float(np.mean(lengths)) if lengths else 0.0,
        "max_text_length": int(max(lengths)) if lengths else 0,
        "most_common": characters.most_common(40),
    }


__all__ = ["Sample", "ShardWriter", "load_image", "write_manifest",
           "read_manifest", "read_manifests", "validate", "corpus_stats",
           "normalise_text", "ValidationReport"]
