"""Latin-script corpora: EMNIST, NIST SD19, IAM, GNHK."""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Iterable

import numpy as np

from ..manifest import Sample, ShardWriter, normalise_text
from ..registry import DatasetSpec
from .generic import IMAGE_SUFFIXES, iter_images


# --------------------------------------------------------------------------
# EMNIST
# --------------------------------------------------------------------------

def _read_idx(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        dimensions = magic & 0xFF
        shape = [count]
        for _ in range(dimensions - 1):
            shape.append(struct.unpack(">I", handle.read(4))[0])
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(shape)


def _emnist_mapping(path: Path) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            mapping[int(parts[0])] = chr(int(parts[1]))
    return mapping


def emnist_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                   ) -> Iterable[Sample]:
    """EMNIST ByClass: 62 case-sensitive classes, 814k 28x28 glyphs.

    The IDX images are stored transposed relative to how they are displayed,
    and they are white-on-black. Both are fixed here rather than left for the
    preprocessor to guess at, because guessing polarity on a 28x28 glyph whose
    ink fills half the frame is exactly where that heuristic fails.
    """
    candidates = list(root.rglob("emnist-byclass-train-images-idx3-ubyte"))
    if not candidates:
        candidates = list(root.rglob("*byclass-train-images*"))
        candidates = [c for c in candidates if not c.name.endswith(".gz")]
    if not candidates:
        raise FileNotFoundError(
            "EMNIST ByClass IDX files not found under " + str(root)
            + " - expected emnist-byclass-train-images-idx3-ubyte")
    base = candidates[0].parent

    mapping_files = list(root.rglob("emnist-byclass-mapping.txt"))
    mapping = _emnist_mapping(mapping_files[0]) if mapping_files else {}

    shards = ShardWriter(spec.id, "byclass")
    samples: list[Sample] = []
    produced = 0

    for split in ("train", "test"):
        images_path = base / f"emnist-byclass-{split}-images-idx3-ubyte"
        labels_path = base / f"emnist-byclass-{split}-labels-idx1-ubyte"
        if not images_path.exists() or not labels_path.exists():
            continue
        images = _read_idx(images_path)
        labels = _read_idx(labels_path)
        for index in range(len(images)):
            if limit is not None and produced >= limit:
                break
            label = int(labels[index])
            text = mapping.get(label)
            if text is None:
                continue
            # Transpose to upright, then invert to dark ink on light paper.
            glyph = 255 - images[index].T
            shard, position = shards.add(np.ascontiguousarray(glyph))
            samples.append(Sample(
                id=f"{spec.id}/{split}/{index:07d}",
                text=text, dataset=spec.id, level="char",
                script="Latin", language="en", writer="unknown",
                shard=shard, index=position,
                height=int(glyph.shape[0]), width=int(glyph.shape[1]),
                split_hint=split,
            ))
            produced += 1
    shards.close()
    return samples


# --------------------------------------------------------------------------
# NIST Special Database 19 (by writer)
# --------------------------------------------------------------------------

SD19_SHARD_HEIGHT = 64
# 25k 64x64 glyphs is ~100 MB in the writer's buffer before it flushes.
# The default 200k would be 800 MB held in memory at once.
SD19_SHARD_SIZE = 25_000


def _sd19_members(archive) -> dict[str, list[tuple[str, str]]]:
    """class dir -> [(partition, member name)], from the zip's index alone."""
    import re as _re
    by_class: dict[str, list[tuple[str, str]]] = {}
    for name in archive.namelist():
        if not name.lower().endswith(".png"):
            continue
        parts = name.split("/")
        if len(parts) < 4:
            continue
        class_dir, partition = parts[-3], parts[-2]
        if not _re.fullmatch(r"[0-9a-f]{2}", class_dir):
            continue
        by_class.setdefault(class_dir, []).append((partition, name))
    return by_class


def nist_sd19_zip_adapter(spec: DatasetSpec, archive_path: Path,
                          limit: int | None = None) -> Iterable[Sample]:
    """Decode by_class.zip into downsampled shards, with no extraction step.

    Reading from the archive rather than an extracted tree is not a
    micro-optimisation. by_class.zip holds 814,255 files; unpacking them onto
    a OneDrive-synced disk takes longer than the download did, and the
    adapter would immediately re-read and downsample every one of them. The
    zip is random-access, so there is nothing to gain by landing them first.

    Members are decoded one at a time and handed straight to the shard
    writer, so neither the PNG bytes nor the decoded glyphs accumulate.

    When ``limit`` is set the cap is applied **per class**, so a smaller
    corpus is still balanced over all 62 classes rather than being the first
    few letters of the alphabet.
    """
    import cv2
    import zipfile

    shards = ShardWriter(spec.id, "by_class", shard_size=SD19_SHARD_SIZE)
    samples: list[Sample] = []
    index = 0

    with zipfile.ZipFile(archive_path) as archive:
        by_class = _sd19_members(archive)
        if not by_class:
            raise FileNotFoundError(
                "no by_class/<hex>/<partition>/*.png members in "
                + str(archive_path))

        per_class = None
        if limit is not None:
            per_class = max(1, limit // max(1, len(by_class)))

        for class_dir in sorted(by_class):
            try:
                text = normalise_text(chr(int(class_dir, 16)))
            except ValueError:
                continue
            if not text or not text.isprintable():
                continue
            members = sorted(by_class[class_dir], key=lambda item: item[1])
            if per_class is not None:
                members = members[:per_class]

            for partition, name in members:
                image = cv2.imdecode(
                    np.frombuffer(archive.read(name), dtype=np.uint8),
                    cv2.IMREAD_GRAYSCALE)
                if image is None or image.size == 0:
                    continue
                # The 128x128 originals are ~13 GB across the whole corpus.
                # 64px keeps far more detail than EMNIST's 28x28 at a
                # sixteenth of the storage, and the model input is 32px tall.
                height, width = image.shape[:2]
                scale = SD19_SHARD_HEIGHT / max(1, height)
                resized = cv2.resize(
                    image, (max(8, int(round(width * scale))),
                            SD19_SHARD_HEIGHT),
                    interpolation=cv2.INTER_AREA)
                shard, position = shards.add(np.ascontiguousarray(resized))
                samples.append(Sample(
                    id=f"{spec.id}/{index:07d}",
                    text=text, dataset=spec.id, level="char",
                    script="Latin", language="en",
                    # NIST cut the hsf_N partitions by writer, so the
                    # partition is the closest thing to a writer group that
                    # by_class preserves.
                    writer=f"sd19-{partition}",
                    shard=shard, index=position,
                    height=int(resized.shape[0]), width=int(resized.shape[1]),
                    split_hint="test" if partition == "hsf_4" else "train",
                ))
                index += 1

    shards.close()
    return samples


def nist_sd19_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                      ) -> Iterable[Sample]:
    """SD19 by_class, from the zip if it is there, else an extracted tree.

    Note this is a plain function, not a generator: a `return` inside a
    generator ends iteration instead of handing back the delegate, so the zip
    path would silently produce nothing.

    The label is the directory name as a hexadecimal ASCII code: ``4c`` is
    'L', ``30`` is '0'. The ``hsf_N`` partition becomes the split hint,
    because NIST cut those partitions by writer - hsf_4 is the standard test
    set and shares no writer with hsf_0-3.

    ``by_write`` is deliberately not read for labels. It was checked: it
    contains 814,255 PNGs and nothing else, and its filenames cannot be
    joined to by_class's. Inventing a label from the field type ('this came
    from the digits box, so it is a digit') would be guessing which digit,
    which is how a corpus quietly fills with wrong labels.
    """
    from ..paths import raw_dir
    archive = raw_dir(spec.id) / "by_class.zip"
    if archive.exists():
        return nist_sd19_zip_adapter(spec, archive, limit)
    return _sd19_from_tree(spec, root, limit)


def _sd19_from_tree(spec: DatasetSpec, root: Path, limit: int | None
                    ) -> Iterable[Sample]:
    """Fallback: an already-extracted by_class/<hex>/<partition>/ tree."""
    class_dirs = [p for p in root.rglob("*")
                  if p.is_dir() and re.fullmatch(r"[0-9a-f]{2}", p.name)]
    if not class_dirs:
        if any(p.is_dir() and re.fullmatch(r"[a-z]\d{4}_\d{2}", p.name)
               for p in root.rglob("*")):
            raise FileNotFoundError(
                "only the by_write partition is present under " + str(root)
                + ", and it carries no transcriptions. Fetch by_class.zip:\n"
                "  python scripts/10_fetch_datasets.py --datasets nist_sd19")
        raise FileNotFoundError(
            "no NIST SD19 class directories (two hex digits) under " + str(root))

    produced = 0
    for class_dir in sorted(class_dirs):
        try:
            text = normalise_text(chr(int(class_dir.name, 16)))
        except ValueError:
            continue
        if not text or not text.isprintable():
            continue
        for image in sorted(class_dir.rglob("*")):
            if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if limit is not None and produced >= limit:
                return
            partition = image.parent.name          # hsf_0 ... hsf_7, train_XX
            yield Sample(
                id=f"{spec.id}/{class_dir.name}/{produced:07d}",
                text=text, dataset=spec.id, level="char",
                script="Latin", language="en",
                # NIST cut hsf_N by writer, so the partition is the closest
                # thing to a writer group this partition preserves.
                writer=f"sd19-{partition}",
                image=str(image),
                split_hint="test" if partition == "hsf_4" else "train",
            )
            produced += 1


# --------------------------------------------------------------------------
# IAM (manually obtained)
# --------------------------------------------------------------------------

def _iam_writers(root: Path) -> dict[str, str]:
    """form id -> writer id, from forms.txt."""
    out: dict[str, str] = {}
    for forms in root.rglob("forms.txt"):
        for line in forms.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out[parts[0]] = parts[1]
    return out


def iam_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                ) -> Iterable[Sample]:
    """IAM lines, from ascii/lines.txt plus the lines/ image tree.

    Only runs if you obtained IAM yourself. See the registry entry: it needs a
    free account, and its licence is non-commercial.
    """
    index_files = [p for p in root.rglob("lines.txt")]
    level = "line"
    if not index_files:
        index_files = [p for p in root.rglob("words.txt")]
        level = "word"
    if not index_files:
        raise FileNotFoundError(
            "IAM ascii index not found. Put lines.txt / words.txt from "
            "ascii.tgz under " + str(root))

    writers = _iam_writers(root)
    image_index: dict[str, Path] = {}
    for path in iter_images(root):
        image_index.setdefault(path.stem, path)

    produced = 0
    for index_file in index_files:
        for line in index_file.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            sample_id, status = parts[0], parts[1]
            if status != "ok":
                continue            # IAM marks badly segmented lines itself
            text = normalise_text(parts[-1].replace("|", " "))
            image = image_index.get(sample_id)
            if image is None or not text:
                continue
            if limit is not None and produced >= limit:
                return
            form = "-".join(sample_id.split("-")[:2])
            yield Sample(
                id=f"{spec.id}/{sample_id}",
                text=text, dataset=spec.id, level=level,
                script="Latin", language="en",
                writer=f"iam-{writers.get(form, form)}",
                image=str(image),
            )
            produced += 1


# --------------------------------------------------------------------------
# GNHK (manually obtained)
# --------------------------------------------------------------------------

def gnhk_adapter(spec: DatasetSpec, root: Path, limit: int | None = None
                 ) -> Iterable[Sample]:
    """GNHK ships page photographs with per-word polygons; crop the words.

    Entries whose text GNHK marks as unreadable (%math%, %SC%, %NA%) are
    dropped: a crop with an unknown transcription is training noise.
    """
    import cv2

    produced = 0
    for meta in sorted(root.rglob("*.json")):
        image_path = next((meta.with_suffix(ext) for ext in (".jpg", ".jpeg", ".png")
                           if meta.with_suffix(ext).exists()), None)
        if image_path is None:
            continue
        try:
            entries = json.loads(meta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(entries, list):
            continue
        page = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if page is None:
            continue
        out_dir = image_path.parent / "_words"
        out_dir.mkdir(exist_ok=True)

        for position, entry in enumerate(entries):
            text = normalise_text(str(entry.get("text", "")))
            if not text or text.startswith("%"):
                continue
            polygon = entry.get("polygon") or {}
            xs = [polygon.get(f"x{i}") for i in range(4)]
            ys = [polygon.get(f"y{i}") for i in range(4)]
            if any(v is None for v in xs + ys):
                continue
            x0, x1 = max(0, int(min(xs))), min(page.shape[1], int(max(xs)))
            y0, y1 = max(0, int(min(ys))), min(page.shape[0], int(max(ys)))
            if x1 - x0 < 6 or y1 - y0 < 6:
                continue
            if limit is not None and produced >= limit:
                return
            crop_path = out_dir / f"{image_path.stem}_{position:04d}.png"
            if not crop_path.exists():
                ok, buffer = cv2.imencode(".png", page[y0:y1, x0:x1])
                if not ok:
                    continue
                buffer.tofile(str(crop_path))
            yield Sample(
                id=f"{spec.id}/{image_path.stem}/{position:04d}",
                text=text, dataset=spec.id, level="word",
                script="Latin", language="en",
                writer=f"gnhk-{image_path.stem.split('_')[0]}",
                image=str(crop_path),
                split_hint="test" if "test" in str(meta).lower() else "train",
            )
            produced += 1


__all__ = ["emnist_adapter", "nist_sd19_adapter", "iam_adapter", "gnhk_adapter"]
