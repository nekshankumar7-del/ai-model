"""Adapters for the two layouts most hand-placed datasets already use.

``<root>/<label>/<image>.png``                 - one directory per class
``<root>/images/... + labels.txt``             - a path/label mapping file

Between them these cover CENSUS-HWR, the HP Labs releases, and most of what
anyone drops into ``data/extracted/`` by hand.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from ..manifest import Sample, normalise_text
from ..registry import DatasetSpec

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm", ".ppm"}


def iter_images(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def find_label_files(root: Path) -> list[Path]:
    names = {"labels.txt", "label.txt", "annotation.txt", "annotations.txt",
             "gt.txt", "train.txt", "test.txt", "val.txt"}
    return [p for p in root.rglob("*") if p.is_file() and p.name.lower() in names]


def _split_hint(path: Path) -> str | None:
    parts = {p.lower() for p in path.parts}
    for hint, names in (("train", {"train", "training"}),
                        ("test", {"test", "testing", "eval"}),
                        ("validation", {"val", "valid", "validation", "dev"})):
        if names & parts:
            return hint
    return None


def _from_label_files(spec: DatasetSpec, root: Path) -> Iterator[Sample]:
    seen = 0
    for label_file in find_label_files(root):
        hint = _split_hint(label_file)
        for line in label_file.read_text(encoding="utf-8",
                                         errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split(" ", 1)
            if len(parts) != 2:
                continue
            rel, text = parts[0].strip(), normalise_text(parts[1])
            image = (label_file.parent / rel)
            if not image.exists():
                matches = list(root.rglob(Path(rel).name))
                if not matches:
                    continue
                image = matches[0]
            yield Sample(
                id=f"{spec.id}/{seen:07d}",
                text=text, dataset=spec.id, level=spec.level,
                script=spec.scripts[0], language=spec.languages[0],
                writer="unknown",
                image=str(image), split_hint=hint,
            )
            seen += 1


def folder_labels_adapter(spec: DatasetSpec, root: Path) -> Iterable[Sample]:
    """Prefer an explicit label file; fall back to directory-per-class."""
    produced = 0
    for sample in _from_label_files(spec, root):
        produced += 1
        yield sample
    if produced:
        return

    for index, path in enumerate(iter_images(root)):
        label = normalise_text(path.parent.name)
        if not label or label.lower() in {"images", "data", "img"}:
            continue
        yield Sample(
            id=f"{spec.id}/{index:07d}",
            text=label, dataset=spec.id, level=spec.level,
            script=spec.scripts[0], language=spec.languages[0],
            writer="unknown", image=str(path), split_hint=_split_hint(path),
        )


__all__ = ["folder_labels_adapter", "iter_images", "find_label_files"]
