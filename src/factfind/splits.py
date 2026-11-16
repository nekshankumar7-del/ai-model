"""Leakage-free splits.

The brief is explicit: split by writer and by client, never the same person or
note in train and test. A writer's handwriting and a client's answers both
leak, so the grouping key is the pair, and whole groups move together.

Groups are formed by connected components over (writer, client): if any page
links writer W to client C, every page of W and every page of C land in the
same split. That is stricter than grouping on either alone, and it is the only
way to guarantee neither leaks.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import SPLITS_FILE


@dataclass
class Sample:
    """One labelled page."""
    image_path: str
    label_path: str
    writer: str
    client: str
    layout: str = ""
    n_fields: int = 0
    meta: dict[str, Any] | None = None

    @property
    def stem(self) -> str:
        return Path(self.image_path).stem


class DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def group_key(samples: Sequence[Sample]) -> dict[str, str]:
    """Map each sample stem to its connected-component group id."""
    ds = DisjointSet()
    for s in samples:
        ds.union(f"w:{s.writer}", f"c:{s.client}")
    return {s.stem: ds.find(f"w:{s.writer}") for s in samples}


def make_splits(samples: Sequence[Sample], train: float = 0.70,
                val: float = 0.15, test: float = 0.15,
                seed: int = 42) -> dict[str, list[str]]:
    """Assign whole groups to train/val/test, targeting the given proportions.

    Groups are placed largest-first into whichever split is furthest below its
    quota. With few writers the realised proportions will not be exact; the
    audit reports what was actually achieved rather than pretending.
    """
    total = train + val + test
    train, val, test = train / total, val / total, test / total

    groups = defaultdict(list)
    for stem, gid in group_key(samples).items():
        groups[gid].append(stem)

    order = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    rng = random.Random(seed)
    # Shuffle within equal sizes so the split is not an artefact of naming.
    rng.shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))

    n_total = sum(len(v) for v in groups.values())
    quotas = {"train": train * n_total, "val": val * n_total, "test": test * n_total}
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    for _gid, stems in order:
        deficits = {k: quotas[k] - len(out[k]) for k in out}
        target = max(deficits, key=lambda k: deficits[k])
        out[target].extend(stems)
    return out


def save_splits(splits: dict[str, list[str]], samples: Sequence[Sample],
                path: Path | str | None = None) -> Path:
    path = Path(path or SPLITS_FILE)
    gk = group_key(samples)
    payload = {
        "splits": splits,
        "groups": {stem: gid for stem, gid in gk.items()},
        "counts": {k: len(v) for k, v in splits.items()},
        "writers": {k: sorted({s.writer for s in samples if s.stem in set(v)})
                    for k, v in splits.items()},
        "clients": {k: sorted({s.client for s in samples if s.stem in set(v)})
                    for k, v in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_splits(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path or SPLITS_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"no splits file at {path}; run scripts/02_prepare_splits.py first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_no_leakage(splits: dict[str, list[str]],
                      samples: Sequence[Sample]) -> list[str]:
    """Return a list of leaks. Empty means the splits are clean."""
    by_stem = {s.stem: s for s in samples}
    problems: list[str] = []
    seen_writer: dict[str, str] = {}
    seen_client: dict[str, str] = {}
    for split_name, stems in splits.items():
        for stem in stems:
            s = by_stem.get(stem)
            if s is None:
                problems.append(f"{split_name}: unknown sample {stem!r}")
                continue
            if s.writer in seen_writer and seen_writer[s.writer] != split_name:
                problems.append(
                    f"writer {s.writer!r} appears in both "
                    f"{seen_writer[s.writer]} and {split_name}"
                )
            if s.client in seen_client and seen_client[s.client] != split_name:
                problems.append(
                    f"client {s.client!r} appears in both "
                    f"{seen_client[s.client]} and {split_name}"
                )
            seen_writer.setdefault(s.writer, split_name)
            seen_client.setdefault(s.client, split_name)
    return sorted(set(problems))


def group_folds(samples: Sequence[Sample], stems: Iterable[str],
                n_folds: int = 5, seed: int = 0) -> list[tuple[list[str], list[str]]]:
    """k-fold by group over train+validation, for model selection."""
    stems = list(stems)
    gk = group_key(samples)
    groups = defaultdict(list)
    for stem in stems:
        groups[gk.get(stem, stem)].append(stem)

    order = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for _gid, members in order:
        smallest = min(range(n_folds), key=lambda i: len(folds[i]))
        folds[smallest].extend(members)

    out = []
    for i in range(n_folds):
        val_stems = folds[i]
        train_stems = [s for j, f in enumerate(folds) if j != i for s in f]
        if val_stems and train_stems:
            out.append((train_stems, val_stems))
    return out


__all__ = ["Sample", "make_splits", "save_splits", "load_splits",
           "verify_no_leakage", "group_folds", "group_key"]
