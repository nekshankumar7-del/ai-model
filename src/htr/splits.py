"""Train / validation / test splits that do not leak a writer across sides.

The rule is simple and the consequence is not: every crop by one writer goes
to exactly one side. A model that has seen 400 of someone's letters and is
then tested on the 401st is not being tested on handwriting it has never seen,
and the number it produces is worth nothing.

Two honest complications are handled explicitly rather than hidden:

* **Some corpora have no writer identity.** EMNIST discarded it. For those,
  the split falls back to a random sample-level split, and the report says so
  per dataset, so nobody later mistakes a within-writer number for a
  generalisation number.
* **Splits are per dataset.** Stratifying this way keeps every corpus
  represented on every side; pooling first would let one large dataset take
  the whole test set and quietly make the test set monolingual.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .manifest import Sample
from .paths import SPLITS_FILE

TRAIN, VALIDATION, TEST = "train", "validation", "test"


def _stable_fraction(key: str, salt: str) -> float:
    """A deterministic value in [0, 1) for a group key.

    Hashing rather than shuffling means adding a dataset does not reshuffle
    the datasets already split, so yesterday's test set is still the test set.
    """
    digest = hashlib.blake2b((salt + "|" + key).encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big") / float(1 << 64)


@dataclass
class SplitReport:
    counts: dict[str, int]
    per_dataset: dict[str, dict[str, Any]]
    writer_independent: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {"counts": self.counts, "per_dataset": self.per_dataset,
                "writer_independent": self.writer_independent}


def assign(samples: Sequence[Sample], *, validation: float = 0.1,
           test: float = 0.1, salt: str = "htr-v1",
           min_writers_for_grouping: int = 4,
           respect_hints: bool = True) -> tuple[dict[str, list[Sample]], SplitReport]:
    """Split by writer where possible, by sample where not."""
    by_dataset: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        by_dataset[sample.dataset].append(sample)

    out: dict[str, list[Sample]] = {TRAIN: [], VALIDATION: [], TEST: []}
    per_dataset: dict[str, dict[str, Any]] = {}
    writer_independent: dict[str, bool] = {}

    for dataset, rows in sorted(by_dataset.items()):
        writers = {r.writer for r in rows}
        grouped = len(writers) >= min_writers_for_grouping and "unknown" not in writers
        writer_independent[dataset] = grouped

        # An official split beats anything invented here, and it is honoured
        # even without writer ids. IAM and EMNIST both ship one that the
        # corpus authors cut by writer; re-cutting it at random because the
        # distributed copy dropped the writer column would put the same hand
        # on both sides and produce a flattering number.
        hints = {r.split_hint for r in rows if r.split_hint}
        use_hints = respect_hints and {"train", "test"} <= hints

        local: dict[str, list[Sample]] = {TRAIN: [], VALIDATION: [], TEST: []}
        for row in rows:
            if use_hints and row.split_hint in ("train", "validation", "test"):
                if row.split_hint in ("test", "validation"):
                    local[TEST if row.split_hint == "test" else VALIDATION
                          ].append(row)
                    continue
                if "validation" in hints:
                    # The corpus defined its own validation side; leave it be.
                    local[TRAIN].append(row)
                    continue
                # Carve validation out of the official train side only, by
                # writer where there is one, so the official test set is left
                # exactly as the corpus defined it.
                key = row.writer if grouped else row.id
                side = VALIDATION if _stable_fraction(
                    dataset + "/" + key, salt) < validation else TRAIN
                local[side].append(row)
                continue

            key = row.writer if grouped else row.id
            fraction = _stable_fraction(dataset + "/" + key, salt)
            if fraction < test:
                local[TEST].append(row)
            elif fraction < test + validation:
                local[VALIDATION].append(row)
            else:
                local[TRAIN].append(row)

        for side, rows_ in local.items():
            out[side].extend(rows_)

        per_dataset[dataset] = {
            "samples": len(rows),
            "writers": len(writers),
            "writer_independent": grouped,
            "used_official_split": bool(use_hints),
            "counts": {side: len(rows_) for side, rows_ in local.items()},
            "note": ("the corpus's own official split, cut by its authors"
                     if use_hints else
                     "split by writer" if grouped else
                     "NO writer ids and no official split - split by sample; "
                     "validation and test numbers for this dataset are "
                     "within-writer and will flatter the model"),
        }

    report = SplitReport(
        counts={side: len(rows) for side, rows in out.items()},
        per_dataset=per_dataset,
        writer_independent=writer_independent,
    )
    return out, report


def check_leakage(splits: dict[str, list[Sample]]) -> dict[str, Any]:
    """Prove the thing the split claims: no group on two sides."""
    groups = {side: {s.group for s in rows} for side, rows in splits.items()}
    overlaps = {}
    sides = list(groups)
    for i, a in enumerate(sides):
        for b in sides[i + 1:]:
            shared = groups[a] & groups[b]
            # 'unknown' writers are by definition not a group; exclude them.
            shared = {g for g in shared if not g.endswith(":unknown")}
            if shared:
                overlaps[f"{a}|{b}"] = sorted(shared)[:20]
    return {"clean": not overlaps, "overlaps": overlaps}


def save(splits: dict[str, list[Sample]], report: SplitReport,
         path: Path | str = SPLITS_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "report": report.as_dict(),
        "leakage": check_leakage(splits),
        "ids": {side: [s.id for s in rows] for side, rows in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def load(samples: Iterable[Sample], path: Path | str = SPLITS_FILE
         ) -> dict[str, list[Sample]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    wanted = {side: set(ids) for side, ids in payload["ids"].items()}
    out: dict[str, list[Sample]] = {side: [] for side in wanted}
    for sample in samples:
        for side, ids in wanted.items():
            if sample.id in ids:
                out[side].append(sample)
                break
    return out


__all__ = ["assign", "check_leakage", "save", "load", "SplitReport",
           "TRAIN", "VALIDATION", "TEST"]
