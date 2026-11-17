"""Turn extracted datasets into validated manifests.

For each dataset: run its adapter, drop what cannot be trained on (empty
labels, over-long labels, unreadable files, blank crops, near-duplicates),
write ``data/processed/manifests/<id>.jsonl``, and record what was dropped.

The dropped counts are reported rather than swallowed. A corpus that loses
40% of its rows to unreadable files is telling you the download is damaged,
and that is worth seeing before a training run, not after.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .adapters import get_adapter
from .manifest import (Sample, corpus_stats, read_manifest, validate,
                       write_manifest)
from .paths import METADATA_DIR, extracted_dir, manifest_path
from .registry import DatasetSpec, get

BUILD_REPORT = METADATA_DIR / "build_report.json"


def build_dataset(spec: DatasetSpec, *, limit: int | None = None,
                  check_images: bool = True, dedupe: bool = True,
                  max_text_len: int = 96) -> dict[str, Any]:
    root = extracted_dir(spec.id)
    started = time.time()

    # Some adapters read the downloaded archive directly rather than an
    # extracted tree (SD19 is 814k files, which is not worth unpacking), so
    # raw/ counts as present just as much as extracted/ does.
    from .paths import raw_dir
    has_data = any(root.iterdir()) or any(raw_dir(spec.id).iterdir())
    if not has_data:
        return {"dataset": spec.id, "status": "not-extracted", "samples": 0,
                "hint": (spec.manual_instructions or
                         "run scripts/10_fetch_datasets.py first")}

    adapter = get_adapter(spec.adapter)
    try:
        produced = list(adapter(spec, root, limit=limit))
    except (FileNotFoundError, NotImplementedError) as exc:
        return {"dataset": spec.id, "status": "adapter-failed", "samples": 0,
                "error": str(exc)}

    kept, report = validate(produced, check_images=check_images,
                            dedupe=dedupe, max_text_len=max_text_len)
    written = write_manifest(spec.id, kept)
    stats = corpus_stats(kept)

    return {
        "dataset": spec.id,
        "name": spec.name,
        "status": "built" if kept else "empty",
        "level": spec.level,
        "scripts": list(spec.scripts),
        "samples": len(kept),
        "manifest": written["manifest"],
        "validation": report.as_dict(),
        "stats": {k: v for k, v in stats.items() if k != "most_common"},
        "seconds": round(time.time() - started, 1),
    }


def build(dataset_ids: Iterable[str], **kwargs: Any) -> dict[str, Any]:
    results = []
    for dataset_id in dataset_ids:
        spec = get(dataset_id)
        print(f"\n[{spec.id}] {spec.name}")
        result = build_dataset(spec, **kwargs)
        results.append(result)
        if result["status"] == "built":
            validation = result["validation"]
            print(f"  {result['samples']:,} samples from "
                  f"{validation['total']:,} candidates "
                  f"({result['stats']['writers']} writer groups, "
                  f"{result['stats']['distinct_characters']} distinct characters)")
            dropped = {k: v for k, v in validation.items()
                       if k in ("empty_label", "too_long", "unreadable",
                                "degenerate_image", "duplicates") and v}
            if dropped:
                print("  dropped: " + ", ".join(f"{k}={v}" for k, v in dropped.items()))
        else:
            print(f"  {result['status']}: {result.get('error') or result.get('hint', '')}")

    payload = {"datasets": results,
               "total_samples": sum(r["samples"] for r in results)}
    BUILD_REPORT.parent.mkdir(parents=True, exist_ok=True)
    BUILD_REPORT.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return payload


def available_manifests() -> list[str]:
    from .paths import MANIFEST_DIR
    return sorted(p.stem for p in MANIFEST_DIR.glob("*.jsonl"))


def load_samples(dataset_ids: Iterable[str], level: str | None = None
                 ) -> list[Sample]:
    """Load manifests, refusing to mix incompatible levels by accident."""
    out: list[Sample] = []
    levels: set[str] = set()
    for dataset_id in dataset_ids:
        rows = list(read_manifest(dataset_id))
        if not rows:
            continue
        found = {r.level for r in rows}
        if level is not None:
            rows = [r for r in rows if r.level == level]
            if not rows:
                continue
        levels |= {r.level for r in rows}
        out.extend(rows)
    if level is None and len(levels) > 1:
        raise ValueError(
            "refusing to mix levels " + ", ".join(sorted(levels)) + " in one "
            "training set. Character crops and word/line crops need different "
            "configurations - pass --level, or use --compose-words to turn "
            "character data into word images.")
    return out


__all__ = ["build", "build_dataset", "load_samples", "available_manifests",
           "BUILD_REPORT"]
