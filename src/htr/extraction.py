"""Archive extraction, with the path checks that downloaded archives need.

Everything under data/raw/ came off the internet. A zip or tar entry is free
to name ``../../../etc/passwd`` or point a symlink anywhere it likes, so every
member is resolved against the destination before anything is written. An
archive that tries it is rejected whole, not silently skipped member by
member.
"""
from __future__ import annotations

import gzip
import json
import re
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Iterable

from .paths import extracted_dir, raw_dir

MARKER = ".extracted.json"


class UnsafeArchive(RuntimeError):
    """An archive member resolved outside the destination directory."""


def _check_member_name(name: str) -> None:
    """Reject a member name that could write outside the destination.

    Purely lexical on purpose. Resolving each member against the filesystem
    means two syscalls per entry, and NIST SD19 has roughly 400,000 of them on
    a OneDrive-synced disk - that alone took longer than the download. It is
    also the stricter check: ``resolve()`` follows symlinks that an archive
    may itself have just created, whereas this never touches the disk.
    """
    if not name or name in (".", "/"):
        return
    normalised = name.replace("\\", "/")
    if normalised.startswith("/") or re.match(r"^[A-Za-z]:", normalised):
        raise UnsafeArchive(f"member {name!r} is an absolute path")
    parts = [p for p in normalised.split("/") if p not in ("", ".")]
    depth = 0
    for part in parts:
        if part == "..":
            depth -= 1
            if depth < 0:
                raise UnsafeArchive(f"member {name!r} escapes the destination")
        else:
            depth += 1


def extract_zip(archive: Path, destination: Path) -> int:
    count = 0
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for info in members:
            _check_member_name(info.filename)
        total = len(members)
        for info in members:
            zf.extract(info, destination)
            count += 1
            if total > 50_000 and count % 50_000 == 0:
                print(f"      {count:,}/{total:,} members", flush=True)
    return count


def extract_tar(archive: Path, destination: Path) -> int:
    count = 0
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for member in members:
            if member.islnk() or member.issym():
                raise UnsafeArchive(f"member {member.name!r} is a link")
            _check_member_name(member.name)
        # data filter is the hardened extraction mode added in Python 3.12;
        # the explicit checks above cover 3.11.
        try:
            tf.extractall(destination, filter="data")
        except TypeError:
            tf.extractall(destination)
        count = len(members)
    return count


def extract_gz(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    out = destination / archive.stem
    with gzip.open(archive, "rb") as src, out.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return 1


def extract_one(archive: Path, destination: Path) -> int:
    """Unpack a single archive. Returns the number of members written."""
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(archive.suffixes[-2:]).lower()
    if zipfile.is_zipfile(archive):
        return extract_zip(archive, destination)
    if tarfile.is_tarfile(archive):
        return extract_tar(archive, destination)
    if suffixes.endswith(".gz"):
        return extract_gz(archive, destination)
    if archive.suffix.lower() == ".rar":
        raise RuntimeError(
            f"{archive.name} is a RAR archive; unpack it by hand into "
            f"{destination} (Python has no stdlib RAR reader)")
    raise RuntimeError(f"unrecognised archive format: {archive.name}")


def extract_nested(root: Path, depth: int = 2) -> int:
    """Unpack archives that were themselves inside an archive.

    EMNIST ships a zip of gzipped IDX files; several IIIT releases ship a tar
    inside a zip. One pass over the tree per level of nesting.
    """
    total = 0
    for _ in range(depth):
        found = False
        for path in list(root.rglob("*")):
            if not path.is_file():
                continue
            low = path.name.lower()
            if low.endswith((".zip", ".tar", ".tar.gz", ".tgz")):
                marker = path.with_suffix(path.suffix + ".done")
                if marker.exists():
                    continue
                try:
                    total += extract_one(path, path.parent / (path.stem + "_x"))
                    marker.touch()
                    found = True
                except (UnsafeArchive, RuntimeError) as exc:
                    print(f"    skipping {path.name}: {exc}")
            elif low.endswith(".gz") and not low.endswith(".tar.gz"):
                out = path.with_suffix("")
                if out.exists():
                    continue
                try:
                    with gzip.open(path, "rb") as src, out.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    total += 1
                    found = True
                except OSError as exc:
                    print(f"    skipping {path.name}: {exc}")
        if not found:
            break
    return total


def extract_dataset(dataset_id: str, *, force: bool = False,
                    nested: bool = True,
                    only: Iterable[str] | None = None) -> dict[str, Any]:
    """Unpack archives from data/raw/<id>/ into data/extracted/<id>/.

    ``only`` restricts it to named files. That matters: NIST SD19's
    by_write.zip is 830,000 members that take longer to unpack than to
    download, and it is an optional archive that the adapter does not read.
    Unpacking whatever happens to be in the directory would do it anyway.
    """
    source = raw_dir(dataset_id)
    destination = extracted_dir(dataset_id)
    marker = destination / MARKER

    wanted = set(only) if only is not None else None
    archives = sorted(p for p in source.iterdir()
                      if p.is_file() and not p.name.endswith(".part")
                      and (wanted is None or p.name in wanted))
    if not archives:
        return {"dataset": dataset_id, "status": "no-archives", "members": 0}

    if marker.exists() and not force:
        state = json.loads(marker.read_text(encoding="utf-8"))
        if state.get("archives") == [a.name for a in archives]:
            return {**state, "status": "cached"}

    members = 0
    for archive in archives:
        print(f"    unpacking {archive.name}")
        members += extract_one(archive, destination)
    if nested:
        members += extract_nested(destination)

    state = {"dataset": dataset_id, "archives": [a.name for a in archives],
             "members": members, "status": "extracted"}
    marker.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def extract_all(dataset_ids: Iterable[str], *, force: bool = False,
                only: dict[str, Iterable[str]] | None = None
                ) -> list[dict[str, Any]]:
    results = []
    for dataset_id in dataset_ids:
        print(f"\n[{dataset_id}] extracting")
        try:
            result = extract_dataset(
                dataset_id, force=force,
                only=(only or {}).get(dataset_id) if only else None)
        except (UnsafeArchive, RuntimeError) as exc:
            result = {"dataset": dataset_id, "status": "error", "error": str(exc)}
            print(f"    ERROR {exc}")
        print(f"    {result['status']}: {result.get('members', 0)} members")
        results.append(result)
    return results


__all__ = ["extract_dataset", "extract_all", "extract_one", "UnsafeArchive"]
