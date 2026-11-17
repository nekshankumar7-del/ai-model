"""Resumable downloads, checksums and the dataset inventory.

These archives are 0.5-6 GB each and this machine sits behind a corporate TLS
proxy. A download that cannot resume is a download that never finishes, so
every transfer writes to ``<name>.part`` and continues with a Range request on
the next run.

Nothing here logs in, submits a form, or fetches from a mirror. Datasets whose
terms require a human to agree are reported, not fetched.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

from .paths import INVENTORY_FILE, ca_bundle, ensure_dirs, raw_dir
from .registry import DatasetSpec, Resource, summary_rows

CHUNK = 1 << 20          # 1 MiB
USER_AGENT = "htr-dataset-fetcher/1.0 (research; python-requests)"
RETRIES = 4


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------

def load_inventory() -> dict[str, Any]:
    if INVENTORY_FILE.exists():
        try:
            return json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"updated": None, "datasets": {}}


def save_inventory(inventory: dict[str, Any]) -> Path:
    ensure_dirs()
    inventory["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    INVENTORY_FILE.write_text(json.dumps(inventory, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    return INVENTORY_FILE


def record(inventory: dict[str, Any], spec: DatasetSpec, **fields: Any) -> dict:
    entry = inventory["datasets"].setdefault(spec.id, {})
    entry.update({
        "name": spec.name,
        "homepage": spec.homepage,
        "licence": spec.licence,
        "licence_url": spec.licence_url,
        "access": spec.access,
        "level": spec.level,
        "scripts": list(spec.scripts),
        "languages": list(spec.languages),
        "commercial_use": spec.commercial_use,
        "approx_samples": spec.approx_samples,
        "notes": spec.notes,
    })
    entry.update(fields)
    return entry


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def sha256_file(path: Path, progress: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK)
            if not block:
                break
            digest.update(block)
            if progress:
                progress(len(block))
    return digest.hexdigest()


# --------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def download_resource(resource: Resource, target_dir: Path, *,
                      verify_tls: str | bool | None = None,
                      timeout: int = 60, quiet: bool = False) -> dict[str, Any]:
    """Fetch one file, resuming a partial download if one is present."""
    verify = ca_bundle() if verify_tls is None else verify_tls
    target = target_dir / resource.filename
    partial = target.with_suffix(target.suffix + ".part")

    if target.exists():
        size = target.stat().st_size
        if resource.bytes_expected and size != resource.bytes_expected:
            if not quiet:
                print(f"    {resource.filename}: size {size} != expected "
                      f"{resource.bytes_expected}, re-downloading")
            target.unlink()
        else:
            if not quiet:
                print(f"    {resource.filename}: already present ({_human(size)})")
            return {"url": resource.url, "file": str(target), "bytes": size,
                    "status": "cached"}

    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        have = partial.stat().st_size if partial.exists() else 0
        if have:
            headers["Range"] = f"bytes={have}-"
        else:
            headers.pop("Range", None)
        try:
            with requests.get(resource.url, stream=True, timeout=timeout,
                              headers=headers, verify=verify,
                              allow_redirects=True) as response:
                if have and response.status_code == 200:
                    # The server ignored the Range header; start over rather
                    # than concatenating a full body onto a partial file.
                    have = 0
                    partial.unlink(missing_ok=True)
                elif have and response.status_code not in (206, 416):
                    response.raise_for_status()
                if response.status_code == 416:
                    partial.replace(target)
                    size = target.stat().st_size
                    return {"url": resource.url, "file": str(target),
                            "bytes": size, "status": "resumed-complete"}
                response.raise_for_status()

                declared = response.headers.get("Content-Length")
                total = (int(declared) + have) if declared else resource.bytes_expected
                mode = "ab" if have else "wb"
                done = have
                started = time.time()
                last_print = 0.0
                with partial.open(mode) as handle:
                    for block in response.iter_content(CHUNK):
                        if not block:
                            continue
                        handle.write(block)
                        done += len(block)
                        now = time.time()
                        if not quiet and now - last_print > 2.0:
                            last_print = now
                            rate = done / max(1e-6, now - started)
                            pct = f"{100 * done / total:5.1f}% " if total else ""
                            print(f"      {resource.filename} {pct}"
                                  f"{_human(done)} @ {_human(rate)}/s",
                                  flush=True)
            partial.replace(target)
            size = target.stat().st_size
            if resource.bytes_expected and size != resource.bytes_expected:
                raise IOError(f"size mismatch: got {size}, "
                              f"expected {resource.bytes_expected}")
            if not quiet:
                print(f"    {resource.filename}: done ({_human(size)})")
            return {"url": resource.url, "file": str(target), "bytes": size,
                    "status": "downloaded"}
        except Exception as exc:                       # noqa: BLE001
            last_error = exc
            if not quiet:
                print(f"    {resource.filename}: attempt {attempt}/{RETRIES} "
                      f"failed ({type(exc).__name__}: {str(exc)[:120]})")
            time.sleep(min(30, 2 ** attempt))

    return {"url": resource.url, "file": None, "bytes": 0, "status": "failed",
            "error": f"{type(last_error).__name__}: {str(last_error)[:300]}"}


def fetch(specs: Iterable[DatasetSpec], *, include_optional: bool = False,
          checksum: bool = True, quiet: bool = False,
          max_bytes: int | None = None) -> dict[str, Any]:
    """Download everything that may legitimately be downloaded.

    Returns the inventory. Datasets whose access is manual or unavailable are
    recorded with their instructions and skipped - they are not failures.
    """
    ensure_dirs()
    inventory = load_inventory()
    budget = max_bytes

    for spec in specs:
        print(f"\n[{spec.id}] {spec.name}")
        print(f"  licence: {spec.licence}")
        if spec.access != "auto":
            print(f"  access : {spec.access.upper()} - not downloaded")
            if spec.manual_instructions:
                for line in spec.manual_instructions.splitlines():
                    print(f"    {line}")
            record(inventory, spec, status=spec.access,
                   instructions=spec.manual_instructions, files=[])
            continue

        if budget is not None and spec.total_bytes > budget:
            print(f"  skipped: {_human(spec.total_bytes)} exceeds the "
                  f"remaining budget of {_human(budget)}")
            record(inventory, spec, status="skipped-budget", files=[])
            continue

        target = raw_dir(spec.id)
        files: list[dict[str, Any]] = []
        ok = True
        for resource in spec.resources:
            if resource.optional and not include_optional:
                continue
            info = download_resource(resource, target, quiet=quiet)
            if info["status"] == "failed":
                ok = False
            elif checksum and info["file"]:
                path = Path(info["file"])
                digest = sha256_file(path)
                if resource.sha256 and digest != resource.sha256:
                    info["status"] = "checksum-mismatch"
                    info["sha256"] = digest
                    ok = False
                    print(f"    {resource.filename}: CHECKSUM MISMATCH")
                else:
                    info["sha256"] = digest
            files.append(info)

        downloaded = sum(f["bytes"] for f in files)
        if budget is not None:
            budget = max(0, budget - downloaded)
        record(inventory, spec, status="downloaded" if ok else "partial",
               files=files, bytes_on_disk=downloaded)

    inventory["registry"] = summary_rows()
    save_inventory(inventory)
    print(f"\ninventory written to {INVENTORY_FILE}")
    return inventory


__all__ = ["fetch", "download_resource", "sha256_file", "load_inventory",
           "save_inventory"]
