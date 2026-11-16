"""On-disk cache for the expensive step: OCR.

Everything downstream of OCR -- segmentation, matching, assignment,
thresholding -- is fast pure Python, so caching exactly here lets training,
re-thresholding and error analysis re-run in seconds instead of re-reading
every page.

The cache key includes the image bytes, so editing or replacing a page
invalidates its entry automatically.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .ocr import TextLine, Word
from .paths import DATA_DIR
from .preprocess import PageQuality

CACHE_DIR = DATA_DIR / "cache" / "ocr"


def _key(image_path: Path | str, engine: str) -> str:
    image_path = Path(image_path)
    digest = hashlib.sha1()
    digest.update(str(engine).encode())
    try:
        digest.update(image_path.read_bytes())
    except OSError:
        digest.update(str(image_path).encode())
    return f"{image_path.stem}-{digest.hexdigest()[:16]}"


def path_for(image_path: Path | str, engine: str = "rapidocr") -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{_key(image_path, engine)}.json"


def load(image_path: Path | str, engine: str = "rapidocr"
         ) -> list[tuple[list[TextLine], dict[str, Any]]] | None:
    """Return per-page (lines, quality) or None if not cached."""
    path = path_for(image_path, engine)
    if not path.exists():
        return None
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    out = []
    for page in blob["pages"]:
        lines = [
            TextLine(
                text=ln["text"], box=tuple(ln["box"]), confidence=ln["confidence"],
                words=[Word(text=w["text"], box=tuple(w["box"]),
                            confidence=w["confidence"]) for w in ln.get("words", [])],
                page_width=ln["page_width"], page_height=ln["page_height"],
                line_index=ln["line_index"], page_index=ln["page_index"],
            )
            for ln in page["lines"]
        ]
        out.append((lines, page["quality"]))
    return out


def store(image_path: Path | str, pages: list[tuple[list[TextLine], PageQuality]],
          engine: str = "rapidocr") -> Path:
    payload = {
        "image": str(image_path),
        "engine": engine,
        "pages": [
            {
                "quality": quality.as_dict() if isinstance(quality, PageQuality) else dict(quality),
                "lines": [
                    {
                        "text": ln.text, "box": list(ln.box),
                        "confidence": ln.confidence,
                        "words": [{"text": w.text, "box": list(w.box),
                                   "confidence": w.confidence} for w in (ln.words or [])],
                        "page_width": ln.page_width, "page_height": ln.page_height,
                        "line_index": ln.line_index, "page_index": ln.page_index,
                    }
                    for ln in lines
                ],
            }
            for lines, quality in pages
        ],
    }
    path = path_for(image_path, engine)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def clear() -> int:
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for p in CACHE_DIR.glob("*.json"):
        p.unlink()
        n += 1
    return n


def size() -> dict[str, Any]:
    if not CACHE_DIR.exists():
        return {"entries": 0, "bytes": 0}
    files = list(CACHE_DIR.glob("*.json"))
    return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}


__all__ = ["load", "store", "clear", "size", "path_for", "CACHE_DIR"]
