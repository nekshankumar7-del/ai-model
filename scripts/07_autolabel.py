"""Attach ground-truth transcriptions to detected crops, automatically.

The detector finds the lines on a page reliably; only recognition fails. So
rather than transcribing crop by crop in the UI, you transcribe the page once
(in data/transcriptions.json, top to bottom) and this aligns those lines to the
detected crops.

Alignment is Needleman-Wunsch over the similarity between each crop's OCR
guess and each transcription line. That matters because the two sequences do
not match one-to-one: the detector finds extra boxes (page furniture, stray
marks) and occasionally merges or splits a line. A positional zip would
silently mislabel every crop after the first discrepancy, which is worse than
no labels at all -- a mislabelled crop actively teaches the recogniser to read
wrongly.

    python scripts/07_autolabel.py              # align and write labels
    python scripts/07_autolabel.py --dry-run    # show the alignment only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from rapidfuzz import fuzz

from factfind.crops import (CropRecord, cut_page, load_labels, merge_records,
                            save_labels, stats)
from factfind.paths import DATA_DIR

TRANSCRIPTS = DATA_DIR / "transcriptions.json"
# Below this similarity we refuse to pair a crop with a transcription line.
# An unlabelled crop costs nothing; a wrongly labelled one poisons training.
MIN_PAIR_SIMILARITY = 25.0
GAP_PENALTY = -12.0


def similarity(guess: str, truth: str) -> float:
    """How plausibly this crop's OCR guess corresponds to this line."""
    if not guess.strip():
        # No guess at all still aligns on position; score it neutrally low.
        return 10.0
    return max(
        fuzz.ratio(guess.lower(), truth.lower()),
        fuzz.partial_ratio(guess.lower(), truth.lower()) * 0.85,
        fuzz.token_set_ratio(guess.lower(), truth.lower()) * 0.9,
    )


def align(crops: list[CropRecord], lines: list[str]
          ) -> list[tuple[CropRecord, str | None, float]]:
    """Needleman-Wunsch alignment preserving reading order."""
    n, m = len(crops), len(lines)
    score = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP_PENALTY
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP_PENALTY

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diagonal = score[i - 1][j - 1] + similarity(crops[i - 1].ocr_guess,
                                                        lines[j - 1])
            score[i][j] = max(diagonal, score[i - 1][j] + GAP_PENALTY,
                              score[i][j - 1] + GAP_PENALTY)

    pairs: list[tuple[CropRecord, str | None, float]] = []
    i, j = n, m
    while i > 0:
        if j > 0:
            diagonal = score[i - 1][j - 1] + similarity(crops[i - 1].ocr_guess,
                                                        lines[j - 1])
            if abs(score[i][j] - diagonal) < 1e-9:
                sim = similarity(crops[i - 1].ocr_guess, lines[j - 1])
                pairs.append((crops[i - 1], lines[j - 1], sim))
                i, j = i - 1, j - 1
                continue
        if abs(score[i][j] - (score[i - 1][j] + GAP_PENALTY)) < 1e-9:
            pairs.append((crops[i - 1], None, 0.0))
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


def main() -> int:
    # Handwritten notes contain arrows, dashes and currency signs that the
    # Windows console's default codepage cannot encode.
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default="knowledge",
                    help="folder holding the page photos")
    ap.add_argument("--transcripts", default=str(TRANSCRIPTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--min-similarity", type=float, default=MIN_PAIR_SIMILARITY)
    args = ap.parse_args()

    path = Path(args.transcripts)
    if not path.exists():
        print(f"STOPPING. No transcriptions at {path}.")
        return 2
    transcripts = {k: v for k, v in
                   json.loads(path.read_text(encoding="utf-8")).items()
                   if not k.startswith("_")}

    images = Path(args.images)
    suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    by_stem = {p.stem: p for p in sorted(images.rglob("*"))
               if p.suffix.lower() in suffixes}

    existing = load_labels()
    all_records: list[CropRecord] = []
    total_paired = total_skipped = 0

    for stem, lines in transcripts.items():
        page = by_stem.get(stem)
        if page is None:
            print(f"  ! no image found for {stem!r} in {images}")
            continue

        # One writer per page here; the pages are different hands and the
        # split must be able to hold a whole hand out.
        crops = cut_page(page, writer=f"writer-{stem}".replace(" ", "-"))
        pairs = align(crops, lines)

        print(f"\n{stem}: {len(crops)} crops, {len(lines)} transcribed lines")
        paired = skipped = 0
        for record, truth, sim in pairs:
            if truth is not None and sim >= args.min_similarity:
                record.text = truth
                paired += 1
                mark = "ok "
            else:
                record.text = ""
                skipped += 1
                mark = "-- "
            print(f"  {mark} sim={sim:5.1f}  guess={record.ocr_guess[:26]!r:<30}"
                  f" -> {str(truth)[:34]!r}")
        total_paired += paired
        total_skipped += skipped
        print(f"  paired {paired}, left blank {skipped}")
        all_records.extend(record for record, _t, _s in pairs)

    print(f"\ntotal: paired {total_paired}, blank {total_skipped}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    # Never clobber a human transcription with an automatic one.
    merged = merge_records(existing, [r for r in all_records if r.text])
    by_image = {r.image: r for r in merged}
    for record in all_records:
        if record.text and not by_image.get(record.image, record).text:
            by_image[record.image].text = record.text
    for record in all_records:
        if record.image not in by_image:
            by_image[record.image] = record
    final = sorted(by_image.values(), key=lambda r: (r.page, r.line_index))
    save_labels(final)

    print(f"\nwrote {len(final)} crop record(s)")
    print("stats:", stats(final))
    print("\nNow fine-tune:  python scripts/06_train_recognizer.py --resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
