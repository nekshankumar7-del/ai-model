"""Download handwriting fonts to train the recogniser on.

Why: the synthetic pre-training corpus is only as good as the handwriting it
renders. Windows ships a dozen script-ish fonts, most of them print rather
than joined cursive, so a model trained on them learns the wrong letter
shapes for the notes we actually get.

These come from Google Fonts, which publishes under the SIL Open Font
Licence. That matters here: the usual handwriting research corpora (IAM, CVL,
Imgur5K) are licensed for non-commercial research only, and this is a
commercial product. OFL fonts can be used commercially without restriction.

    python scripts/08_fetch_fonts.py
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import requests

from factfind.paths import DATA_DIR

FONT_DIR = DATA_DIR / "fonts"
API = "https://api.github.com/repos/google/fonts/contents/ofl/{family}"

# Joined/cursive scripts first -- those are what the real notes look like --
# then printed handwriting for variety.
FAMILIES = [
    # cursive / joined
    "dancingscript", "satisfy", "zeyada", "mrssaintdelafield", "allura",
    "greatvibes", "parisienne", "yellowtail", "marckscript", "cookie",
    "damion", "italianno", "niconne", "tangerine", "sacramento",
    "petitformalscript", "badscript", "clickerscript", "meddon",
    "lovedbytheking", "dawningofanewday", "mrdehaviland", "pinyonscript",
    "qwigley", "rougescript", "ruthie", "yesteryear", "montecarlo",
    # printed handwriting
    "caveat", "indieflower", "shadowsintolight", "patrickhand", "kalam",
    "architectsdaughter", "gloriahallelujah", "rocksalt", "homemadeapple",
    "reeniebeanie", "coveredbyyourgrace", "nothingyoucoulddo",
    "justanotherhand", "labelleaurore", "amaticsc", "caveatbrush",
    "comingsoon", "schoolbell", "swankyandmoomoo", "waitingforthesunrise",
    "handlee", "neucha", "sriracha", "itim", "gaegu", "singleday",
]


def ca_bundle() -> str | bool:
    """Use the corporate CA bundle if one was set up for pip."""
    local = Path(__file__).resolve().parents[1] / ".venv" / "corp-ca-bundle.pem"
    if local.exists():
        return str(local)
    return os.environ.get("REQUESTS_CA_BUNDLE") or True


def pick_font(entries: list[dict]) -> dict | None:
    """Prefer a static Regular; fall back to a variable font."""
    ttfs = [e for e in entries if e.get("name", "").lower().endswith(".ttf")]
    if not ttfs:
        return None
    for entry in ttfs:
        if "-Regular." in entry["name"]:
            return entry
    for entry in ttfs:
        if "[" in entry["name"]:          # variable font, e.g. Caveat[wght].ttf
            return entry
    return ttfs[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(FONT_DIR))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    verify = ca_bundle()

    session = requests.Session()
    # A generic client name: no personal or company details go to the service.
    session.headers.update({"User-Agent": "factfind-font-fetcher/1.0",
                            "Accept": "application/vnd.github+json"})

    families = FAMILIES[: args.limit] if args.limit else FAMILIES
    got, failed = [], []

    for i, family in enumerate(families, 1):
        target_existing = list(out.glob(f"{family}*.ttf"))
        if target_existing:
            got.append(target_existing[0].name)
            continue
        try:
            response = session.get(API.format(family=family), timeout=30,
                                   verify=verify)
            if response.status_code == 403:
                print("  GitHub rate limit hit; stopping early.")
                break
            response.raise_for_status()
            entry = pick_font(response.json())
            if entry is None:
                failed.append((family, "no ttf in family"))
                continue
            blob = session.get(entry["download_url"], timeout=60, verify=verify)
            blob.raise_for_status()
            name = f"{family}-{entry['name']}".replace("[", "_").replace("]", "_")
            (out / name).write_bytes(blob.content)
            got.append(name)
            print(f"  [{i}/{len(families)}] {name} "
                  f"({len(blob.content) / 1024:.0f} KB)")
        except Exception as exc:
            failed.append((family, f"{type(exc).__name__}: {exc}"))
        time.sleep(0.15)                  # be polite to the API

    (out / "_LICENCE.txt").write_text(
        "Fonts downloaded from the Google Fonts repository (github.com/google/"
        "fonts), ofl/ directory. All are published under the SIL Open Font "
        "Licence 1.1, which permits commercial use. They are used here only to "
        "render synthetic training images for a handwriting recogniser.\n",
        encoding="utf-8",
    )

    print(f"\n{len(got)} font(s) available in {out}")
    if failed:
        print(f"{len(failed)} failed:")
        for family, why in failed[:10]:
            print(f"  - {family}: {why[:70]}")
    if not got:
        print("\nNo fonts downloaded. The recogniser will fall back to the "
              "Windows fonts, which are mostly print rather than cursive.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
