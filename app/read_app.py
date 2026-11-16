"""Handwriting reader - upload a page, see every line read back side by side.

This view is only about reading text. No field mapping, no knowledge file: a
crop of the line on the left, what the model made of it on the right, and the
accuracy when you tell it what the line actually says.

Corrections are saved as training data, so using this view to check the model
is the same act as improving it.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from factfind.crops import (CROPS_DIR, cut_page, load_labels, merge_records,
                            save_labels)
from factfind.paths import DATA_DIR
from factfind.recognizer_train import levenshtein

PAGES_DIR = DATA_DIR / "pages"
ACCEPTED = ["jpg", "jpeg", "png", "pdf", "tif", "tiff", "bmp", "webp"]
CROP_HEIGHT = 52
CROP_MAX_WIDTH = 760


@st.cache_data(show_spinner=False, max_entries=1024)
def scaled(path: Path, mtime: float | None = None):
    from PIL import Image

    image = Image.open(path).convert("L")
    scale = CROP_HEIGHT / max(1, image.height)
    if image.width * scale > CROP_MAX_WIDTH:
        scale = CROP_MAX_WIDTH / max(1, image.width)
    return image.resize((max(16, int(image.width * scale)),
                         max(12, int(image.height * scale))))


def accuracy(predicted: str, truth: str) -> float:
    """Character accuracy, 0..1."""
    truth = truth.strip()
    if not truth:
        return 0.0
    return max(0.0, 1.0 - levenshtein(predicted.strip().lower(), truth.lower())
               / len(truth))


def main() -> None:
    st.title("🔍 Read handwriting")
    st.caption(
        "Every line the detector finds, with what the model read beside it. "
        "Type the correct text to score it — your corrections are saved and "
        "become training data."
    )

    records = load_labels()

    with st.sidebar:
        st.header("Pages")
        uploads = st.file_uploader("Upload note photos", type=ACCEPTED,
                                   accept_multiple_files=True)
        if uploads and st.button(f"Read {len(uploads)} file(s)", type="primary",
                                 use_container_width=True):
            PAGES_DIR.mkdir(parents=True, exist_ok=True)
            incoming = []
            bar = st.progress(0.0)
            for i, upload in enumerate(uploads, 1):
                target = PAGES_DIR / upload.name
                target.write_bytes(upload.getvalue())
                bar.progress((i - 1) / len(uploads), text=f"Reading {upload.name}…")
                try:
                    incoming.extend(cut_page(target))
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
            bar.empty()
            merged = merge_records(load_labels(), incoming)
            save_labels(merged)
            st.success(f"Read {len(incoming)} line(s).")
            st.rerun()

        if st.button("Clear all pages", use_container_width=True):
            save_labels([])
            st.rerun()

    # The uploader also lives here, not only in the sidebar: this is the first
    # thing anyone wants to do on this screen.
    with st.container(border=True):
        main_uploads = st.file_uploader(
            "📤 Upload handwritten pages to read",
            type=ACCEPTED, accept_multiple_files=True, key="main-upload",
        )
        if main_uploads and st.button(
                f"Read {len(main_uploads)} file(s)", type="primary",
                key="main-read", use_container_width=True):
            PAGES_DIR.mkdir(parents=True, exist_ok=True)
            incoming = []
            bar = st.progress(0.0)
            for i, upload in enumerate(main_uploads, 1):
                target = PAGES_DIR / upload.name
                target.write_bytes(upload.getvalue())
                bar.progress((i - 1) / len(main_uploads),
                             text=f"Reading {upload.name}…")
                try:
                    incoming.extend(cut_page(target))
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
            bar.empty()
            save_labels(merge_records(load_labels(), incoming))
            st.success(f"Read {len(incoming)} line(s).")
            st.rerun()

    if not records:
        st.info("Upload one or more pages above to begin.")
        return

    pages = sorted({r.page for r in records})
    page = st.selectbox("Page", pages)
    rows = [r for r in records if r.page == page]

    scored = [r for r in rows if r.text.strip()]
    if scored:
        mean = sum(accuracy(r.ocr_guess, r.text) for r in scored) / len(scored)
        exact = sum(1 for r in scored
                    if r.ocr_guess.strip().lower() == r.text.strip().lower())
        cols = st.columns(4)
        cols[0].metric("Lines found", len(rows))
        cols[1].metric("Lines scored", len(scored))
        cols[2].metric("Character accuracy", f"{mean:.0%}")
        cols[3].metric("Exact lines", f"{exact}/{len(scored)}")
    else:
        st.metric("Lines found", len(rows))
        st.caption("Type the correct text next to a line to start scoring.")

    st.divider()
    header = st.columns([4, 3, 3])
    header[0].markdown("**The line on the page**")
    header[1].markdown("**What the model read**")
    header[2].markdown("**What it actually says**")

    with st.form(f"read-{page}"):
        corrections: dict[str, str] = {}
        for record in rows:
            left, middle, right = st.columns([4, 3, 3])
            with left:
                path = record.resolved()
                if path.exists():
                    st.image(scaled(path, path.stat().st_mtime))
                else:
                    st.warning("crop missing")
            with middle:
                guess = record.ocr_guess.strip() or "(nothing read)"
                if record.text.strip():
                    acc = accuracy(record.ocr_guess, record.text)
                    colour = "#1a7f37" if acc >= 0.9 else (
                        "#9a6700" if acc >= 0.5 else "#cf222e")
                    st.markdown(
                        f"<div style='font-family:monospace;font-size:0.95rem'>"
                        f"{guess}</div>"
                        f"<div style='color:{colour};font-size:0.8rem'>"
                        f"{acc:.0%} correct</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"<div style='font-family:monospace;font-size:0.95rem'>"
                        f"{guess}</div>"
                        f"<div style='color:#888;font-size:0.8rem'>"
                        f"confidence {record.ocr_confidence:.0%}</div>",
                        unsafe_allow_html=True,
                    )
            with right:
                corrections[record.image] = st.text_input(
                    f"truth-{record.image}", value=record.text,
                    key=f"t-{record.image}", label_visibility="collapsed",
                    placeholder="type the correct text",
                )

        if st.form_submit_button("Save corrections", type="primary",
                                 use_container_width=True):
            by_image = {r.image: r for r in records}
            for image, text in corrections.items():
                if image in by_image:
                    by_image[image].text = text.strip()
            save_labels(sorted(by_image.values(),
                               key=lambda r: (r.page, r.line_index)))
            st.success("Saved. These corrections are now training data.")
            st.rerun()

    if scored:
        with st.expander("Side-by-side table"):
            st.dataframe(pd.DataFrame([
                {"model read": r.ocr_guess, "actually says": r.text,
                 "accuracy": f"{accuracy(r.ocr_guess, r.text):.0%}"}
                for r in scored
            ]), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    st.set_page_config(page_title="Read handwriting", page_icon="🔍",
                       layout="wide")
    main()
