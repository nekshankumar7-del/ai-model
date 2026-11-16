"""Crop labelling tool - builds the training set for the handwriting recogniser.

    streamlit run app/label_app.py --server.address localhost

Point it at a folder of note photos. It detects the lines, shows you each one
cropped, and you type what it says. The detector's own (usually wrong) reading
is offered as a starting point, so most lines are a quick correction rather
than typing from scratch.

Work is saved to data/crops/labels.jsonl after every page, so you can stop and
come back. Nothing leaves this machine.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from factfind.crops import (CROPS_DIR, CropRecord, cut_directory, cut_page,
                            load_labels, merge_records, save_labels,
                            split_by_writer, stats, transcribed)
from factfind.paths import DATA_DIR

# Uploaded pages are copied here so the crops keep resolving between sessions.
PAGES_DIR = DATA_DIR / "pages"
ACCEPTED = ["jpg", "jpeg", "png", "pdf", "tif", "tiff", "bmp", "webp"]

# Only set the page config when this file is the entry point. When it is
# imported as a page of the single combined app, the host has already done it
# and calling it twice is an error.
if __name__ == "__main__":
    st.set_page_config(page_title="Label handwriting crops", page_icon="✍️",
                       layout="wide")

# Enough transcribed lines for fine-tuning to be worth running at all.
TARGET_LINES = 2000
# Lines shown per screen. Small enough that every crop and its box fit
# together without scrolling, which is what makes this quick to work through.
BATCH_SIZE = 8
# Crops are rendered at a fixed height so one line of handwriting stays one
# compact row, whatever resolution the photo was taken at.
CROP_HEIGHT = 56
CROP_MAX_WIDTH = 1000


@st.cache_data(show_spinner=False, max_entries=512)
def scaled_crop(path: Path, mtime: float | None = None):
    """Load a crop scaled to a consistent row height."""
    from PIL import Image

    image = Image.open(path).convert("L")
    scale = CROP_HEIGHT / max(1, image.height)
    if image.width * scale > CROP_MAX_WIDTH:
        # A very long line scales by width instead, so the aspect ratio is
        # never distorted -- squashed handwriting is harder to read, not easier.
        scale = CROP_MAX_WIDTH / max(1, image.width)
    width = max(16, int(image.width * scale))
    height = max(12, int(image.height * scale))
    return image.resize((width, height), Image.LANCZOS)


def load_state() -> list[CropRecord]:
    if "records" not in st.session_state:
        st.session_state.records = load_labels()
    return st.session_state.records


def persist() -> None:
    save_labels(st.session_state.records)


def main() -> None:
    st.title("✍️ Label handwriting crops")
    st.caption(
        "Type what each line actually says. These corrections are what the "
        "recogniser learns from - they are the difference between reading "
        "your advisers' handwriting and guessing at it."
    )

    records = load_state()
    current = stats(records)

    with st.sidebar:
        st.header("Add pages")

        uploads = st.file_uploader(
            "Upload note photos", type=ACCEPTED, accept_multiple_files=True,
            help="JPG, PNG or PDF. Files are copied into data/pages/ so the "
                 "crops keep working after you close the browser.",
        )
        if uploads and st.button(f"Detect lines in {len(uploads)} file(s)",
                                 type="primary", use_container_width=True):
            PAGES_DIR.mkdir(parents=True, exist_ok=True)
            incoming = []
            progress = st.progress(0.0)
            for i, upload in enumerate(uploads, 1):
                target = PAGES_DIR / upload.name
                target.write_bytes(upload.getvalue())
                progress.progress((i - 1) / len(uploads),
                                  text=f"Detecting lines in {upload.name}…")
                try:
                    incoming.extend(cut_page(target))
                except Exception as exc:
                    st.error(f"{upload.name}: {exc}")
            progress.empty()
            before = len(st.session_state.records)
            st.session_state.records = merge_records(
                st.session_state.records, incoming)
            persist()
            st.success(f"Added {len(st.session_state.records) - before} new crop(s).")
            st.rerun()

        with st.expander("Or use a folder on disk"):
            folder = st.text_input("Folder of note photos", value="knowledge")
            if st.button("Detect lines in this folder", use_container_width=True):
                path = Path(folder)
                if not path.exists():
                    st.error(f"No such folder: {path}")
                else:
                    with st.spinner("Detecting lines…"):
                        incoming = cut_directory(path)
                    before = len(st.session_state.records)
                    st.session_state.records = merge_records(
                        st.session_state.records, incoming)
                    persist()
                    added = len(st.session_state.records) - before
                    st.success(f"Added {added} new crop(s) "
                               f"from {len(incoming)} detected.")
                    st.rerun()

        st.divider()
        st.header("Progress")
        st.metric("Transcribed", current["n_transcribed"])
        st.metric("Remaining", current["n_remaining"])
        st.progress(min(1.0, current["n_transcribed"] / TARGET_LINES),
                    text=f"{current['n_transcribed']} of ~{TARGET_LINES} lines")
        st.write(f"- pages: **{current['n_pages']}**")
        st.write(f"- writers: **{current['n_writers']}**")
        st.write(f"- characters: **{current['n_characters']:,}**")

        if current["n_transcribed"] and current["n_writers"] < 2:
            st.warning(
                "Only one writer so far. Validation cannot show the model "
                "generalises to a new hand until there are at least two, and "
                "ideally eight."
            )

        st.divider()
        if st.button("Export training set", use_container_width=True):
            done = transcribed(st.session_state.records)
            if len(done) < 20:
                st.error("Too few transcriptions to train on yet.")
            else:
                train, val = split_by_writer(done)
                st.success(f"Ready: {len(train)} train / {len(val)} validation. "
                           f"Run scripts/06_train_recognizer.py")

    if not records:
        st.info(
            "No crops yet. Put your note photos in a folder, enter it in the "
            "sidebar and press **Detect lines in this folder**."
        )
        return

    # ---- pick a page to work on
    pages = sorted({r.page for r in records})
    done_by_page = {
        p: sum(1 for r in records if r.page == p and r.text.strip())
        for p in pages
    }
    total_by_page = {p: sum(1 for r in records if r.page == p) for p in pages}

    def page_label(p: str) -> str:
        return f"{p}  ({done_by_page[p]}/{total_by_page[p]} done)"

    unfinished = [p for p in pages if done_by_page[p] < total_by_page[p]]
    default_index = pages.index(unfinished[0]) if unfinished else 0
    page = st.selectbox("Page", pages, index=default_index, format_func=page_label)

    hide_done = st.checkbox("Hide lines I've already done", value=True)

    page_records = [r for r in records if r.page == page]
    writer_value = page_records[0].writer if page_records else "unknown"
    writer = st.text_input(
        "Who wrote this page?", value=writer_value,
        placeholder="e.g. adviser-01",
        help="A stable id per adviser. Splits group on this so one person's "
             "handwriting never appears in both training and validation.",
    )
    if writer != writer_value:
        for record in page_records:
            record.writer = writer.strip() or "unknown"
        persist()
    if writer.strip() in ("", "unknown"):
        st.warning(
            "Set a writer id before labelling this page. Without it the "
            "training split cannot keep one adviser's handwriting out of "
            "validation, and the accuracy numbers will be optimistic.",
            icon="⚠️",
        )

    visible = [r for r in page_records if not (hide_done and r.text.strip())]
    if not visible:
        st.success("Every line on this page is transcribed. Pick another page.")
        return

    # Paginate. Rendering 130 image+input pairs at once is both unusable and
    # slow; a short batch keeps every line on one screen.
    offset_key = f"offset-{page}"
    offset = st.session_state.get(offset_key, 0)
    offset = max(0, min(offset, max(0, len(visible) - 1)))
    batch = visible[offset: offset + BATCH_SIZE]

    st.caption(
        f"Lines {offset + 1}-{offset + len(batch)} of {len(visible)} to go on "
        f"this page. Tab moves to the next box. Leave a box empty to skip a "
        f"line that isn't readable text."
    )

    with st.form(f"page-{page}-{offset}"):
        values: dict[str, str] = {}
        for record in batch:
            crop_path = record.resolved()
            if crop_path.exists():
                st.image(scaled_crop(crop_path))
            else:
                st.warning(f"missing crop: {record.image}")
            values[record.image] = st.text_input(
                f"line {record.line_index}",
                value=record.text or record.ocr_guess,
                key=f"in-{record.image}-{offset}",
                label_visibility="collapsed",
                placeholder="type what this line says",
            )

        saved = st.form_submit_button("Save and continue", type="primary",
                                      use_container_width=True)

    if saved:
        by_image = {r.image: r for r in st.session_state.records}
        for image, text in values.items():
            if image in by_image:
                by_image[image].text = text.strip()
        persist()
        # With finished lines hidden the queue shrinks under us, so staying
        # put advances to the next unlabelled batch automatically.
        st.session_state[offset_key] = 0 if hide_done else offset + BATCH_SIZE
        st.rerun()

    back, forward = st.columns(2)
    if back.button("Previous", disabled=offset == 0, use_container_width=True):
        st.session_state[offset_key] = max(0, offset - BATCH_SIZE)
        st.rerun()
    if forward.button("Skip ahead", disabled=offset + BATCH_SIZE >= len(visible),
                      use_container_width=True):
        st.session_state[offset_key] = offset + BATCH_SIZE
        st.rerun()

    with st.expander("All transcriptions so far"):
        done = transcribed(st.session_state.records)
        if done:
            st.dataframe(
                pd.DataFrame([
                    {"page": r.page, "writer": r.writer, "line": r.line_index,
                     "text": r.text, "detector read": r.ocr_guess}
                    for r in done
                ]),
                use_container_width=True, hide_index=True,
            )
        else:
            st.write("Nothing transcribed yet.")


if __name__ == "__main__":
    main()
