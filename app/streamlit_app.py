"""Fact-find extraction - testing front end.

Upload one or many photos/PDFs of client notes and see exactly what the
pipeline would fill, what it refuses to fill, and why.

    streamlit run app/streamlit_app.py

Nothing is uploaded anywhere: the app runs the same Python pipeline locally
that the Angular app runs in the browser.
"""
from __future__ import annotations

import io
import json
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
# The labelling tool is imported as a page of this app, so its folder must be
# importable however Streamlit was invoked.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import streamlit as st

from factfind.extract import (STATUS_FILLED, STATUS_MISSING, STATUS_UNSURE,
                              ExtractionResult, load_extractor)
from factfind.knowledge import KnowledgeError, load_knowledge
from factfind.labels import flatten_labels
from factfind.ocr import engine_report
from factfind.preprocess import UPLOAD_TYPES, load_pages_from_bytes

st.set_page_config(page_title="Fact-find extraction", page_icon="📄",
                   layout="wide", initial_sidebar_state="expanded")

ACCEPTED = UPLOAD_TYPES

STATUS_STYLE = {
    STATUS_FILLED: ("✅", "filled"),
    STATUS_UNSURE: ("⚠️", "unsure"),
    STATUS_MISSING: ("—", "missing"),
}


# --------------------------------------------------------------------------
# Loading (cached across reruns)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_knowledge():
    return load_knowledge()


@st.cache_resource(show_spinner=False)
def get_extractor(use_models: bool, threshold_shift: float):
    extractor = load_extractor(use_models=use_models)
    if threshold_shift:
        shifted = {}
        for key, value in (extractor.thresholds or {}).items():
            shifted[key] = float(min(1.01, max(0.0, value + threshold_shift)))
        extractor.thresholds = shifted
    return extractor


def pipeline_version() -> str:
    """A fingerprint of the pipeline code and the trained models.

    Results are cached by file content, which is what makes re-runs instant --
    but that cache must not survive a change to the pipeline or a retrained
    model, or the app keeps serving stale extractions and looks broken.
    """
    import hashlib

    digest = hashlib.sha1()
    for folder, pattern in ((ROOT / "src" / "factfind", "*.py"),
                            (ROOT / "models", "*.joblib")):
        for path in sorted(folder.glob(pattern)):
            digest.update(path.name.encode())
            digest.update(str(path.stat().st_mtime_ns).encode())
    return digest.hexdigest()[:12]


@st.cache_data(show_spinner=False, max_entries=64)
def run_extraction(file_bytes: bytes, filename: str, use_models: bool,
                   threshold_shift: float, version: str) -> dict[str, Any]:
    """Cached by file content, so re-running with the same files is instant."""
    extractor = get_extractor(use_models, threshold_shift)
    started = time.time()
    result = extractor.extract_bytes(file_bytes, filename)
    payload = result.to_dict()
    payload["elapsed"] = time.time() - started
    payload["n_lines"] = sum(len(p.lines) for p in result.pages)
    payload["ocr_text"] = "\n".join(
        ln.text for p in result.pages for ln in p.lines
    )
    return payload


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------

def fields_dataframe(fields: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for f in fields:
        icon, _ = STATUS_STYLE.get(f["status"], ("", f["status"]))
        rows.append({
            "": icon,
            "Field": f["key"],
            "Label": f["label"],
            "Kind": f["kind"],
            "Value": "" if f["value"] is None else str(f["value"]),
            "Confidence": round(float(f["confidence"]), 3),
            "Status": f["status"],
            "Candidates": ", ".join(str(c) for c in f.get("candidates", [])[:3]),
            "Read as": f.get("raw", ""),
            "Why": f.get("reason", ""),
            "Page": f.get("page", 0) + 1,
        })
    return pd.DataFrame(rows)


def _section_title(kn, section_key: str) -> tuple[str, str]:
    """('Dependants', 'row 2') for 'dependants[1]'."""
    base = section_key.split("[")[0]
    try:
        title = kn.section(base).title
    except KeyError:
        title = base
    suffix = ""
    if "[" in section_key:
        index = int(section_key[section_key.index("[") + 1: section_key.index("]")])
        suffix = f"row {index + 1}"
    return title, suffix


def extracted_view(fields: list[dict[str, Any]], kn,
                   include_unsure: bool = True) -> None:
    """The extracted data as a reader sees it: section by section, in the
    knowledge file's own order, with the value next to its field label."""
    wanted = [
        f for f in fields
        if f["status"] == STATUS_FILLED
        or (include_unsure and f["status"] == STATUS_UNSURE)
    ]
    if not wanted:
        st.info("Nothing was extracted from this document.")
        return

    # Keep the knowledge file's section and field order rather than
    # alphabetical, so the output reads like the form it fills.
    section_order = {s.id: i for i, s in enumerate(kn.sections)}
    field_order = {f.qualified_key: i for i, f in enumerate(kn.fields)}

    grouped: dict[str, list[dict[str, Any]]] = {}
    for f in wanted:
        grouped.setdefault(f["key"].split(".", 1)[0], []).append(f)

    def section_sort(key: str) -> tuple[int, int]:
        base = key.split("[")[0]
        index = int(key[key.index("[") + 1: key.index("]")]) if "[" in key else 0
        return section_order.get(base, 999), index

    for section_key in sorted(grouped, key=section_sort):
        items = sorted(grouped[section_key],
                       key=lambda f: field_order.get(f["field"], 999))
        title, suffix = _section_title(kn, section_key)
        n_filled = sum(1 for f in items if f["status"] == STATUS_FILLED)
        header = f"**{title}**" + (f" · {suffix}" if suffix else "")
        st.markdown(f"{header}  <span style='color:#888'>"
                    f"({n_filled} filled of {len(items)})</span>",
                    unsafe_allow_html=True)

        rows = []
        for f in items:
            filled = f["status"] == STATUS_FILLED
            value = f["value"] if filled else None
            if isinstance(value, bool):
                value = "Yes" if value else "No"
            alternatives = [str(c) for c in f.get("candidates", [])[:3]]
            rows.append({
                "": "✅" if filled else "⚠️",
                "Field": f["label"],
                "Value": "" if value is None else str(value),
                "Confidence": f"{float(f['confidence']) * 100:.0f}%",
                "Suggestions": "" if filled else " / ".join(alternatives),
                "Read as": f.get("raw", ""),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True,
                     hide_index=True)

    unsure = [f for f in wanted if f["status"] == STATUS_UNSURE]
    if unsure and include_unsure:
        st.caption(
            f"⚠️ {len(unsure)} value(s) were read but not filled: the pipeline "
            "was not confident enough. The suggestions column shows what it "
            "considered, best first."
        )


def _rows_for_section(section_id: str, extracted: dict[str, Any]) -> int:
    """How many rows of a repeating section the documents produced."""
    highest = -1
    prefix = f"{section_id}["
    for key in extracted:
        if key.startswith(prefix):
            try:
                highest = max(highest, int(key[len(prefix): key.index("]")]))
            except ValueError:
                continue
    return highest + 1 if highest >= 0 else 1


def _only_if_met(fld, values: dict[str, Any], prefix: str) -> bool:
    """Evaluate a field's onlyIf against what is currently in the form.

    The knowledge file says a field whose condition fails is hidden and saved
    empty, so the form mirrors that rather than showing a box the CRM would
    reject.
    """
    if not fld.only_if:
        return True
    dependency = values.get(f"{prefix}.{fld.only_if['key']}")
    if "filled" in fld.only_if:
        return (dependency not in (None, "")) == bool(fld.only_if["filled"])
    return dependency == fld.only_if.get("equals")


def _status_chip(entry: dict[str, Any] | None) -> str:
    if not entry:
        return ""
    confidence = float(entry.get("confidence", 0.0))
    if entry.get("status") == STATUS_FILLED:
        return f"✅ {confidence * 100:.0f}%"
    return f"⚠️ {confidence * 100:.0f}% — needs checking"


def _render_field(fld, flat_key: str, entry: dict[str, Any] | None,
                  key_prefix: str, prefill_unsure: bool = False) -> Any:
    """One input, with the widget that matches the field's kind."""
    extracted_value = entry.get("value") if entry else None
    if entry and entry.get("status") != STATUS_FILLED:
        # By default an unsure value is never pre-filled: that is the whole
        # point of the unsure status. The adviser can opt into seeing them
        # when the recogniser is weak and correcting beats retyping.
        if prefill_unsure:
            # bestGuess is what the pipeline actually read; candidates are the
            # alternatives it weighed. Prefer the former, fall back to the
            # strongest alternative.
            extracted_value = entry.get("bestGuess")
            if extracted_value in (None, ""):
                candidates = entry.get("candidates") or []
                extracted_value = candidates[0] if candidates else None
        else:
            extracted_value = None
    widget_key = f"{key_prefix}:{flat_key}"
    chip = _status_chip(entry)
    label = f"{fld.label}  ·  {chip}" if chip else fld.label

    help_text = None
    if entry:
        bits = [f"read as: {entry.get('raw', '')!r}"]
        candidates = [str(c) for c in entry.get("candidates", [])[:3]]
        if candidates and entry.get("status") != STATUS_FILLED:
            bits.append("candidates: " + " / ".join(candidates))
        if entry.get("reason"):
            bits.append(entry["reason"])
        help_text = " | ".join(b for b in bits if b)

    kind = fld.kind

    if kind == "choice":
        options = [""] + fld.allowed_values()
        current = str(extracted_value) if extracted_value is not None else ""
        if current and current not in options:
            options.append(current)      # allowOther kept the raw text
        return st.selectbox(label, options, index=options.index(current),
                            key=widget_key, help=help_text) or None

    if kind == "yesNo":
        options = ["", "Yes", "No"]
        current = "" if extracted_value is None else ("Yes" if extracted_value else "No")
        chosen = st.selectbox(label, options, index=options.index(current),
                              key=widget_key, help=help_text)
        return None if chosen == "" else (chosen == "Yes")

    if kind == "date":
        value = None
        if extracted_value:
            try:
                value = date.fromisoformat(str(extracted_value))
            except ValueError:
                value = None
        chosen = st.date_input(label, value=value, key=widget_key,
                               help=help_text, format="DD/MM/YYYY",
                               min_value=date(1900, 1, 1))
        return chosen.isoformat() if chosen else None

    if kind in {"number", "money"}:
        value = None
        if isinstance(extracted_value, (int, float)):
            value = float(extracted_value)
        return st.number_input(
            label, value=value, key=widget_key, help=help_text,
            min_value=float(fld.minimum) if fld.minimum is not None else None,
            max_value=float(fld.maximum) if fld.maximum is not None else None,
            step=1.0 if kind == "number" else 100.0,
            placeholder="not found",
        )

    text = st.text_input(label, value=str(extracted_value or ""),
                         key=widget_key, help=help_text,
                         placeholder="not found")
    return text.strip() or None


def factfind_form(kn, extracted: dict[str, Any], case_type: str,
                  key_prefix: str = "form",
                  prefill_unsure: bool = False) -> dict[str, Any]:
    """The whole fact find, built from the knowledge file, pre-filled.

    Every section, field, label, choice and rule comes from
    fact-find-knowledge.json -- nothing about any field is written here, so
    editing that file changes this form with no code change.
    """
    values: dict[str, Any] = {}

    for section in kn.sections:
        if section.when and section.when != case_type:
            continue
        section_fields = [f for f in section.fields
                          if not f.when or f.when == case_type]
        if not section_fields:
            continue

        n_found = sum(
            1 for f in section_fields
            for k in extracted
            if k.startswith(f"{section.id}[") or k == f"{section.id}.{f.key}"
        )
        rows = _rows_for_section(section.id, extracted) if section.is_repeating else 1
        title = f"{section.title}"
        if n_found:
            title += f"  ·  {n_found} found"

        with st.expander(title, expanded=bool(n_found)):
            for row in range(rows):
                prefix = f"{section.id}[{row}]" if section.is_repeating else section.id
                if section.is_repeating and rows > 1:
                    st.markdown(f"**{section.title.rstrip('s')} {row + 1}**")

                columns = st.columns(2)
                position = 0
                for fld in section_fields:
                    if not _only_if_met(fld, values, prefix):
                        continue
                    flat_key = f"{prefix}.{fld.key}"
                    with columns[position % 2]:
                        values[flat_key] = _render_field(
                            fld, flat_key, extracted.get(flat_key), key_prefix,
                            prefill_unsure=prefill_unsure)
                    position += 1
                if section.is_repeating and row < rows - 1:
                    st.divider()
    return values


def values_to_sections(values: dict[str, Any]) -> dict[str, Any]:
    """Flat form values -> the per-section payload the CRM saves."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or value == "":
            continue
        section, _, rest = key.partition(".")
        if section.endswith("]"):
            base, _, index = section.rpartition("[")
            index = int(index.rstrip("]"))
            rows = out.setdefault(base, [])
            while len(rows) <= index:
                rows.append({})
            rows[index][rest] = value
        else:
            out.setdefault(section, {})[rest] = value
    # Drop rows that ended up completely empty.
    for key, value in list(out.items()):
        if isinstance(value, list):
            out[key] = [row for row in value if row]
            if not out[key]:
                del out[key]
    return out


def summary_metrics(payload: dict[str, Any]) -> None:
    fields = payload["fields"]
    n_filled = sum(1 for f in fields if f["status"] == STATUS_FILLED)
    n_unsure = sum(1 for f in fields if f["status"] == STATUS_UNSURE)
    cols = st.columns(5)
    cols[0].metric("Filled", n_filled)
    cols[1].metric("Unsure", n_unsure)
    cols[2].metric("Text lines", payload.get("n_lines", 0))
    cols[3].metric("Pages", payload["summary"]["pages"])
    cols[4].metric("Time", f"{payload.get('elapsed', 0):.1f}s")


def render_file(name: str, payload: dict[str, Any], raw: bytes) -> None:
    summary_metrics(payload)

    for warning in payload.get("warnings", []):
        st.warning(warning)

    fields = payload["fields"]
    if not fields:
        st.info("Nothing on this page matched a field in the knowledge file.")
    else:
        tab_data, tab_all, tab_filled, tab_unsure, tab_json, tab_page = st.tabs(
            ["Extracted data", "All fields", "Will fill", "Needs a human",
             "JSON", "Page & OCR"]
        )
        with tab_data:
            extracted_view(fields, get_knowledge())
        with tab_all:
            st.dataframe(fields_dataframe(fields), use_container_width=True,
                         hide_index=True)
        with tab_filled:
            filled = [f for f in fields if f["status"] == STATUS_FILLED]
            if filled:
                st.dataframe(fields_dataframe(filled), use_container_width=True,
                             hide_index=True)
            else:
                st.info("Nothing cleared the fill threshold on this document.")
        with tab_unsure:
            unsure = [f for f in fields if f["status"] == STATUS_UNSURE]
            if unsure:
                st.caption(
                    "The pipeline read something here but will not fill it. "
                    "Each row shows the top 3 candidates it considered."
                )
                st.dataframe(fields_dataframe(unsure), use_container_width=True,
                             hide_index=True)
            else:
                st.success("Nothing was left unsure.")
        with tab_json:
            st.caption("This is the payload the Angular app would apply to the form.")
            st.json(payload["sections"])
            st.download_button(
                "Download this document's JSON",
                data=json.dumps(payload["sections"], indent=2, default=str),
                file_name=f"{Path(name).stem}-extracted.json",
                mime="application/json",
                key=f"dl-{name}",
            )
        with tab_page:
            try:
                pages = load_pages_from_bytes(raw, name)
                page_no = 0
                if len(pages) > 1:
                    page_no = st.slider("Page", 1, len(pages), 1,
                                        key=f"pg-{name}") - 1
                st.image(pages[page_no][:, :, ::-1],
                         caption=f"{name} - page {page_no + 1} of {len(pages)}",
                         use_container_width=True)
            except Exception as exc:
                st.info(f"Could not render a preview: {exc}")
            with st.expander("Raw OCR text"):
                st.code(payload.get("ocr_text", "") or "(nothing read)")


def combine(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Merge several documents into one fact find.

    Where two documents fill the same field, the higher-confidence value wins
    and the conflict is recorded rather than silently dropped.
    """
    best: dict[str, dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []

    for name, payload in results.items():
        for f in payload["fields"]:
            if f["status"] != STATUS_FILLED:
                continue
            key = f["key"]
            current = best.get(key)
            if current is None:
                best[key] = {**f, "source": name}
                continue
            if str(current["value"]) != str(f["value"]):
                conflicts.append({
                    "field": key,
                    "kept": current["value"], "from": current["source"],
                    "discarded": f["value"], "also_in": name,
                    "kept_confidence": current["confidence"],
                    "discarded_confidence": f["confidence"],
                })
            if f["confidence"] > current["confidence"]:
                best[key] = {**f, "source": name}

    flat = {k: v["value"] for k, v in best.items()}
    sections: dict[str, Any] = {}
    for key, value in flat.items():
        section, _, rest = key.partition(".")
        if section.endswith("]"):
            base, _, idx = section.rpartition("[")
            idx = int(idx.rstrip("]"))
            rows = sections.setdefault(base, [])
            while len(rows) <= idx:
                rows.append({})
            rows[idx][rest] = value
        else:
            sections.setdefault(section, {})[rest] = value
    return {"sections": sections, "fields": best, "conflicts": conflicts}


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

def main() -> None:
    # One server, two tools. Labelling lives here rather than on its own port
    # so there is a single thing to start and a single URL to remember.
    mode = st.sidebar.radio(
        "Tool",
        ["🔍 Read handwriting", "📄 Extract fact find", "✍️ Label handwriting"],
        key="tool-mode",
        help="Read shows every line and what the model made of it. Extract "
             "maps those lines onto the fact-find form. Label builds the "
             "training set that teaches the recogniser your handwriting.",
    )
    st.sidebar.divider()

    if mode.startswith("🔍"):
        import read_app
        read_app.main()
        return
    if mode.startswith("✍️"):
        import label_app
        label_app.main()
        return

    st.title("📄 Fact-find extraction")
    st.caption(
        "Upload photos or PDFs of client notes. Everything runs locally - "
        "no file leaves this machine, and no external AI API is called."
    )

    try:
        kn = get_knowledge()
    except KnowledgeError as exc:
        st.error("The knowledge file is not usable, so nothing can run:")
        st.code(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Settings")
        use_models = st.toggle(
            "Use trained models", value=True,
            help="Off runs the rule-and-fuzzy baseline only, which is useful "
                 "for seeing what the models actually add.",
        )
        caution = st.select_slider(
            "Caution", options=["Fill more", "Balanced", "Fill less"],
            value="Balanced",
            help="Shifts every fill threshold. 'Fill less' refuses more "
                 "values; 'Fill more' fills values the thresholds would "
                 "normally send to unsure.",
        )
        shift = {"Fill more": -0.12, "Balanced": 0.0, "Fill less": 0.12}[caution]

        extractor = get_extractor(use_models, shift)
        st.divider()
        st.subheader("Loaded")
        st.write(
            f"- knowledge v{kn.version}: **{len(kn.fields)} fields** "
            f"in {len(kn.sections)} sections"
        )
        st.write(f"- linker: **{'yes' if extractor.linker else 'no'}**"
                 + (f" ({extractor.linker.model_name})" if extractor.linker else ""))
        st.write(f"- confidence model: "
                 f"**{'yes' if extractor.confidence_model else 'no'}**")
        engines = engine_report()
        st.write(f"- OCR engine: **{engines['default']}**")
        if not extractor.confidence_model:
            st.info(
                "No trained models found, so this is the baseline. "
                "Run `scripts/03_train.py` to train them."
            )
        if extractor.thresholds:
            with st.expander("Fill thresholds"):
                st.json({k: round(v, 3) for k, v in extractor.thresholds.items()})

    uploads = st.file_uploader(
        "Documents", type=ACCEPTED, accept_multiple_files=True,
        help="Select as many files as you like - JPG, PNG or PDF.",
    )

    if not uploads:
        st.info("Choose one or more files to begin.")
        with st.expander("What the statuses mean"):
            st.markdown(
                "- **✅ filled** - confident enough to put straight into the form.\n"
                "- **⚠️ unsure** - something was read, but not confidently "
                "enough to fill. The top 3 candidates are shown so you can pick.\n"
                "- **— missing** - nothing on the page looked like this field.\n\n"
                "The pipeline is tuned so that a wrong fill is far worse than "
                "an unsure one. It will refuse rather than guess, and it never "
                "fills an NI number, email or phone number it cannot verify "
                "against that field's pattern."
            )
        return

    results: dict[str, dict[str, Any]] = {}
    raw_by_name: dict[str, bytes] = {}
    progress = st.progress(0.0, text="Reading documents…")

    for i, upload in enumerate(uploads, 1):
        raw = upload.getvalue()
        raw_by_name[upload.name] = raw
        progress.progress((i - 1) / len(uploads),
                          text=f"Reading {upload.name} ({i} of {len(uploads)})…")
        try:
            results[upload.name] = run_extraction(raw, upload.name, use_models,
                                                  shift, pipeline_version())
        except Exception as exc:
            st.error(f"{upload.name}: {type(exc).__name__}: {exc}")
    progress.progress(1.0, text=f"Read {len(results)} document(s)")
    progress.empty()

    if not results:
        return

    total_filled = sum(
        sum(1 for f in p["fields"] if f["status"] == STATUS_FILLED)
        for p in results.values()
    )
    total_unsure = sum(
        sum(1 for f in p["fields"] if f["status"] == STATUS_UNSURE)
        for p in results.values()
    )
    total_time = sum(p.get("elapsed", 0) for p in results.values())

    st.subheader("Across all documents")
    cols = st.columns(4)
    cols[0].metric("Documents", len(results))
    cols[1].metric("Fields to fill", total_filled)
    cols[2].metric("Need a human", total_unsure)
    cols[3].metric("Total time", f"{total_time:.1f}s")

    merged = combine(results)
    tab_form, tab_merged, tab_conflicts, tab_each = st.tabs(
        ["📝 Fact-find form", "Merged fact find",
         f"Conflicts ({len(merged['conflicts'])})", "Document by document"]
    )

    with tab_form:
        # Every prediction, not just the confident ones: an unsure field still
        # shows its candidates in the tooltip so the adviser can resolve it.
        best: dict[str, Any] = {}
        for payload in results.values():
            for f in payload["fields"]:
                current = best.get(f["key"])
                if current is None or f["confidence"] > current["confidence"]:
                    best[f["key"]] = f

        n_filled = sum(1 for f in best.values() if f["status"] == STATUS_FILLED)
        n_unsure = len(best) - n_filled

        # If nothing was filled, say why here rather than leaving an empty
        # form and no explanation.
        all_warnings: list[str] = []
        for payload in results.values():
            all_warnings.extend(payload.get("warnings", []))
        if n_filled == 0 and all_warnings:
            for warning in dict.fromkeys(all_warnings):
                st.warning(warning, icon="⚠️")
            if n_unsure:
                st.info(
                    f"{n_unsure} field(s) were read but not confidently enough "
                    f"to fill. They are marked ⚠️ below — hover one to see what "
                    f"the recogniser read and its top candidates, then correct "
                    f"it by hand.",
                    icon="💡",
                )

        st.caption(
            f"Fields are built from the knowledge file. "
            f"**{n_filled}** filled from your documents, **{n_unsure}** read "
            f"but not confident enough to fill (hover the ⚠️ for candidates). "
            f"Everything is editable before you save."
        )
        case_type = st.radio(
            "Case type", ["general", "mortgage", "protection"],
            horizontal=True, key="case-type",
            help="Controls which sections and fields exist, per the "
                 "knowledge file's `when` rules.",
        )
        prefill_unsure = st.checkbox(
            "Pre-fill fields the model is unsure about",
            value=True,
            help="On: every field the model read is filled in, including the "
                 "ones marked ⚠️, so you correct rather than retype. Those "
                 "values are NOT trusted — check each one. Turn this off to "
                 "leave unsure fields empty, so a wrong value can never reach "
                 "the form by accident.",
        )
        if prefill_unsure:
            st.caption(
                "⚠️ marked fields are filled with the model's best guess and "
                "are **not** verified. Check each before saving."
            )
        form_values = factfind_form(kn, best, case_type,
                                    prefill_unsure=prefill_unsure)
        sections_payload = values_to_sections(form_values)

        st.divider()
        left, right = st.columns([1, 3])
        with left:
            completed = sum(1 for v in form_values.values() if v not in (None, ""))
            st.metric("Fields completed", f"{completed} / {len(form_values)}")
        with right:
            st.download_button(
                "Download completed fact find (JSON)",
                data=json.dumps(sections_payload, indent=2, default=str),
                file_name="factfind-completed.json", mime="application/json",
                type="primary", use_container_width=True,
            )
        with st.expander("Payload the Angular app would save"):
            st.json(sections_payload)

    with tab_merged:
        st.caption(
            "All documents merged into one fact find. Where two documents "
            "disagree, the more confident value is kept."
        )
        if merged["sections"]:
            extracted_view(list(merged["fields"].values()), get_knowledge(),
                           include_unsure=False)
            st.download_button(
                "Download merged JSON",
                data=json.dumps(merged["sections"], indent=2, default=str),
                file_name="factfind-merged.json", mime="application/json",
            )
            with st.expander("As JSON (what the Angular app applies to the form)"):
                st.json(merged["sections"])
        else:
            st.info("No field cleared its fill threshold in any document.")

    with tab_conflicts:
        if merged["conflicts"]:
            st.caption("Same field, different answers. Worth a look.")
            st.dataframe(pd.DataFrame(merged["conflicts"]),
                         use_container_width=True, hide_index=True)
        else:
            st.success("No document contradicted another.")

    with tab_each:
        for name, payload in results.items():
            with st.expander(name, expanded=len(results) == 1):
                render_file(name, payload, raw_by_name[name])


if __name__ == "__main__":
    main()
