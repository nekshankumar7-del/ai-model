# HTR training pipeline - inspection findings and design decisions

Written before any code was added, as a record of what the project already had
and why the new module is shaped the way it is.

## 1. What the project already is

- **Language / runtime:** Python 3.11 in `.venv/`, plus a TypeScript/Angular
  inference port under `angular/`.
- **Purpose:** read photographs and PDFs of adviser fact-find notes and fill
  form fields, refusing to guess. Everything about a field comes from
  `knowledge/fact-find-knowledge.json` at runtime.
- **Existing pipeline** (`scripts/00`-`08`, `src/factfind/`):
  preprocess -> OCR (PP-OCRv4 via RapidOCR) -> line segmentation ->
  rule matcher -> gradient-boosting re-ranker (`linker.py`) ->
  calibrated confidence (`confidence.py`) -> filled / unsure / missing,
  exported to ONNX for the browser.
- **An HTR model already exists.** `src/factfind/recognizer.py` has a
  CRNN-CTC (CNN -> 2-layer BiLSTM -> linear head, ~2M params, 32px input,
  dynamic width, blank index 0) with greedy and lexicon-constrained CTC
  decoding, plus ONNX export. `src/factfind/recognizer_train.py` trains it in
  two stages: pre-train on lines rendered from handwriting fonts using the
  knowledge file's own vocabulary, then fine-tune on real labelled crops.

## 2. The gap this task fills

The README names it directly: *"No fine-tuned handwriting recogniser yet ...
it needs thousands of real labelled crops."* The existing recogniser has only
two data sources - synthetic font renderings and whatever crops an adviser has
hand-transcribed in `app/label_app.py`. Both are tiny. The recogniser is the
accuracy bottleneck for names and free text.

Two further limits block simply pointing the old trainer at public data:

- `Charset.default()` is **ASCII-only** (`A-Za-z0-9` plus 17 punctuation
  marks). It cannot represent Devanagari or any other Indic script.
- The character index order is baked into every exported ONNX model, so the
  existing charset must not be mutated.

## 3. Design decisions

| Decision | Why |
| --- | --- |
| New package `src/htr/`, existing `src/factfind/` untouched | The shipped fact-find pipeline works and is exported to the browser. Public-dataset training is a different concern with a different charset and a different data layout. |
| Reuse `factfind.recognizer.CRNN` rather than copy it | It is already parameterised by `n_classes`, so the same proven architecture serves both. One architecture, one ONNX export path. |
| New `UnicodeCharset`, built **from the data** | Devanagari, Bengali, Tamil and the rest need real Unicode support, and a charset derived from the corpus is the only way to guarantee every label is encodable. NFC normalisation is applied so visually identical strings share one encoding. |
| Character-level and word-level datasets get **separate configs** | A CTC model trained on single glyphs learns a degenerate alignment. EMNIST/NIST/DHCD train the `char` config; IIIT-HW word images train the `word` config. They are never silently merged. |
| Optional **word composition** from character datasets | Character corpora can still help the word model if their glyphs are stitched into word images with realistic spacing. This is opt-in (`--compose-words`), not the default. |
| Writer-independent splits wherever writer IDs exist | NIST SD19 `by_write` and the IIIT corpora carry writer IDs. Splitting on them is the difference between a real generalisation number and a flattering one. The existing `factfind/splits.py` already does grouped splits for pages; the same principle is applied here at crop level. |
| Resumable downloads, manifest-driven everything | The archives are 0.5-6 GB each over a corporate proxy. A download that cannot resume is a download that never finishes. |
| CRNN by default, transformer encoder available | CPU-only box (12 cores, no CUDA). CRNN-CTC trains in hours; a transformer encoder is provided behind `--arch transformer` for when a GPU is available. TrOCR was rejected: ~60 MB and far too slow for the browser target this project ships to. |

## 4. Dataset availability - what was actually verified

Probed live on 2026-11-17 from this machine (corporate TLS interception; all
requests go through `.venv/corp-ca-bundle.pem`).

### Downloadable automatically

| Dataset | Script | Level | Size | Licence / terms |
| --- | --- | --- | ---: | --- |
| EMNIST (`gzip.zip`) | Latin | character | 562 MB | NIST, US Government work - free to use, citation requested |
| NIST SD19 (`by_write.zip`) | Latin | character, **with writer IDs** | 568 MB | NIST, US Government work - public domain in the US |
| UCI Devanagari Handwritten Character Dataset | Devanagari | character | ~95 MB | CC BY 4.0 |
| IIIT-HW-Dev / IIIT-HW-Hindi v1 | Devanagari | **word** | 1.8 GB | CVIT IIIT-H, research use |
| IIIT-HW-Telugu v1 | Telugu | word | 3.7 GB | CVIT IIIT-H, research use |
| IIIT-INDIC-HW-WORDS (bn, gu, kn, ml, or, pa, ta, ur) | 8 Indic scripts | word | 2.6-6.0 GB each | CVIT IIIT-H, research use |

**Correction after inspecting the download.** `by_write.zip` was registered
first, because it is the only SD19 partition that preserves writer identity.
It turned out to contain 814,255 PNG files and nothing else - no `.mit`
files, no transcriptions - and its filenames cannot be joined to
`by_class`'s. It is unlabelled, so `by_class.zip` (1.03 GB) is the default
and `by_write` is optional. `by_class` keeps the `hsf_N` partition in each
path, and NIST cut those by writer, so `hsf_4` serves as a writer-disjoint
test split.

**The default download set is EMNIST + NIST SD19 only** (1.1 GB, English).
The Indic corpora are registered, verified live and one flag away
(`--datasets uci_dhcd iiit_hw_dev`), but they are not fetched by default: this
project reads English fact-find notes, and ~38 GB of Indic word images is not
a sensible default for that. See `docs/htr-datasets.md`.

### Not downloadable - registered, documented, adapters provided

| Dataset | What blocks it |
| --- | --- |
| **IAM Handwriting Database** | Requires a free registered account at `fki.tic.heia-fr.ch`, and that host does not respond from this network at all. Its licence is **CC BY-NC 4.0** - non-commercial research only, which matters for a commercial advisory firm. Registering and bypassing the login was not attempted. |
| **GNHK (GoodNotes Handwriting Kollection)** | Licence is CC BY 4.0 and the terms are genuinely open, but the download behind `goodnotes.com/gnhk` no longer serves the dataset and `static.gnhk-dataset.net` does not resolve. Must be requested from the authors. |
| **HP Labs India Indic datasets** | The LipiTk index page is still up, but every archive it links to lives on `mile.ee.iisc.ernet.in`, which returns 404 (and 502 on the `.ac.in` successor). The data is effectively gone from its official host. |
| **CENSUS-HWR** (1.8 M English words, CC BY 4.0) | The paper points at `censustree.org` for the download; that site no longer exposes a dataset link. Worth re-checking - it is the best open-licensed English word corpus found. |

For each of these, `src/htr/adapters/` contains a converter, so dropping the
archive into `data/raw/<id>/` by hand is all that is needed once you obtain it.

### Deliberately not used

Third-party mirrors of IAM (Hugging Face, Kaggle) were found but **not used**.
They redistribute data whose own terms require registration with the original
provider, and the task is explicit about not bypassing access controls.

## 5. Consequence for English

With IAM, GNHK and CENSUS-HWR all unavailable automatically, the only
English handwriting the pipeline can fetch on its own is **character-level**
(EMNIST, NIST SD19 by_class). Both ship a writer-disjoint official split that
the splitter honours, so the character numbers are honest.

English *word* and *line* recognition therefore leans on:

1. real glyphs composed into word images (`--compose-words`), and
2. the existing font-rendered synthetic corpus in
   `factfind/recognizer_train.py`.

This is stated plainly rather than papered over: composed words have no
genuine cursive joins and no natural inter-letter spacing, so a word-level
English number from this pipeline alone is an optimistic estimate.

**One exception was added on request.** `scripts/download_iam_handwriting.py`
pulls IAM lines from a Hugging Face mirror (`Teklia/IAM-line`, 10,373 lines
with IAM's own writer-disjoint split). It is registered separately as
`iam_hf`, is not part of any default, and records its provenance to disk. The
licence does not change by being mirrored - CC BY-NC 4.0, non-commercial
research only - so it is a benchmarking reference, not a route to a shippable
model. The canonical `iam` entry still points at the original provider.
