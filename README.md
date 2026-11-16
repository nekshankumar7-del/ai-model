# Fact-find extraction

Reads photos and PDFs of handwritten or typed client notes and fills fact-find
fields, refusing to guess when it is not confident. Trains in Python, ships to
the Angular app as quantised ONNX that runs in the browser - no server, no
external AI API.

Every field, label, alias, choice and validation rule comes from
`knowledge/fact-find-knowledge.json` at runtime. Nothing about a field is
hard-coded: add a field to that file, re-export, and both the Python and the
browser pipeline pick it up.

## Status

The pipeline is complete and working end to end. **It has been trained and
measured on generated pages, because `data/dataset/` is empty** - see
[Using your own data](#using-your-own-data). Numbers below are from those
generated pages and are an optimistic upper bound on real handwriting.

Test split (74 pages, 3 writers held out, scored once):

| | baseline (rules) | final (with models) |
| --- | ---: | ---: |
| wrong-fill rate | 32.8% | **4.6%** |
| exact match | 44.7% | **36.2%** |
| coverage | 58.0% | **37.7%** |
| wrong values filled | 300 | **24** |

The models cut wrong fills roughly sevenfold. The brief's per-kind targets were
met on validation but **missed on test for 7 of 11 kinds** - the thresholds
were fitted on only two validation writers and did not carry to three unseen
ones. That gap is the main finding, and `reports/final.md` sets out exactly
what would close it. The targets have not been lowered.

## Quick start

```bash
# 1. environment (Python 3.11)
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 2. generate pages to work with (skip if you have real data)
.venv/Scripts/python.exe scripts/00_generate_synthetic.py --pages 420 --writers 14

# 3. audit, split, train, evaluate, export
.venv/Scripts/python.exe scripts/01_audit.py          --data data/synthetic
.venv/Scripts/python.exe scripts/02_prepare_splits.py --data data/synthetic
.venv/Scripts/python.exe scripts/03_train.py          --data data/synthetic
.venv/Scripts/python.exe scripts/04_evaluate_test.py  --data data/synthetic
.venv/Scripts/python.exe scripts/05_export_onnx.py

# 4. check the ONNX reproduces the Python predictions
.venv/Scripts/python.exe -m pytest tests/test_onnx_parity.py -v

# 5. try it
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address localhost
```

`--server.address localhost` keeps the app off your network, which matters
when you start feeding it real client notes.

## Using your own data

Put each page in `data/dataset/` with a JSON label file of the same name:

```
data/dataset/
  smith-note-1.jpg
  smith-note-1.json
```

The label file uses the knowledge file's own keys, flat or nested:

```json
{
  "personal-details.fullName": "John Smith",
  "personal-details.dateOfBirth": "1982-03-14",
  "employment.grossAnnualSalary": 45000,
  "dependants[0].name": "Amy Smith",
  "dependants[0].relationshipType": "Daughter",
  "_meta": { "writer": "adviser-ab", "client": "client-00123", "layout": "label_value" }
}
```

`_meta.writer` and `_meta.client` matter: splits are built from connected
components over those two, so no hand and no person can appear on both sides
of a split. Without them every page becomes its own group, which is safe but
wastes data.

Then run steps 3-5 above with `--data data/dataset`. Start with
`scripts/01_audit.py`, which reports duplicates, unreadable pages, labels the
OCR disagrees with, and how many more examples each field needs.

## How it decides

```
image/pdf
  -> preprocess      page crop, perspective, deskew, shadow flattening,
                     ruled-line removal, adaptive threshold
  -> OCR             PP-OCRv4 via RapidOCR (ONNX). Tesseract supported with
                     FACTFIND_OCR=tesseract if the binary is installed.
  -> segment         split each line into candidate (label, value) readings
  -> match           score every field from the knowledge file: label
                     similarity, does the value parse under that kind, which
                     section heading is in force
  -> linker          re-rank the candidates (gradient boosting, ONNX)
  -> assign          one value per field; rows for repeating sections
  -> confidence      calibrated P(correct), from recogniser confidence,
                     candidate margins, parser validity, cross-field checks
                     and image quality (gradient boosting + isotonic, ONNX)
  -> decide          filled | unsure | missing
```

Three outcomes, and the distinction is the whole point:

- **filled** - the calibrated probability cleared that kind's threshold.
- **unsure** - something was read but not confidently enough. Top 3 candidates
  are returned so an adviser chooses. On the test split, **83% of unsure
  fields had the right answer in the top 3**.
- **missing** - nothing on the page looked like this field.

NI numbers, emails and phone numbers are never filled unless they satisfy that
field's pattern exactly, whatever the model thinks.

### Why the linker is a re-ranker, not a 107-way classifier

The brief asks for a model that predicts the field key per line. It is built
as a re-ranker over the rule matcher's candidates instead, deliberately: the
knowledge file already names every field's labels, so the rules narrow 107
fields to a handful with near-perfect recall. A flat softmax over field keys
would have to relearn that mapping, needs examples of every class, and breaks
every time you add a field to the knowledge file - which the file is expressly
designed to let you do without retraining. Model family (logistic regression /
linear SVM / gradient boosting) is chosen by grouped cross-validation;
gradient boosting won.

## Layout

```
knowledge/   the knowledge file - the source of truth for all field logic
src/factfind/
  knowledge.py    loader and validator
  preprocess.py   page preparation + image-quality features
  ocr.py          pluggable OCR engines
  segment.py      label/value splitting, section tracking
  parsers.py      per-kind parsers; refuse rather than guess
  matcher.py      rule + fuzzy field scoring (the baseline)
  linker.py       candidate re-ranker
  confidence.py   P(correct) + threshold selection
  extract.py      the pipeline
  evaluate.py     metrics: exact, wrong-fill, unsure, missing, coverage curve
  audit.py        dataset audit incl. duplicate and personal-data screening
  splits.py       leakage-free grouped splits
  synth.py        synthetic page generator
  export_onnx.py  ONNX export + the Angular spec
scripts/     the numbered pipeline, run in order
app/         Streamlit testing front end (multi-file upload)
angular/     TypeScript pipeline + exported assets; see angular/README.md
reports/     audit.md, final.md, metrics JSON, coverage curve
models/      trained models and a model card each
tests/       ONNX parity tests
```

## Client data

`data/` is in `.gitignore` and nothing under it should ever be committed.
Reports contain counts, field names and file stems only - never a client
value. `scripts/01_audit.py` screens the label files for NI numbers, emails,
phone numbers, passport numbers and postcodes and tells you how many it found
so you can anonymise before sharing the folder.

The Streamlit app and the Angular app both run entirely locally. No document
is uploaded anywhere and no external API is called.

## Known limitations

- **Trained on generated pages.** Real handwriting is harder. Re-run the
  pipeline on real notes before trusting any number here.
- **Thresholds do not yet generalise across writers.** Validation met the
  targets; test did not. More writers is the fix.
- **No fine-tuned handwriting recogniser yet.** Step 4.2 of the brief (a CRNN-CTC
  or fine-tuned recogniser on cropped value regions) is not built, because it
  needs thousands of real labelled crops. Until then, names and free text are
  limited by the stock recogniser, which is where most of the remaining error
  sits.
- **The browser OCR adapter is tesseract.js**, while the thresholds were
  calibrated against PP-OCRv4. Re-fit with `FACTFIND_OCR=tesseract` if you ship
  tesseract - see `angular/README.md`.

## License

This project is licensed under the [MIT License](LICENSE).
