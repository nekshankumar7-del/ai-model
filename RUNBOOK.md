# Runbook - everything you can build or run

Every command in this file is run from the repository root, on Windows, using
the virtualenv's interpreter directly:

```bash
.venv/Scripts/python.exe ...
```

Use that rather than activating the venv or calling bare `python` - the system
Python has none of these packages, and the error it gives is confusing.

---

## 0. Smart App Control workaround (resolved 17 Nov 2026)

**Smart App Control blocks native Python DLLs on the `E:\` drive** (NumPy,
PyTorch, OpenCV, ONNX Runtime). The fix: a second venv under the user
profile, which Smart App Control does not enforce on:

```
C:\Users\NEKSHAN KUMAR\python-envs\aimodel\
```

All commands in this runbook now use this venv:

```bash
"C:\Users\NEKSHAN KUMAR\python-envs\aimodel\Scripts\python.exe" ...
```

The old `.venv` on `E:\` still exists but its native extensions will not
load while Smart App Control is enforcing. The registry value is still 1:

```powershell
(Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy').VerifiedAndReputablePolicyState
# 0 = off, 1 = enforcing (blocked), 2 = audit only
```

If Smart App Control is ever turned off (Windows Security -> App & browser
control -> Smart App Control -> Off), the original `.venv` will work again.

Zscaler is a separate thing and is **not** the blocker. It intercepts TLS,
which is why downloads use the CA bundle; that is already handled. The
bundle is at `C:\Users\NEKSHAN KUMAR\python-envs\aimodel\corp-ca-bundle.pem`.

---

## 1. Setup

```bash
# Python 3.11 — venv under user profile to avoid Smart App Control
"C:\Users\NEKSHAN KUMAR\AppData\Local\Programs\Python\Python311\python.exe" -m venv "C:\Users\NEKSHAN KUMAR\python-envs\aimodel"

set PY="C:\Users\NEKSHAN KUMAR\python-envs\aimodel\Scripts\python.exe"
set CERT="C:\Users\NEKSHAN KUMAR\python-envs\aimodel\corp-ca-bundle.pem"

%PY% -m pip install -r requirements.txt --cert %CERT% --timeout 300
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cpu --cert %CERT% --timeout 600
```

Check it worked:

```bash
"C:\Users\NEKSHAN KUMAR\python-envs\aimodel\Scripts\python.exe" -c "import numpy, torch, cv2, onnxruntime, streamlit; print('OK')"
```

---

## 2. Run the app

The Streamlit app is the only runnable UI. It opens on
**http://localhost:8501**.

```bash
# the fact-find extractor - upload photos/PDFs, see filled / unsure / missing
.venv/Scripts/python.exe -m streamlit run app/streamlit_app.py --server.address localhost

# transcribe cropped values, to build training data for the recogniser
.venv/Scripts/python.exe -m streamlit run app/label_app.py --server.address localhost

# read-only viewer
.venv/Scripts/python.exe -m streamlit run app/read_app.py --server.address localhost
```

`--server.address localhost` keeps it off the network. Leave it on once real
client notes go through it.

If the app loads but every field comes back empty, `models/` is empty - build
them with section 3.

**There is no Angular app to run.** `angular/` is TypeScript source plus
exported ONNX assets, meant to be dropped into an existing Angular project.
It has no `package.json` of its own. See `angular/README.md`.

---

## 3. Build the fact-find models

The original pipeline. Run the steps in order; each writes into `models/` and
`reports/`.

```bash
# generate pages to work with (skip if you have real labelled data)
.venv/Scripts/python.exe scripts/00_generate_synthetic.py --pages 420 --writers 14

.venv/Scripts/python.exe scripts/01_audit.py          --data data/synthetic
.venv/Scripts/python.exe scripts/02_prepare_splits.py --data data/synthetic
.venv/Scripts/python.exe scripts/03_train.py          --data data/synthetic
.venv/Scripts/python.exe scripts/04_evaluate_test.py  --data data/synthetic
.venv/Scripts/python.exe scripts/05_export_onnx.py
```

With your own data, put pages and matching JSON labels in `data/dataset/` and
pass `--data data/dataset` instead. See "Using your own data" in `README.md`.

`04_evaluate_test.py` is deliberately one-shot - it refuses to re-run on the
locked test split without `--allow-rerun`, so test numbers cannot be quietly
tuned.

| script | what it does |
| --- | --- |
| `00_generate_synthetic.py` | renders fact-find pages from the knowledge file |
| `01_audit.py` | duplicates, unreadable pages, label disagreements, personal-data screen |
| `02_prepare_splits.py` | leakage-free 70/15/15 splits grouped by writer and client |
| `03_train.py` | baseline rules, then the linker and confidence models |
| `04_evaluate_test.py` | the locked test split, once |
| `05_export_onnx.py` | ONNX + the Angular spec |
| `07_autolabel.py` | attaches transcriptions to detected crops automatically |
| `08_fetch_fonts.py` | downloads handwriting fonts for the synthetic renderer |

---

## 4. Train the handwriting recogniser on your own notes

Two stages: pre-train on synthetic handwriting, then fine-tune on crops your
advisers transcribed in `label_app.py`.

```bash
# stage 1 only - needs no real data
.venv/Scripts/python.exe scripts/06_train_recognizer.py --pretrain-only

# both stages (needs at least 50 transcribed crops)
.venv/Scripts/python.exe scripts/06_train_recognizer.py
```

---

## 5. Train on public handwriting datasets

The pipeline added for the public-corpus work. Full background in
`docs/htr-datasets.md` and `docs/htr-findings.md`.

### 5.1 See what is registered

```bash
.venv/Scripts/python.exe scripts/10_fetch_datasets.py --list
```

17 datasets with licence, access mode, size, script and level. A `*` marks
the default set.

### 5.2 Download

```bash
# default English set: EMNIST + NIST SD19 by_class, 1.6 GB
.venv/Scripts/python.exe scripts/10_fetch_datasets.py

# named datasets
.venv/Scripts/python.exe scripts/10_fetch_datasets.py --datasets uci_dhcd iiit_hw_dev

# everything downloadable, with a size cap
.venv/Scripts/python.exe scripts/10_fetch_datasets.py --datasets all --max-gb 40
```

Downloads resume, are size-checked and SHA-256 hashed into
`data/metadata/inventory.json`. Re-running skips what is already there.

**Already downloaded**: `emnist` (561.8 MB) and `nist_sd19` (1,031.6 MB).

Datasets needing registration (IAM, GNHK) are reported with instructions and
skipped, never fetched from a mirror.

### 5.3 IAM from a mirror - benchmarking only

```bash
.venv/Scripts/python.exe -m pip install -U datasets pillow
.venv/Scripts/python.exe scripts/download_iam_handwriting.py
```

IAM is **CC BY-NC 4.0, non-commercial research only** - mirror or not. Use it
to compare against published IAM numbers, not for anything that ships.

### 5.4 Build manifests

```bash
.venv/Scripts/python.exe scripts/11_build_manifests.py

# a fast subset first
.venv/Scripts/python.exe scripts/11_build_manifests.py --limit 5000
```

Converts every corpus to one schema, validates, drops empty/over-long/
unreadable/blank/duplicate rows and reports what went. Writes
`data/processed/manifests/<id>.jsonl`.

### 5.5 Train

```bash
# character model
.venv/Scripts/python.exe scripts/12_train_htr.py --level char

# word model, with character crops stitched into word images
.venv/Scripts/python.exe scripts/12_train_htr.py --level word --compose-words

# smoke run - minutes, not hours
.venv/Scripts/python.exe scripts/12_train_htr.py --level char --epochs 2 --max-samples 4000
```

Does the writer-independent split, builds the Unicode charset from the
training split only, trains with early stopping, scores the held-out test
split and exports ONNX.

Useful flags:

| flag | default | why |
| --- | --- | --- |
| `--arch` | `crnn` | also `crnn-large`, `transformer` |
| `--epochs` | 20 | |
| `--batch-size` | 32 | |
| `--max-samples` | all | cap the training split |
| `--augment` | 1.0 | 0 disables augmentation |
| `--preprocess` | `light` | `none`, `light`, `full` |
| `--workers` | 0 | raise it on a machine that is not I/O bound |
| `--device` | `cpu` | `cuda` if a GPU appears |

This box is CPU-only (12 cores, no CUDA), so a full run is hours. Use the
smoke run to confirm the path works first.

### 5.6 Evaluate and export

```bash
# score the saved checkpoint on the same held-out split
.venv/Scripts/python.exe scripts/13_evaluate_htr.py --name htr-char-crnn

# read some images with it
.venv/Scripts/python.exe scripts/13_evaluate_htr.py --name htr-char-crnn --images "data/crops/*.png"

# export to ONNX and verify parity against PyTorch
.venv/Scripts/python.exe scripts/14_export_htr_onnx.py --name htr-char-crnn
```

---

## 6. Tests

```bash
# the whole suite
.venv/Scripts/python.exe -m pytest tests/ -q

# the HTR pipeline end to end on generated crops - no network, no datasets
.venv/Scripts/python.exe -m pytest tests/test_htr_pipeline.py -q

# the exported ONNX reproduces the Python predictions
.venv/Scripts/python.exe -m pytest tests/test_onnx_parity.py -v
```

`test_htr_pipeline.py` is the fastest check that an environment is working:
it needs no downloaded data and covers adapters, validation, splits, charset,
augmentation, a training step, CTC decoding, ONNX export and parity.

---

## 7. Where things land

```
data/
  raw/          downloaded archives, one directory per dataset
  extracted/    unpacked archives
  processed/    normalised crops, shards and per-dataset manifests
  metadata/     inventory.json, build_report.json, htr_splits.json
  dataset/      your own labelled pages
  synthetic/    generated pages
models/         trained models + model cards
models/htr/     public-dataset checkpoints, charsets and ONNX exports
reports/        audit.md, final.md, metrics JSON, figures
angular/assets/ the exported browser bundle
```

`data/` is in `.gitignore` in its entirety and nothing under it should ever be
committed - it holds client material.

---

## 8. When something fails

| symptom | cause |
| --- | --- |
| `DLL load failed ... Application Control policy` | Section 0. Not a Python problem. |
| `SSLError: certificate verify failed` | Zscaler TLS interception. Set `HTR_CA_BUNDLE` to `.venv/corp-ca-bundle.pem`, or let `htr/paths.py` find it. |
| `STOPPING. PyTorch is not installed.` | Section 1. |
| `no manifests built` | Run `11_build_manifests.py` first. |
| `refusing to mix levels` | Character and word corpora need separate runs. Pass `--level char` or `--level word`. |
| `WARNING: N batches skipped as non-finite` | Crops too narrow for their labels. Raise `--preprocess` quality or check the adapter. |
| extraction takes longer than the download | Expected on OneDrive-synced paths with 100k+ file archives. SD19 is read straight from its zip for this reason. |
| test numbers look too good | Check `data/metadata/htr_splits.json`. If a dataset says "split by sample", its numbers are within-writer and will flatter the model. |
