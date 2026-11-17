# Handwriting datasets: inventory, licences and access

Everything the pipeline knows about is in `src/htr/registry.py`. This page is
the human-readable version, plus the licence position for each entry.

Regenerate the live view at any time:

```bash
.venv/Scripts/python.exe scripts/10_fetch_datasets.py --list
```

All URLs were probed from this machine on **2026-11-17**. The network sits
behind a TLS-intercepting corporate proxy, so every request uses
`.venv/corp-ca-bundle.pem`; `src/htr/paths.py:ca_bundle()` finds it
automatically, and `HTR_CA_BUNDLE` overrides it.

## Downloaded by default

| id | dataset | script | level | size | licence |
| --- | --- | --- | --- | ---: | --- |
| `emnist` | EMNIST ByClass | Latin | character | 562 MB | US Government work (NIST) - free to use, citation requested |
| `nist_sd19` | NIST SD19, `by_class` | Latin | character | 1.03 GB | US Government work (NIST) - public domain in the US |

1.6 GB in total. Both are English, both are character-level, and both permit
commercial use.

### Why `by_class` and not `by_write`

`by_write` is the SD19 partition that preserves writer identity, so it was
the obvious first choice. It was downloaded and inspected, and it does not
work: the archive contains **814,255 PNG files and nothing else** - no `.mit`
files, no transcriptions, no label index of any kind. Writer identity without
labels cannot train a recogniser.

The labels live in `by_class`, where the class is the directory name as a hex
ASCII code (`4c` is `L`). The two cannot be joined, because the filenames are
unrelated: `c1000_45_00000.png` in `by_write` against `hsf_6_00156.png` in
`by_class`.

So `by_class` is the default and `by_write` stays registered as optional, for
anyone who wants the writer grouping for unsupervised work. What `by_class`
does keep is the `hsf_N` partition in the path, and NIST cut those partitions
by writer - `hsf_4` is the standard test set and shares no writer with
`hsf_0`-`hsf_3`. The adapter uses that as the split hint, so the test split is
NIST's own.

SD19 and EMNIST are the same images: EMNIST is SD19 ByClass downsampled to
28x28 and deskewed. Both are registered because SD19 keeps the original
128x128, which matters for composing character crops into word images - a
28x28 glyph upscaled into a word is mush.

SD19 is also **never unpacked**. `by_class.zip` holds 814,255 files, and
writing those onto a OneDrive-synced disk takes longer than downloading them
did. The adapter reads members straight out of the zip, downsamples each to
64px tall and streams them into memory-mapped `.npy` shards, so the only
files created are a handful of shards. `--limit` on that dataset is applied
per class, so a smaller corpus stays balanced across all 62 classes instead
of being the first few letters of the alphabet.

## How the splits are decided

In order of preference, per dataset:

1. **The corpus's own official split**, when it ships one. EMNIST, SD19
   (`hsf_4`) and IAM all cut theirs by writer, and re-cutting at random
   because a redistributed copy dropped the writer column would put the same
   hand on both sides. Validation is carved out of the official *train* side
   only, so the official test set is left exactly as its authors defined it.
2. **By writer**, when there are at least four distinct writer ids.
3. **By sample**, when there is neither - and the split report then says so
   in as many words: *"validation and test numbers for this dataset are
   within-writer and will flatter the model"*.

Which of the three was used is recorded per dataset in
`data/metadata/htr_splits.json`, alongside a leakage check that asserts no
writer group appears on two sides.

## Available on request, not downloaded by default

| id | dataset | script | level | size | licence |
| --- | --- | --- | --- | ---: | --- |
| `uci_dhcd` | Devanagari Handwritten Character Dataset | Devanagari | character | ~95 MB | CC BY 4.0 |
| `iiit_hw_dev` | IIIT-HW-Dev / IIIT-HW-Hindi v1 | Devanagari | word | 1.8 GB | CVIT IIIT-H, research use |
| `iiit_hw_telugu` | IIIT-HW-Telugu v1 | Telugu | word | 3.7 GB | CVIT IIIT-H, research use |
| `iiit_indic_bn` | IIIT-INDIC-HW-WORDS | Bengali | word | 3.4 GB | CVIT IIIT-H, research use |
| `iiit_indic_gu` | IIIT-INDIC-HW-WORDS | Gujarati | word | 4.0 GB | CVIT IIIT-H, research use |
| `iiit_indic_kn` | IIIT-INDIC-HW-WORDS | Kannada | word | 5.2 GB | CVIT IIIT-H, research use |
| `iiit_indic_ml` | IIIT-INDIC-HW-WORDS | Malayalam | word | 6.0 GB | CVIT IIIT-H, research use |
| `iiit_indic_or` | IIIT-INDIC-HW-WORDS | Odia | word | 2.9 GB | CVIT IIIT-H, research use |
| `iiit_indic_pa` | IIIT-INDIC-HW-WORDS | Gurmukhi | word | 3.6 GB | CVIT IIIT-H, research use |
| `iiit_indic_ta` | IIIT-INDIC-HW-WORDS | Tamil | word | 5.1 GB | CVIT IIIT-H, research use |
| `iiit_indic_ur` | IIIT-INDIC-HW-WORDS | Arabic (Urdu) | word | 2.6 GB | CVIT IIIT-H, research use |

These are all live and directly downloadable - they are simply not in the
default set, because the brief for this project is English fact-find notes:

```bash
.venv/Scripts/python.exe scripts/10_fetch_datasets.py --datasets uci_dhcd iiit_hw_dev
```

The whole Indic collection is ~38 GB, so `--max-gb` is there to stop a
`--datasets all` run filling the disk.

## Registered but not downloadable here

Each of these has a working adapter. Obtain the archive yourself, drop it in
`data/raw/<id>/`, and `scripts/11_build_manifests.py` picks it up.

### IAM Handwriting Database - `iam`

The reference English line-level corpus, 115k labelled lines, 657 writers.

* **Blocked by:** a free account at `fki.tic.heia-fr.ch`. No attempt was made
  to work around that, and the host did not respond from this network at all.
* **Licence:** CC BY-NC 4.0. **Non-commercial research only.** Wealthmax is a
  commercial firm, so a model trained on IAM is not something this project can
  ship - worth settling before anyone spends a week on it.
* **Mirrors were deliberately not used.** Copies exist on Hugging Face and
  Kaggle. They redistribute data whose own terms require registration with the
  original provider, so using them would be bypassing an access control by
  proxy.
* To use it: register, download `lines.tgz`, `words.tgz` and `ascii.tgz`, put
  them in `data/raw/iam/`. The adapter reads `ascii/lines.txt`, keeps only
  lines IAM itself marks `ok`, and takes writer ids from `forms.txt`.

### IAM via a mirror - `iam_hf`

Added at the user's explicit request, after the above was put to them, for
benchmarking against published IAM numbers:

```bash
pip install -U datasets pillow
python scripts/download_iam_handwriting.py
```

This pulls `Teklia/IAM-line` (10,373 lines: 6,482 train / 976 validation /
2,915 test) from the Hugging Face hub into `data/extracted/iam_hf/`, writes a
`PROVENANCE.json` recording the mirror, the date and the licence, and is read
by the generic folder-label adapter.

The licence does not change by being mirrored: **CC BY-NC 4.0, non-commercial
research only**. It is for comparison, not for anything Wealthmax ships. The
canonical `iam` entry still points at the original provider.

The mirror drops IAM's writer column but keeps its official train/validation/
test split, which IAM's authors cut by writer. The splitter honours an
official split in preference to re-cutting one - see below.

### GNHK - GoodNotes Handwriting Kollection - `gnhk`

39k words of camera-captured English handwriting "in the wild", **CC BY 4.0**,
genuinely open terms.

* **Blocked by:** the download is gone. `goodnotes.com/gnhk` now serves
  marketing pages, and `static.gnhk-dataset.net` does not resolve.
* Request it from the authors listed in
  <https://github.com/GoodNotes/GNHK-dataset>. The adapter reads the per-image
  JSON polygons and cuts word crops.
* This is the one to chase: open licence, camera-captured, unconstrained
  English - the closest public match to photographed adviser notes.

### CENSUS-HWR - `census_hwr`

1.8M English handwritten words from the 1930 and 1940 US censuses, CC BY 4.0.

* **Blocked by:** the paper (arXiv 2305.16275) points at `censustree.org`, and
  that site no longer exposes a dataset link.
* Best open-licensed English word corpus found. Worth re-checking periodically.

### HP Labs India Indic characters - `hp_labs_indic`

* **Blocked by:** the data is gone from its official host. The LipiTk index
  page is still up, but every archive it links to lives on
  `mile.ee.iisc.ernet.in`, which returns 404; the `.ac.in` successor returns
  502.
* No mirror was substituted.

## What this leaves

Automatically obtainable English handwriting is **character-level only**.
IAM, GNHK and CENSUS-HWR - every freely-licensed English word or line corpus -
are all unavailable for the reasons above.

So the word-level English model is trained on:

1. **Real glyphs composed into words** (`--compose-words`). Characters are
   drawn from one NIST writer per word and stitched with realistic spacing,
   baseline jitter and per-glyph scale. Real ink, real writer variation,
   synthetic layout.
2. **The existing font-rendered synthetic corpus** in
   `factfind/recognizer_train.py`, which renders the knowledge file's own
   vocabulary in handwriting fonts.

That is a real limitation, not a formality. Composed words do not have genuine
cursive joins or natural inter-letter spacing, so a word-level English number
from this pipeline is an optimistic estimate until GNHK or real labelled crops
arrive. The character-level numbers are sound.

## Licence obligations when you publish

* **NIST (EMNIST, SD19):** cite the dataset. EMNIST asks for the Cohen et al.
  2017 paper; SD19 asks for the NIST SRD citation.
* **CC BY 4.0 (UCI DHCD, GNHK, CENSUS-HWR):** attribute the creator and state
  the licence.
* **CC BY-NC 4.0 (IAM):** attribution *and* non-commercial use only.
* **CVIT (IIIT-*):** research use, cite the IIIT-INDIC-HW-WORDS / IIIT-HW-Dev
  papers, do not redistribute the data.

The inventory at `data/metadata/inventory.json` records licence, URL, size,
SHA-256 and download status for everything actually fetched, so the provenance
of any trained model is recoverable after the fact.
