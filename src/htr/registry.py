"""The dataset inventory: what exists, where it comes from, and on what terms.

Every entry records a licence and an access mode. Nothing here bypasses a
registration wall, a login or a licence restriction: datasets that require
human agreement are registered with ``access="manual"``, download page and
all, so the pipeline can say exactly what to fetch and where to drop it, and
then carry on with everything that is genuinely open.

Sizes are the Content-Length the host reported when this registry was last
verified (2026-11-17). They are a sanity check on a finished download, not a
substitute for a checksum.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Access = Literal["auto", "manual", "unavailable"]
Level = Literal["char", "word", "line", "page"]


@dataclass(frozen=True)
class Resource:
    """One downloadable file belonging to a dataset."""
    url: str
    filename: str
    bytes_expected: int | None = None
    sha256: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    name: str
    homepage: str
    licence: str
    licence_url: str
    access: Access
    level: Level
    scripts: tuple[str, ...]
    languages: tuple[str, ...]
    adapter: str
    resources: tuple[Resource, ...] = ()
    approx_samples: int | None = None
    default: bool = False
    commercial_use: bool | None = None
    notes: str = ""
    manual_instructions: str = ""
    # False when the adapter reads the downloaded archive directly, so there
    # is nothing to gain from unpacking hundreds of thousands of files first.
    extract: bool = True
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_bytes(self) -> int:
        return sum(r.bytes_expected or 0 for r in self.resources if not r.optional)


_CVIT = "https://cvit.iiit.ac.in/research/projects/cvit-projects/indic-hw-data"
_CVIT_WORDS = "https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words"
_CVIT_TERMS = ("CVIT, IIIT Hyderabad - released for research use; cite the "
               "IIIT-INDIC-HW-WORDS / IIIT-HW-Dev papers. No redistribution.")

# --------------------------------------------------------------------------
# English / Latin
# --------------------------------------------------------------------------

EMNIST = DatasetSpec(
    id="emnist",
    name="EMNIST (Extended MNIST)",
    homepage="https://www.nist.gov/itl/products-and-services/emnist-dataset",
    licence="US Government work (NIST) - free to use, citation requested",
    licence_url="https://www.nist.gov/itl/products-and-services/emnist-dataset",
    access="auto",
    level="char",
    scripts=("Latin",),
    languages=("en",),
    adapter="emnist",
    resources=(Resource(
        url="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip",
        filename="gzip.zip",
        bytes_expected=561753746,
    ),),
    approx_samples=814255,
    default=True,
    commercial_use=True,
    notes=("The ByClass split is used: 62 classes (0-9, A-Z, a-z), 814k "
           "images, the only EMNIST split that keeps case apart."),
    tags=("digits", "letters"),
)

NIST_SD19 = DatasetSpec(
    id="nist_sd19",
    name="NIST Special Database 19 (by writer)",
    homepage="https://www.nist.gov/srd/nist-special-database-19",
    licence="US Government work (NIST) - public domain in the United States",
    licence_url="https://www.nist.gov/srd/nist-special-database-19",
    access="auto",
    level="char",
    scripts=("Latin",),
    languages=("en",),
    adapter="nist_sd19",
    resources=(
        Resource(url="https://s3.amazonaws.com/nist-srd/SD19/by_class.zip",
                 filename="by_class.zip", bytes_expected=1031576378),
        Resource(url="https://s3.amazonaws.com/nist-srd/SD19/by_write.zip",
                 filename="by_write.zip", bytes_expected=568113446,
                 optional=True),
    ),
    approx_samples=814255,
    default=True,
    commercial_use=True,
    notes=("by_class carries the labels, in the directory name as a hex ASCII "
           "code. by_write is the partition with writer identity, but it was "
           "checked and contains 814,255 PNGs and NOTHING else - no .mit "
           "files, no transcriptions - and its filenames (c1000_45_00000.png) "
           "cannot be joined to by_class's (hsf_6_00156.png). So the two "
           "cannot be combined, and by_write alone is unlabelled. It stays "
           "registered as optional for anyone who wants the writer grouping "
           "for unsupervised work. These are the same images EMNIST is built "
           "from, at their original 128x128 rather than downsampled to 28x28, "
           "which is what makes them worth having alongside it."),
    extract=False,
    tags=("digits", "letters", "full-resolution"),
)

IAM = DatasetSpec(
    id="iam",
    name="IAM Handwriting Database 3.0",
    homepage="https://fki.tic.heia-fr.ch/databases/iam-handwriting-database",
    licence="CC BY-NC 4.0 - non-commercial research only",
    licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
    access="manual",
    level="line",
    scripts=("Latin",),
    languages=("en",),
    adapter="iam",
    approx_samples=115320,
    default=False,
    commercial_use=False,
    notes=("Requires a free registered account. The host did not respond from "
           "this network when the registry was verified. Its NC licence "
           "forbids commercial use - which matters for a commercial firm."),
    manual_instructions=(
        "1. Register at https://fki.tic.heia-fr.ch/register\n"
        "2. Download lines.tgz, words.tgz and ascii.tgz from the IAM page\n"
        "3. Put them in data/raw/iam/ and re-run the build step.\n"
        "Do not use third-party mirrors: they redistribute data whose own "
        "terms require registration with the original provider."),
    tags=("lines", "words", "writer-ids"),
)

IAM_HF = DatasetSpec(
    id="iam_hf",
    name="IAM lines via a Hugging Face mirror (reference use only)",
    homepage="https://huggingface.co/datasets/Teklia/IAM-line",
    licence="CC BY-NC 4.0 - non-commercial research only",
    licence_url="https://creativecommons.org/licenses/by-nc/4.0/",
    access="manual",
    level="line",
    scripts=("Latin",),
    languages=("en",),
    adapter="folder_labels",
    approx_samples=10373,
    default=False,
    commercial_use=False,
    notes=("A third-party mirror, fetched by scripts/download_iam_handwriting.py "
           "at the user's explicit request, for benchmarking against published "
           "IAM numbers. The canonical `iam` entry above still points at the "
           "original provider, which is where anything that ships must come "
           "from - and even then the NC licence rules out commercial use. "
           "Ships IAM's own writer-disjoint train/validation/test split, which "
           "the splitter honours rather than re-cutting."),
    manual_instructions=(
        "python scripts/download_iam_handwriting.py\n"
        "then: python scripts/11_build_manifests.py --datasets iam_hf"),
    tags=("lines", "reference-only", "non-commercial"),
)

GNHK = DatasetSpec(
    id="gnhk",
    name="GNHK - GoodNotes Handwriting Kollection",
    homepage="https://github.com/GoodNotes/GNHK-dataset",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    access="manual",
    level="word",
    scripts=("Latin",),
    languages=("en",),
    adapter="gnhk",
    approx_samples=39000,
    default=False,
    commercial_use=True,
    notes=("Openly licensed, but the download behind goodnotes.com/gnhk no "
           "longer serves the data and static.gnhk-dataset.net does not "
           "resolve. Request it from the paper's authors."),
    manual_instructions=(
        "Contact the authors listed in https://github.com/GoodNotes/GNHK-dataset\n"
        "Put train_data.zip / test_data.zip in data/raw/gnhk/ and re-run."),
    tags=("in-the-wild", "camera", "paragraphs"),
)

CENSUS_HWR = DatasetSpec(
    id="census_hwr",
    name="CENSUS-HWR",
    homepage="https://arxiv.org/abs/2305.16275",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    access="manual",
    level="word",
    scripts=("Latin",),
    languages=("en",),
    adapter="folder_labels",
    approx_samples=1812014,
    default=False,
    commercial_use=True,
    notes=("1.8M English handwritten word images from the 1930/1940 US "
           "census - the best openly licensed English word corpus found. The "
           "paper points at censustree.org, which no longer exposes a "
           "download link. Worth re-checking."),
    manual_instructions=(
        "Check https://censustree.org/data for a current download link.\n"
        "Unpack into data/extracted/census_hwr/ as <label>/<image>.png, or "
        "as an images/ folder beside a labels.txt."),
    tags=("words", "historical"),
)

# --------------------------------------------------------------------------
# Indic
# --------------------------------------------------------------------------

UCI_DHCD = DatasetSpec(
    id="uci_dhcd",
    name="Devanagari Handwritten Character Dataset (UCI)",
    homepage="https://archive.ics.uci.edu/dataset/389/devanagari+handwritten+character+dataset",
    licence="CC BY 4.0",
    licence_url="https://creativecommons.org/licenses/by/4.0/",
    access="auto",
    level="char",
    scripts=("Devanagari",),
    languages=("hi", "ne"),
    adapter="uci_dhcd",
    resources=(Resource(
        url=("https://archive.ics.uci.edu/static/public/389/"
             "devanagari+handwritten+character+dataset.zip"),
        filename="dhcd.zip",
    ),),
    approx_samples=92000,
    default=False,
    commercial_use=True,
    notes="46 classes: 36 consonants and 10 digits, 32x32 PNG, 2000 each.",
    tags=("digits", "consonants"),
)

IIIT_HW_DEV = DatasetSpec(
    id="iiit_hw_dev",
    name="IIIT-HW-Dev / IIIT-HW-Hindi v1 (Devanagari words)",
    homepage=_CVIT,
    licence=_CVIT_TERMS,
    licence_url=_CVIT,
    access="auto",
    level="word",
    scripts=("Devanagari",),
    languages=("hi",),
    adapter="iiit_words",
    resources=(Resource(
        url=("https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/"
             "IIIT-HW-Hindi_v1.tar.gz"),
        filename="IIIT-HW-Hindi_v1.tar.gz", bytes_expected=1822862305,
    ),),
    approx_samples=95381,
    default=False,
    commercial_use=None,
    notes=("Word-level Devanagari with writer identity - the most useful "
           "Indic set here, and the reason the word config exists."),
    tags=("words", "writer-ids"),
)

IIIT_HW_TELUGU = DatasetSpec(
    id="iiit_hw_telugu",
    name="IIIT-HW-Telugu v1",
    homepage=_CVIT,
    licence=_CVIT_TERMS,
    licence_url=_CVIT,
    access="auto",
    level="word",
    scripts=("Telugu",),
    languages=("te",),
    adapter="iiit_words",
    resources=(Resource(
        url=("https://cvit.iiit.ac.in/images/Projects/wordlevel-Indicscripts/"
             "IIIT-HW-Telugu_v1.tar.gz"),
        filename="IIIT-HW-Telugu_v1.tar.gz", bytes_expected=3730676516,
    ),),
    approx_samples=120000,
    default=False,
    commercial_use=None,
    tags=("words", "writer-ids"),
)

_INDIC_WORDS = {
    "bengal": ("iiit_indic_bn", "Bengali", "bn", 3366779345),
    "gu": ("iiit_indic_gu", "Gujarati", "gu", 3985465748),
    "kn": ("iiit_indic_kn", "Kannada", "kn", 5185781771),
    "ma": ("iiit_indic_ml", "Malayalam", "ml", 6039159032),
    "od": ("iiit_indic_or", "Odia", "or", 2867961480),
    "pn": ("iiit_indic_pa", "Gurmukhi", "pa", 3574169363),
    "ta": ("iiit_indic_ta", "Tamil", "ta", 5121780332),
    "ur": ("iiit_indic_ur", "Arabic", "ur", 2627935586),
}

INDIC_WORD_SETS: list[DatasetSpec] = [
    DatasetSpec(
        id=dsid,
        name="IIIT-INDIC-HW-WORDS (" + script + ")",
        homepage=_CVIT_WORDS,
        licence=_CVIT_TERMS,
        licence_url=_CVIT_WORDS,
        access="auto",
        level="word",
        scripts=(script,),
        languages=(lang,),
        adapter="iiit_words",
        resources=(Resource(
            url=("https://cvit.iiit.ac.in/images/Projects/iiit-indic-hw-words/"
                 + code + ".zip"),
            filename=code + ".zip", bytes_expected=size,
        ),),
        approx_samples=100000,
        default=False,
        commercial_use=None,
        notes="Part of the 872k-instance, 135-writer IIIT-INDIC-HW-WORDS release.",
        tags=("words", "writer-ids"),
    )
    for code, (dsid, script, lang, size) in _INDIC_WORDS.items()
]

HP_LABS_INDIC = DatasetSpec(
    id="hp_labs_indic",
    name="HP Labs India isolated Indic characters (LipiTk)",
    homepage="http://lipitk.sourceforge.net/datasets/dvngchardata.htm",
    licence="HP Labs India research licence - see the LipiTk dataset page",
    licence_url="http://lipitk.sourceforge.net/datasets/dvngchardata.htm",
    access="unavailable",
    level="char",
    scripts=("Devanagari", "Tamil", "Telugu", "Kannada"),
    languages=("hi", "ta", "te", "kn"),
    adapter="folder_labels",
    approx_samples=27000,
    default=False,
    commercial_use=None,
    notes=("The LipiTk index page is still up, but every archive it links to "
           "lives on mile.ee.iisc.ernet.in, which returns 404 (502 on the "
           ".ac.in successor). The data is gone from its official host; no "
           "mirror was substituted."),
    manual_instructions=(
        "If you obtain hpl-devnagari-iso-char-offline-1.0.rar from the "
        "authors, unpack it into data/extracted/hp_labs_indic/ as "
        "<class>/<image>.tiff and re-run the build step."),
    tags=("isolated-characters",),
)

# --------------------------------------------------------------------------

ALL: list[DatasetSpec] = [
    EMNIST, NIST_SD19, IAM, IAM_HF, GNHK, CENSUS_HWR,
    UCI_DHCD, IIIT_HW_DEV, IIIT_HW_TELUGU, *INDIC_WORD_SETS, HP_LABS_INDIC,
]

BY_ID: dict[str, DatasetSpec] = {d.id: d for d in ALL}


def get(dataset_id: str) -> DatasetSpec:
    try:
        return BY_ID[dataset_id]
    except KeyError:
        known = ", ".join(sorted(BY_ID))
        raise KeyError("unknown dataset " + repr(dataset_id)
                       + "; known: " + known) from None


def select(ids: list[str] | None = None, *, level: str | None = None,
           access: str | None = None, defaults_only: bool = False
           ) -> list[DatasetSpec]:
    """Resolve a selection. 'all' means every registered dataset."""
    if ids and list(ids) != ["all"]:
        chosen = [get(i) for i in ids]
    else:
        chosen = list(ALL)
    if defaults_only:
        chosen = [d for d in chosen if d.default]
    if level:
        chosen = [d for d in chosen if d.level == level]
    if access:
        chosen = [d for d in chosen if d.access == access]
    return chosen


def summary_rows() -> list[dict]:
    return [{
        "id": d.id, "name": d.name, "access": d.access, "level": d.level,
        "scripts": list(d.scripts), "languages": list(d.languages),
        "licence": d.licence, "licence_url": d.licence_url,
        "homepage": d.homepage, "default": d.default,
        "commercial_use": d.commercial_use,
        "approx_samples": d.approx_samples,
        "bytes_expected": d.total_bytes or None,
        "urls": [r.url for r in d.resources],
        "notes": d.notes,
    } for d in ALL]


__all__ = ["DatasetSpec", "Resource", "ALL", "BY_ID", "get", "select",
           "summary_rows"]
