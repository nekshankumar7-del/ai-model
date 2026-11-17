"""Per-dataset converters into the unified manifest.

Each adapter is ``(spec, extracted_dir, limit=None) -> Iterable[Sample]``. It
owns exactly one job: understand one corpus's own layout and emit rows that
look like every other corpus's rows. Nothing else in the pipeline knows that
EMNIST is an IDX file, that NIST SD19 encodes the class in a directory name,
or that IIIT ships a space-separated label file.

Adapters exist for datasets that cannot be downloaded here too (IAM, GNHK,
CENSUS-HWR, HP Labs). Obtaining the archive by hand and dropping it into
``data/raw/<id>/`` is then the only manual step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

from ..manifest import Sample
from ..registry import DatasetSpec
from .generic import folder_labels_adapter
from .indic import iiit_words_adapter, uci_dhcd_adapter
from .latin import emnist_adapter, gnhk_adapter, iam_adapter, nist_sd19_adapter

Adapter = Callable[[DatasetSpec, Path], Iterable[Sample]]

ADAPTERS: dict[str, Adapter] = {
    "emnist": emnist_adapter,
    "nist_sd19": nist_sd19_adapter,
    "iam": iam_adapter,
    "gnhk": gnhk_adapter,
    "uci_dhcd": uci_dhcd_adapter,
    "iiit_words": iiit_words_adapter,
    "folder_labels": folder_labels_adapter,
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise KeyError(f"no adapter named {name!r}; have: "
                       + ", ".join(sorted(ADAPTERS))) from None


__all__ = ["ADAPTERS", "get_adapter", "Adapter"]
