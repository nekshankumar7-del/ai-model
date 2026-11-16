"""The feature vectors the learned models consume.

Both shipped models take a fixed-length vector of named floats. The names and
their order are written into the Angular spec, so the TypeScript side builds
byte-identical inputs. Nothing here depends on a Python-only library at
inference time -- every feature is arithmetic over values the browser pipeline
already has.

Two vectors:

  linker      - scores one (line, field) hypothesis: "is this the right field
                for this line?"
  confidence  - scores an assigned value: "is this extracted value correct?"
                It is the linker vector plus cross-field checks and image
                quality.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .knowledge import SUPPORTED_KINDS

# Stable orders. Appending is safe; reordering breaks every exported model.
KIND_ORDER: list[str] = sorted(SUPPORTED_KINDS)
SEPARATOR_ORDER: list[str] = [
    "prefix",
    "tight-dash",
    "colon", "leader", "dash", "arrow", "gap", "words",
    "narrative", "narrative-guess", "above", "below",
]

BASE_FEATURES: list[str] = [
    "label_score", "label_margin", "parse_score", "parse_valid",
    "split_confidence", "ocr_confidence", "section_bonus", "combined_score",
    "rel_x", "rel_y", "rel_w", "rel_h",
    "label_len", "value_len", "line_len",
    "n_candidates", "candidate_margin", "is_sensitive",
    "value_has_digit", "value_has_alpha",
]

CROSS_FIELD_FEATURES: list[str] = [
    "flag_retirement_before_age", "flag_retirement_gap",
    "flag_dependant_older_than_client", "flag_dependant_age",
    "flag_relationship_age_conflict", "flag_term_implausible",
    "flag_surname_shared", "flag_money_ratio_to_median",
    "flag_only_if_violated",
]

IMAGE_FEATURES: list[str] = [
    "img_blur_score", "img_contrast", "img_brightness", "img_ink_ratio",
    "img_skew_angle", "img_page_found", "img_shadow_spread",
]

# Scaling applied before the model sees a feature, so a blur variance of 900
# and a probability of 0.9 are on comparable ground. Browser side applies the
# same transform from the spec.
LOG_SCALED = {"label_len", "value_len", "line_len", "img_blur_score",
              "img_contrast", "img_shadow_spread", "flag_dependant_age",
              "flag_money_ratio_to_median", "flag_retirement_gap"}
DIV_255 = {"img_brightness"}


def linker_feature_names() -> list[str]:
    return (BASE_FEATURES
            + [f"kind_{k}" for k in KIND_ORDER]
            + [f"sep_{s}" for s in SEPARATOR_ORDER]
            + ["same_section", "is_default_section", "is_other_section"])


def confidence_feature_names() -> list[str]:
    return linker_feature_names() + ["linker_score"] + CROSS_FIELD_FEATURES + IMAGE_FEATURES


def _scale(name: str, value: float) -> float:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return 0.0
    value = float(value)
    if name in LOG_SCALED:
        return float(np.log1p(abs(value)) * (1.0 if value >= 0 else -1.0))
    if name in DIV_255:
        return value / 255.0
    return value


def build_vector(raw: dict[str, Any], names: Iterable[str]) -> np.ndarray:
    """Project a loose feature dict onto the fixed, scaled vector."""
    return np.array([_scale(n, raw.get(n, 0.0)) for n in names], dtype=np.float32)


def linker_features(match, default_section_id: str) -> dict[str, float]:
    """Feature dict for one (line, field) hypothesis."""
    raw = dict(match.as_features())
    for k in KIND_ORDER:
        raw[f"kind_{k}"] = float(match.field.kind == k)
    for s in SEPARATOR_ORDER:
        raw[f"sep_{s}"] = float(match.pair.separator == s)
    raw["same_section"] = float(match.field.section_id == match.line.section_id)
    raw["is_default_section"] = float(match.field.section_id == default_section_id)
    raw["is_other_section"] = float(match.line.section_id == "__other__")
    return raw


def confidence_features(match, default_section_id: str,
                        linker_score: float,
                        flags: dict[str, float] | None = None,
                        image: dict[str, float] | None = None) -> dict[str, float]:
    raw = linker_features(match, default_section_id)
    raw["linker_score"] = float(linker_score)
    for name in CROSS_FIELD_FEATURES:
        raw[name] = float((flags or {}).get(name[len("flag_"):], 0.0))
    for name in IMAGE_FEATURES:
        raw[name] = float((image or {}).get(name[len("img_"):], 0.0))
    return raw


def spec_block() -> dict[str, Any]:
    """The part of the Angular spec that describes these vectors."""
    return {
        "linker": {
            "featureNames": linker_feature_names(),
            "kindOrder": KIND_ORDER,
            "separatorOrder": SEPARATOR_ORDER,
        },
        "confidence": {"featureNames": confidence_feature_names()},
        "scaling": {
            "log1pSigned": sorted(LOG_SCALED),
            "divide255": sorted(DIV_255),
            "note": "every other feature is passed through unchanged",
        },
    }


__all__ = ["linker_feature_names", "confidence_feature_names", "build_vector",
           "linker_features", "confidence_features", "spec_block",
           "KIND_ORDER", "SEPARATOR_ORDER"]
