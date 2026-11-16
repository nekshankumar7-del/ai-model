"""Export the shipped models to ONNX, plus the spec the Angular app needs.

What ships to the browser:

  det.onnx / rec.onnx / cls.onnx   text detection and recognition (RapidOCR,
                                   PP-OCRv4). These run under onnxruntime-web.
  linker.onnx                      re-ranks field candidates for a line
  confidence.onnx                  P(this extracted value is correct)
  factfind-spec.json               preprocessing, feature order, scaling,
                                   calibration knots, thresholds, charsets,
                                   and every field rule read from the
                                   knowledge file

The two small models take a fixed-length float32 vector whose names and order
come from features.py and are written into the spec, so the TypeScript side
builds byte-identical inputs.

Isotonic calibration is exported as interpolation knots rather than as an ONNX
op: it is a piecewise-linear function, the browser can apply it in three lines,
and this keeps the graph to a single tree ensemble.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .confidence import ConfidenceModel
from .features import confidence_feature_names, linker_feature_names, spec_block
from .knowledge import Knowledge, CHOICES_FROM, normalise_label
from .linker import FieldLinker
from .paths import ANGULAR_DIR, MODELS_DIR


# --------------------------------------------------------------------------
# sklearn -> ONNX
# --------------------------------------------------------------------------

def _patch_bool_attributes() -> None:
    """Make skl2onnx emit ints for boolean tree attributes.

    skl2onnx hands `nodes_missing_value_tracks_true` to onnx as a list of
    numpy bools. onnx's AttributeProto.ints rejects bools outright, so every
    tree-ensemble export fails with "Expected an int, got a boolean". Casting
    at the container boundary fixes it without touching the graph semantics
    (ONNX models the attribute as 0/1 anyway).

    Idempotent, so repeated exports in one process are safe.
    """
    from skl2onnx.common._container import ModelComponentContainer

    if getattr(ModelComponentContainer, "_factfind_bool_patch", False):
        return

    original = ModelComponentContainer.add_node

    def coerce(value):
        if isinstance(value, (bool, np.bool_)):
            return int(value)
        if isinstance(value, np.ndarray):
            if value.dtype == bool or value.dtype == object:
                return [int(v) for v in value.ravel()]
            return value
        if isinstance(value, (list, tuple)) and value:
            # A single bool anywhere in the list poisons the whole attribute,
            # so rewrite every element rather than only the bools.
            if any(isinstance(v, (bool, np.bool_)) for v in value):
                return [int(v) for v in value]
        return value

    def add_node(self, op_type, inputs, outputs, op_domain="", op_version=None,
                 name=None, **attrs):
        attrs = {k: coerce(v) for k, v in attrs.items()}
        return original(self, op_type, inputs, outputs, op_domain=op_domain,
                        op_version=op_version, name=name, **attrs)

    ModelComponentContainer.add_node = add_node
    ModelComponentContainer._factfind_bool_patch = True


def onnx_authoring_available() -> tuple[bool, str]:
    """Can we convert sklearn models to ONNX on this machine?

    `onnxruntime` (running models) and `onnx` (authoring them) are separate
    packages with separate native extensions, and a security policy can block
    one without the other. Checking explicitly turns an obscure DLL error into
    a clear message and a working fallback.
    """
    try:
        import onnx  # noqa: F401
        import skl2onnx  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def export_sklearn(model, n_features: int, path: Path,
                   name: str = "model") -> dict[str, Any]:
    """Convert a fitted sklearn classifier to ONNX with a float32 input."""
    _patch_bool_attributes()
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    initial_types = [("features", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(
        model, initial_types=initial_types, target_opset=17,
        options={id(model): {"zipmap": False}},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    return {"name": name, "path": str(path), "bytes": path.stat().st_size,
            "n_features": n_features}


def quantise(src: Path, dst: Path) -> dict[str, Any]:
    """Dynamic int8 quantisation. Falls back to a copy if it is not supported
    for this graph, which is reported rather than hidden."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    try:
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QUInt8)
        return {"quantised": True, "bytes": dst.stat().st_size}
    except Exception as exc:
        shutil.copyfile(src, dst)
        return {"quantised": False, "reason": f"{type(exc).__name__}: {exc}",
                "bytes": dst.stat().st_size}


def copy_ocr_models(dst_dir: Path) -> list[dict[str, Any]]:
    """Copy the RapidOCR ONNX models so the browser bundle is self-contained."""
    import rapidocr_onnxruntime

    src_dir = Path(rapidocr_onnxruntime.__file__).parent / "models"
    wanted = {
        "det.onnx": "ch_PP-OCRv4_det_infer.onnx",
        "rec.onnx": "ch_PP-OCRv4_rec_infer.onnx",
        "cls.onnx": "ch_ppocr_mobile_v2.0_cls_infer.onnx",
    }
    out = []
    dst_dir.mkdir(parents=True, exist_ok=True)
    for dst_name, src_name in wanted.items():
        src = src_dir / src_name
        if not src.exists():
            out.append({"name": dst_name, "missing": str(src)})
            continue
        dst = dst_dir / dst_name
        shutil.copyfile(src, dst)
        out.append({"name": dst_name, "source": src_name,
                    "bytes": dst.stat().st_size})
    return out


# --------------------------------------------------------------------------
# Calibration knots
# --------------------------------------------------------------------------

def isotonic_knots(calibrator, fallback_points: int = 512) -> dict[str, Any] | None:
    """Export an IsotonicRegression as (x, y) knots for linear interpolation.

    sklearn's IsotonicRegression is itself a piecewise-linear interpolator over
    `X_thresholds_` / `y_thresholds_`, so exporting those breakpoints
    reproduces it exactly. Sampling a uniform grid instead would round off
    every step and leave the browser disagreeing with Python by a couple of
    points of probability -- enough to flip a fill decision near a threshold.
    """
    if calibrator is None:
        return None

    xs = getattr(calibrator, "X_thresholds_", None)
    ys = getattr(calibrator, "y_thresholds_", None)
    if xs is not None and ys is not None and len(xs) == len(ys) and len(xs) > 1:
        return {
            "x": [float(v) for v in np.asarray(xs, dtype=float)],
            "y": [float(v) for v in np.asarray(ys, dtype=float)],
            "method": "isotonic-breakpoints",
        }

    # Older/other calibrators: fall back to a dense grid and say so.
    grid = np.linspace(0.0, 1.0, fallback_points)
    try:
        values = calibrator.transform(grid)
    except Exception:
        return None
    return {"x": [float(v) for v in grid], "y": [float(v) for v in values],
            "method": "sampled-grid"}


# --------------------------------------------------------------------------
# The Angular spec
# --------------------------------------------------------------------------

def field_spec(kn: Knowledge) -> list[dict[str, Any]]:
    """Every field rule the browser needs, read from the knowledge file."""
    out = []
    for fld in kn.fields:
        entry: dict[str, Any] = {
            "key": fld.qualified_key,
            "section": fld.section_id,
            "field": fld.key,
            "label": fld.label,
            "kind": fld.kind,
            "aliases": list(fld.aliases),
            "labelForms": fld.label_forms(),
            "required": fld.required,
        }
        if fld.when:
            entry["when"] = fld.when
        if fld.only_if:
            entry["onlyIf"] = fld.only_if
        if fld.minimum is not None:
            entry["min"] = fld.minimum
        if fld.maximum is not None:
            entry["max"] = fld.maximum
        if fld.min_age is not None:
            entry["minAge"] = fld.min_age
        if fld.vocabulary:
            entry["vocabulary"] = list(fld.vocabulary)
        if fld.allow_other:
            entry["allowOther"] = True
        choices = fld.resolved_choices()
        if choices:
            entry["choices"] = [
                {"value": c.value, "label": c.label,
                 "surfaceForms": c.all_surface_forms()}
                for c in choices
            ]
        out.append(entry)
    return out


# Character sets the browser constrains recognition with, per kind.
CHARSETS = {
    "date": "0123456789/-. ",
    "number": "0123456789.,",
    "money": "0123456789.,£$€k ",
    "phone": "0123456789 +()-",
    "niNumber": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
    "email": "abcdefghijklmnopqrstuvwxyz0123456789@._-+",
}


def build_spec(kn: Knowledge, linker: FieldLinker | None,
               confidence: ConfidenceModel | None,
               artefacts: dict[str, Any],
               metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    from . import __version__
    from .evaluate import WRONG_FILL_TARGETS
    from .parsers import BLANK_TOKENS, NO_FORMS, YES_FORMS
    from .segment import SEP_PATTERNS

    thresholds = dict(confidence.thresholds) if confidence else {}

    return {
        "specVersion": 1,
        "pipelineVersion": __version__,
        "knowledgeVersion": kn.version,
        "generated": artefacts.get("generated"),
        "preprocessing": {
            "maxSide": 2200,
            "minSide": 600,
            "steps": ["pageCrop", "perspectiveWarp", "deskew",
                      "flattenIllumination", "removeRuledLines",
                      "adaptiveThreshold"],
            "adaptiveThreshold": {"blockSize": 31, "C": 12, "method": "gaussian"},
            "deskew": {"maxAngleDegrees": 15.0, "minAngleDegrees": 0.25},
            "ruledLineRemoval": {"horizontalDivisor": 14, "verticalDivisor": 14},
        },
        "ocr": {
            "engine": "PP-OCRv4 (RapidOCR ONNX)",
            "models": {"detection": "det.onnx", "recognition": "rec.onnx",
                       "classification": "cls.onnx"},
            "textScoreFloor": 0.15,
            "lineMerge": {"yTolerance": 0.6, "xGapTextHeights": 2.5},
        },
        "segmentation": {
            "separators": [name for _pattern, name in SEP_PATTERNS],
            "maxLabelWords": 6,
        },
        "matching": {
            "labelScoreFloor": 0.45,
            "assignMinScore": 0.42,
            "weights": {"label": 0.56, "parse": 0.34, "split": 0.10},
            "sectionBonus": {"sameSection": 0.12, "defaultSection": 0.0,
                             "otherSection": -0.10, "suppressed": -1.0},
            "invalidParsePenalty": 0.45,
        },
        "parsing": {
            "blankTokens": sorted(BLANK_TOKENS),
            "yesForms": sorted(YES_FORMS),
            "noForms": sorted(NO_FORMS),
            "charsets": CHARSETS,
            "niNumberPattern": "^(?!BG|GB|NK|KN|TN|NT|ZZ)[ABCEGHJ-PRSTW-Z][ABCEGHJ-NPRSTW-Z]\\d{6}[A-D]$",
            "emailPattern": "^[A-Za-z0-9._%+\\-]+@[A-Za-z0-9.\\-]+\\.[A-Za-z]{2,24}$",
            "ukPhone": {"digits": 11, "leadingZero": True, "mobilePrefix": "07"},
            "choiceSnapThreshold": 0.86,
            "choiceMarginThreshold": 0.08,
            "vocabularySnapThreshold": 0.92,
        },
        "features": spec_block(),
        "models": {
            "linker": {
                "file": "linker.onnx" if onnx_authoring_available()[0] else "linker.json",
                "kept": linker is not None,
                "family": linker.model_name if linker else None,
                "blend": linker.blend if linker else None,
                "featureNames": linker_feature_names(),
                "inputName": "features",
                "outputName": "probabilities",
                "positiveClassIndex": 1,
            },
            "confidence": {
                "file": "confidence.onnx" if onnx_authoring_available()[0] else "confidence.json",
                "kept": confidence is not None,
                "featureNames": confidence_feature_names(),
                "inputName": "features",
                "outputName": "probabilities",
                "positiveClassIndex": 1,
                "calibration": isotonic_knots(confidence.calibrator) if confidence else None,
            },
        },
        "decision": {
            "thresholds": thresholds,
            "hardFloors": {"niNumber": 0.97, "email": 0.95, "phone": 0.95},
            "statuses": ["filled", "unsure", "missing"],
            "unsureTopK": 3,
            "note": ("fill only when the calibrated probability reaches the "
                     "threshold for that field's kind; everything else is "
                     "returned as unsure with its top 3 candidates"),
        },
        "targets": {"wrongFillByKind": WRONG_FILL_TARGETS},
        "choicesFrom": CHOICES_FROM,
        "fields": field_spec(kn),
        "sections": [
            {"id": s.id, "title": s.title, "headings": list(s.headings),
             "default": s.is_default, "when": s.when, "save": s.save,
             "repeating": s.is_repeating}
            for s in kn.sections
        ],
        "otherHeadings": list(kn.other_headings),
        "artefacts": artefacts,
        "metrics": metrics or {},
    }


# --------------------------------------------------------------------------
# Top level
# --------------------------------------------------------------------------

@dataclass
class ExportResult:
    out_dir: Path
    spec_path: Path
    artefacts: dict[str, Any]
    total_bytes: int

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)


def export_all(kn: Knowledge, out_dir: Path | str | None = None,
               models_dir: Path | str | None = None,
               include_ocr: bool = True,
               metrics: dict[str, Any] | None = None) -> ExportResult:
    from datetime import datetime, timezone

    out_dir = Path(out_dir or (ANGULAR_DIR / "assets" / "factfind"))
    models_dir = Path(models_dir or MODELS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    artefacts: dict[str, Any] = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [],
    }

    can_onnx, onnx_error = onnx_authoring_available()
    if not can_onnx:
        artefacts["onnxAuthoringBlocked"] = onnx_error
        print("  ONNX authoring is unavailable, falling back to JSON trees.")
        print(f"    reason: {onnx_error}")

    def export_one(model, n_features: int, name: str) -> dict[str, Any]:
        """ONNX when we can author it, JSON trees when we cannot."""
        if can_onnx:
            raw = out_dir / f"_{name}.raw.onnx"
            info = export_sklearn(model, n_features, raw, name)
            q = quantise(raw, out_dir / f"{name}.onnx")
            raw.unlink(missing_ok=True)
            return {**info, "name": f"{name}.onnx", "format": "onnx", **q}
        from .export_trees import export_model
        info = export_model(model, out_dir / f"{name}.json")
        return {**info, "name": f"{name}.json"}

    linker = None
    linker_path = models_dir / "linker.joblib"
    if linker_path.exists():
        linker = FieldLinker.load(linker_path)
        artefacts["files"].append(
            export_one(linker.model, len(linker_feature_names()), "linker"))

    confidence = None
    conf_path = models_dir / "confidence.joblib"
    if conf_path.exists():
        confidence = ConfidenceModel.load(conf_path)
        artefacts["files"].append(
            export_one(confidence.model, len(confidence_feature_names()),
                       "confidence"))

    if include_ocr:
        artefacts["ocr"] = copy_ocr_models(out_dir)

    spec = build_spec(kn, linker, confidence, artefacts, metrics=metrics)
    spec_path = out_dir / "factfind-spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")

    total = sum(p.stat().st_size for p in out_dir.glob("*") if p.is_file())
    artefacts["totalBytes"] = total
    spec["artefacts"]["totalBytes"] = total
    spec_path.write_text(json.dumps(spec, indent=2, default=str), encoding="utf-8")

    return ExportResult(out_dir=out_dir, spec_path=spec_path,
                        artefacts=artefacts, total_bytes=total)


__all__ = ["export_all", "export_sklearn", "quantise", "build_spec",
           "copy_ocr_models", "isotonic_knots", "ExportResult"]
