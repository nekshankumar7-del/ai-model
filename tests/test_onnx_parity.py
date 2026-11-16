"""Step 6 - the ONNX models must reproduce the Python predictions.

Loads the exported ONNX with onnxruntime and checks it against the fitted
sklearn models on 50 real samples drawn from the pipeline, not on random
noise: random vectors can agree while real ones diverge, because tree models
differ only near their split points.

    python -m pytest tests/test_onnx_parity.py -v
    python tests/test_onnx_parity.py              # same checks, plain output
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factfind.confidence import ConfidenceModel
from factfind.confidence import FEATURE_NAMES as CONF_FEATURES
from factfind.dataset import collect_many, discover_samples
from factfind.extract import Extractor
from factfind.features import build_vector
from factfind.knowledge import load_knowledge
from factfind.linker import FEATURE_NAMES as LINK_FEATURES
from factfind.linker import FieldLinker
from factfind.paths import ANGULAR_DIR, MODELS_DIR, SYNTHETIC_DIR, DATASET_DIR

EXPORT_DIR = ANGULAR_DIR / "assets" / "factfind"
N_SAMPLES = 50
# Dynamic int8 quantisation changes the arithmetic, so the comparison is
# "does it make the same decision", with a tolerance on the probability.
PROBABILITY_TOLERANCE = 0.02
DECISION_TOLERANCE = 0.0          # no sample may flip side of its threshold


def _data_dir() -> Path:
    for candidate in (DATASET_DIR, SYNTHETIC_DIR):
        if candidate.exists() and discover_samples(candidate):
            return candidate
    pytest.skip("no labelled pages available to draw parity samples from")


def _real_feature_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Collect feature vectors the pipeline actually produces."""
    kn = load_knowledge()
    samples = discover_samples(_data_dir())
    linker = None
    if (MODELS_DIR / "linker.joblib").exists():
        linker = FieldLinker.load(MODELS_DIR / "linker.joblib")

    extractor = Extractor(kn, linker=linker)
    pages = collect_many(samples[:25], kn, extractor, progress=False)

    link_rows = [build_vector(e.features, LINK_FEATURES)
                 for p in pages for e in p.linker]
    conf_rows = [build_vector(e.features, CONF_FEATURES)
                 for p in pages for e in p.confidence]

    if len(link_rows) < N_SAMPLES or len(conf_rows) < N_SAMPLES:
        pytest.skip(
            f"not enough pipeline samples for parity "
            f"({len(link_rows)} linker, {len(conf_rows)} confidence; "
            f"need {N_SAMPLES} of each)"
        )

    rng = np.random.default_rng(0)
    link_idx = rng.choice(len(link_rows), N_SAMPLES, replace=False)
    conf_idx = rng.choice(len(conf_rows), N_SAMPLES, replace=False)
    return (np.vstack([link_rows[i] for i in link_idx]),
            np.vstack([conf_rows[i] for i in conf_idx]))


def _run_onnx(path: Path, X: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.astype(np.float32)})
    # The probability tensor is the 2-D float output.
    for out in outputs:
        arr = np.asarray(out)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 1]
    # Some converters emit a sequence of dicts for probabilities.
    for out in outputs:
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return np.array([float(d[max(d)]) for d in out])
    raise AssertionError(f"could not find a probability output in {path.name}")


@pytest.fixture(scope="module")
def vectors():
    return _real_feature_vectors()


@pytest.mark.skipif(not (EXPORT_DIR / "linker.json").exists(),
                    reason="linker.json not exported (ONNX path in use)")
def test_linker_json_parity(vectors):
    """The JSON tree export must reproduce sklearn exactly.

    This is the path in use when the ONNX authoring library is unavailable.
    Unlike quantised ONNX it is lossless, so the tolerance is tight.
    """
    from factfind.export_trees import predict_from_json

    X, _ = vectors
    linker = FieldLinker.load(MODELS_DIR / "linker.joblib")
    expected = linker.model.predict_proba(X)[:, 1]
    payload = json.loads((EXPORT_DIR / "linker.json").read_text(encoding="utf-8"))
    actual = predict_from_json(payload, X)

    max_diff = float(np.max(np.abs(expected - actual)))
    assert max_diff <= 1e-9, (
        f"linker JSON differs from sklearn by up to {max_diff:.3e}"
    )


@pytest.mark.skipif(not (EXPORT_DIR / "confidence.json").exists(),
                    reason="confidence.json not exported (ONNX path in use)")
def test_confidence_json_parity(vectors):
    from factfind.export_trees import predict_from_json

    _, X = vectors
    model = ConfidenceModel.load(MODELS_DIR / "confidence.joblib")
    expected = model.model.predict_proba(X)[:, 1]
    payload = json.loads((EXPORT_DIR / "confidence.json").read_text(encoding="utf-8"))
    actual = predict_from_json(payload, X)

    max_diff = float(np.max(np.abs(expected - actual)))
    assert max_diff <= 1e-9, (
        f"confidence JSON differs from sklearn by up to {max_diff:.3e}"
    )


@pytest.mark.skipif(not (EXPORT_DIR / "linker.onnx").exists(),
                    reason="linker.onnx not exported")
def test_linker_parity(vectors):
    X, _ = vectors
    linker = FieldLinker.load(MODELS_DIR / "linker.joblib")
    expected = linker.model.predict_proba(X)[:, 1]
    actual = _run_onnx(EXPORT_DIR / "linker.onnx", X)

    max_diff = float(np.max(np.abs(expected - actual)))
    assert max_diff <= PROBABILITY_TOLERANCE, (
        f"linker ONNX differs from Python by up to {max_diff:.5f} "
        f"over {len(X)} real samples (tolerance {PROBABILITY_TOLERANCE})"
    )
    flips = int((np.round(expected) != np.round(actual)).sum())
    assert flips == 0, f"{flips} of {len(X)} samples changed class"


@pytest.mark.skipif(not (EXPORT_DIR / "confidence.onnx").exists(),
                    reason="confidence.onnx not exported")
def test_confidence_parity(vectors):
    _, X = vectors
    model = ConfidenceModel.load(MODELS_DIR / "confidence.joblib")
    expected_raw = model.model.predict_proba(X)[:, 1]
    actual_raw = _run_onnx(EXPORT_DIR / "confidence.onnx", X)

    max_diff = float(np.max(np.abs(expected_raw - actual_raw)))
    assert max_diff <= PROBABILITY_TOLERANCE, (
        f"confidence ONNX differs from Python by up to {max_diff:.5f} "
        f"over {len(X)} real samples (tolerance {PROBABILITY_TOLERANCE})"
    )


@pytest.mark.skipif(not (EXPORT_DIR / "confidence.onnx").exists(),
                    reason="confidence.onnx not exported")
def test_calibration_knots_match_python(vectors):
    """The browser applies calibration from knots; it must match isotonic."""
    spec = json.loads((EXPORT_DIR / "factfind-spec.json").read_text(encoding="utf-8"))
    knots = spec["models"]["confidence"]["calibration"]
    model = ConfidenceModel.load(MODELS_DIR / "confidence.joblib")
    if model.calibrator is None or knots is None:
        pytest.skip("no calibrator was fitted")

    _, X = vectors
    raw = model.model.predict_proba(X)[:, 1]
    expected = model.calibrator.transform(raw)
    actual = np.interp(raw, knots["x"], knots["y"])

    max_diff = float(np.max(np.abs(expected - actual)))
    assert max_diff <= 0.01, (
        f"calibration knots differ from the isotonic fit by up to {max_diff:.5f}"
    )


@pytest.mark.skipif(not (EXPORT_DIR / "factfind-spec.json").exists(),
                    reason="spec not exported")
def test_spec_feature_order_matches_code():
    """A reordered feature list silently corrupts every browser prediction."""
    spec = json.loads((EXPORT_DIR / "factfind-spec.json").read_text(encoding="utf-8"))
    assert spec["models"]["linker"]["featureNames"] == list(LINK_FEATURES)
    assert spec["models"]["confidence"]["featureNames"] == list(CONF_FEATURES)


@pytest.mark.skipif(not (EXPORT_DIR / "factfind-spec.json").exists(),
                    reason="spec not exported")
def test_spec_covers_every_field():
    kn = load_knowledge()
    spec = json.loads((EXPORT_DIR / "factfind-spec.json").read_text(encoding="utf-8"))
    spec_keys = {f["key"] for f in spec["fields"]}
    missing = set(kn.field_keys) - spec_keys
    assert not missing, f"spec is missing {len(missing)} fields: {sorted(missing)[:10]}"


@pytest.mark.skipif(not EXPORT_DIR.exists(), reason="nothing exported")
def test_bundle_within_budget():
    total = sum(p.stat().st_size for p in EXPORT_DIR.glob("*") if p.is_file())
    mb = total / (1024 * 1024)
    assert mb <= 30.0, f"browser bundle is {mb:.2f} MB, over the 30 MB budget"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
