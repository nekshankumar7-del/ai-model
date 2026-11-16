"""Export the two small tree models as JSON instead of ONNX.

Why this exists: on this machine the Windows Application Control policy blocks
`onnx`'s native extension, so sklearn -> ONNX conversion cannot run at all
(`onnxruntime` is fine; it is only the authoring library that is blocked).

Rather than leaving the Angular side stranded, the linker and confidence
models are written as plain JSON. They are gradient-boosted trees of a few
hundred nodes: evaluating one is a walk down a binary tree, which is a dozen
lines of TypeScript and needs no runtime at all. The exported file is smaller
than the ONNX equivalent and removes a dependency from the browser bundle.

The recogniser is a different matter -- it is a real neural network and still
ships as ONNX, exported through torch, which is not blocked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def export_hist_gradient_boosting(model, path: Path) -> dict[str, Any]:
    """Serialise a fitted HistGradientBoostingClassifier.

    The raw score is `baseline + sum over trees of the leaf value reached`,
    then a logistic transform for binary classification -- the same arithmetic
    sklearn does internally.
    """
    predictors = getattr(model, "_predictors", None)
    if predictors is None:
        raise TypeError(
            f"{type(model).__name__} has no _predictors; this exporter only "
            f"handles HistGradientBoostingClassifier"
        )

    baseline = float(np.ravel(model._baseline_prediction)[0])
    trees: list[dict[str, Any]] = []

    for stage in predictors:
        for predictor in stage:
            nodes = predictor.nodes
            trees.append({
                "isLeaf": [bool(n) for n in nodes["is_leaf"]],
                "value": [float(v) for v in nodes["value"]],
                "featureIdx": [int(v) for v in nodes["feature_idx"]],
                "threshold": [float(v) for v in nodes["num_threshold"]],
                "left": [int(v) for v in nodes["left"]],
                "right": [int(v) for v in nodes["right"]],
                "missingGoToLeft": [bool(v) for v in nodes["missing_go_to_left"]],
            })

    payload = {
        "type": "histGradientBoostingBinary",
        "baseline": baseline,
        "nFeatures": int(getattr(model, "n_features_in_", 0)),
        "nTrees": len(trees),
        "link": "logistic",
        "trees": trees,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return {"path": str(path), "bytes": path.stat().st_size,
            "nTrees": len(trees), "format": "json-trees"}


def predict_from_json(payload: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Reference implementation, used to prove the export is faithful.

    The TypeScript evaluator must reproduce this exactly; the parity test
    compares both against sklearn.
    """
    raw = np.full(X.shape[0], payload["baseline"], dtype=np.float64)
    for tree in payload["trees"]:
        is_leaf = tree["isLeaf"]
        value = tree["value"]
        feature = tree["featureIdx"]
        threshold = tree["threshold"]
        left = tree["left"]
        right = tree["right"]
        missing_left = tree["missingGoToLeft"]

        for row in range(X.shape[0]):
            node = 0
            while not is_leaf[node]:
                x = X[row, feature[node]]
                if np.isnan(x):
                    node = left[node] if missing_left[node] else right[node]
                elif x <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            raw[row] += value[node]
    return 1.0 / (1.0 + np.exp(-raw))


def export_model(model, path: Path) -> dict[str, Any]:
    """Export whatever the training picked, or say plainly that it cannot."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    if isinstance(model, HistGradientBoostingClassifier):
        return export_hist_gradient_boosting(model, path)
    raise TypeError(
        f"JSON export supports HistGradientBoostingClassifier only; this "
        f"model is {type(model).__name__}. Either retrain so gradient "
        f"boosting wins model selection, or restore the ONNX path by "
        f"allowing onnx's native extension in the Application Control policy."
    )


__all__ = ["export_model", "export_hist_gradient_boosting", "predict_from_json"]
