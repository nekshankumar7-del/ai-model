"""Confidence model: "is this extracted value correct?"

A gradient-boosted classifier over recogniser confidence, candidate margins,
fuzzy scores, parser validity, cross-field checks and image quality, then
calibrated so the number it emits can be read as a probability.

Thresholds are then chosen *per kind* on the validation set as the lowest
threshold whose wrong-fill rate still meets that kind's target -- which
maximises coverage subject to the priority-1 constraint, exactly the trade the
brief asks for. The resulting coverage is reported, never traded away quietly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from .evaluate import WRONG_FILL_TARGETS
from .features import build_vector, confidence_feature_names
from .knowledge import Field

FEATURE_NAMES = confidence_feature_names()

# Never fill one of these below this probability, whatever the threshold
# search concludes. Belt and braces for the "zero silent errors" fields.
HARD_FLOOR = {"niNumber": 0.97, "email": 0.95, "phone": 0.95}


@dataclass
class ConfidenceExample:
    features: dict[str, float]
    correct: int                 # 1 = the extracted value matched the label
    kind: str
    qualified_key: str
    group: str
    stem: str = ""


class ConfidenceModel:
    """Calibrated P(correct) plus the thresholds that use it."""

    def __init__(self, model=None, calibrator=None,
                 thresholds: dict[str, float] | None = None,
                 metrics: dict[str, Any] | None = None):
        self.model = model
        self.calibrator = calibrator
        self.thresholds = thresholds or {}
        self.metrics = metrics or {}
        self.feature_names = list(FEATURE_NAMES)

    # -- training ------------------------------------------------------
    @staticmethod
    def vectorise(examples: Sequence[ConfidenceExample]):
        X = np.vstack([build_vector(e.features, FEATURE_NAMES) for e in examples])
        y = np.array([e.correct for e in examples], dtype=np.int32)
        groups = np.array([e.group for e in examples])
        kinds = np.array([e.kind for e in examples])
        return X, y, groups, kinds

    @classmethod
    def fit(cls, train: Sequence[ConfidenceExample],
            validation: Sequence[ConfidenceExample],
            seed: int = 0, verbose: bool = True) -> "ConfidenceModel":
        """Fit on train, calibrate and choose thresholds on validation."""
        Xtr, ytr, _g, _k = cls.vectorise(train)
        if len(set(ytr.tolist())) < 2:
            raise ValueError(
                "confidence model needs both correct and incorrect examples in "
                f"train; got only class {sorted(set(ytr.tolist()))}"
            )

        model = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
            min_samples_leaf=15, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=seed,
        )
        model.fit(Xtr, ytr)

        Xva, yva, _gva, kva = cls.vectorise(validation)
        raw = model.predict_proba(Xva)[:, 1]

        # Isotonic calibration on the validation split. Isotonic is chosen
        # over Platt because the raw scores pile up near 1 and we care most
        # about resolving the high end, where the fill decision happens.
        calibrator = None
        if len(set(yva.tolist())) > 1 and len(yva) >= 50:
            calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            calibrator.fit(raw, yva)
            calibrated = calibrator.transform(raw)
        else:
            calibrated = raw

        metrics: dict[str, Any] = {
            "n_train": int(len(ytr)), "n_validation": int(len(yva)),
            "train_positive_rate": float(ytr.mean()),
            "validation_positive_rate": float(yva.mean()),
        }
        if len(set(yva.tolist())) > 1:
            metrics["roc_auc_raw"] = float(roc_auc_score(yva, raw))
            metrics["roc_auc_calibrated"] = float(roc_auc_score(yva, calibrated))
            metrics["brier_raw"] = float(brier_score_loss(yva, raw))
            metrics["brier_calibrated"] = float(brier_score_loss(yva, calibrated))
            if verbose:
                print(f"  confidence: AUC={metrics['roc_auc_raw']:.4f} "
                      f"Brier {metrics['brier_raw']:.4f} -> "
                      f"{metrics['brier_calibrated']:.4f} after calibration")

        obj = cls(model=model, calibrator=calibrator, metrics=metrics)
        obj.thresholds = obj.choose_thresholds(calibrated, yva, kva, verbose=verbose)
        return obj

    # -- thresholds ----------------------------------------------------
    @staticmethod
    def choose_thresholds(probabilities: np.ndarray, correct: np.ndarray,
                          kinds: np.ndarray, verbose: bool = True) -> dict[str, float]:
        """Lowest threshold per kind that still meets that kind's target.

        Scanned from high to low: we keep lowering the bar (gaining coverage)
        for as long as the wrong-fill rate stays within target. If no
        threshold meets the target -- even 1.0 -- the kind is pinned at 1.0,
        which fills nothing, and the training script reports the miss rather
        than quietly relaxing the target.
        """
        thresholds: dict[str, float] = {}
        grid = np.round(np.arange(0.30, 1.0001, 0.005), 4)

        for kind in sorted(set(kinds.tolist())):
            target = WRONG_FILL_TARGETS.get(kind, 0.01)
            mask = kinds == kind
            p, y = probabilities[mask], correct[mask]
            chosen = 1.01            # nothing passes -> fill nothing
            best_coverage = 0.0
            for t in grid:
                filled = p >= t
                n_filled = int(filled.sum())
                if n_filled == 0:
                    continue
                wrong_rate = float((~y[filled].astype(bool)).sum()) / n_filled
                if wrong_rate <= target + 1e-12:
                    coverage = n_filled / max(1, len(y))
                    if coverage >= best_coverage:
                        best_coverage, chosen = coverage, float(t)
            floor = HARD_FLOOR.get(kind)
            if floor is not None:
                chosen = max(chosen, floor)
            thresholds[kind] = min(chosen, 1.01)
            if verbose:
                state = "no threshold meets target" if chosen > 1.0 else f"t={chosen:.3f}"
                print(f"    {kind:<10} target<={target:.3f}  {state}  "
                      f"coverage={best_coverage:.3f}")

        thresholds["default"] = max(
            [v for k, v in thresholds.items() if v <= 1.0] or [0.9]
        )
        return thresholds

    # -- inference -----------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            return np.clip(self.calibrator.transform(raw), 0.0, 1.0)
        return raw

    def predict_proba_one(self, features: dict[str, float], fld: Field) -> float:
        X = build_vector(features, FEATURE_NAMES).reshape(1, -1)
        return float(self.predict_proba(X)[0])

    def threshold_for(self, fld: Field) -> float:
        if fld.qualified_key in self.thresholds:
            return self.thresholds[fld.qualified_key]
        return float(self.thresholds.get(fld.kind, self.thresholds.get("default", 0.9)))

    # -- persistence ---------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        joblib.dump({
            "model": self.model, "calibrator": self.calibrator,
            "thresholds": self.thresholds, "metrics": self.metrics,
            "feature_names": self.feature_names,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "ConfidenceModel":
        blob = joblib.load(Path(path))
        obj = cls(model=blob["model"], calibrator=blob.get("calibrator"),
                  thresholds=blob.get("thresholds", {}),
                  metrics=blob.get("metrics", {}))
        obj.feature_names = blob.get("feature_names", list(FEATURE_NAMES))
        return obj

    def calibration_table(self, probabilities: np.ndarray, correct: np.ndarray,
                          bins: int = 10) -> list[dict[str, float]]:
        """Reliability table: predicted vs observed, for the model card."""
        edges = np.linspace(0, 1, bins + 1)
        rows = []
        for i in range(bins):
            lo, hi = edges[i], edges[i + 1]
            mask = (probabilities >= lo) & (probabilities < hi if i < bins - 1
                                            else probabilities <= hi)
            if not mask.any():
                continue
            rows.append({
                "bin_low": float(lo), "bin_high": float(hi),
                "n": int(mask.sum()),
                "mean_predicted": float(probabilities[mask].mean()),
                "observed_correct": float(correct[mask].mean()),
            })
        return rows


__all__ = ["ConfidenceModel", "ConfidenceExample", "FEATURE_NAMES", "HARD_FLOOR"]
