"""Layout / field linker.

The brief asks for a model that predicts the field key for each line. This is
built as a *re-ranker over candidates* rather than a flat 107-way classifier,
for a concrete reason: the knowledge file already tells us each field's
labels, so the rule matcher can narrow 107 fields to a handful per line with
near-perfect recall. A 107-way softmax would have to relearn that mapping from
scratch, needs examples of every class, and collapses whenever you add a field
to the knowledge file -- which the file is explicitly designed to let you do
without retraining.

So the model answers "is THIS field the right one for THIS line?" over the
candidates, from features that are all computable in the browser. Adding a
field to the knowledge file changes the candidate set, not the model.

Three model families are tried and the best by grouped cross-validation wins:
logistic regression, linear SVM (Platt-scaled), gradient boosting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .features import build_vector, linker_feature_names, linker_features

FEATURE_NAMES = linker_feature_names()


def build_model(name: str, seed: int = 0):
    """The candidate model families named in the brief."""
    if name == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0,
                                       class_weight="balanced",
                                       random_state=seed)),
        ])
    if name == "linear_svm":
        # LinearSVC has no predict_proba; Platt-scale it so the output is a
        # probability we can threshold and calibrate like the others.
        return Pipeline([
            ("scale", StandardScaler()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=seed,
                          dual="auto", max_iter=5000),
                method="sigmoid", cv=3)),
        ])
    if name == "gbdt":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.07, max_leaf_nodes=31,
            min_samples_leaf=20, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15,
            random_state=seed,
        )
    raise ValueError(f"unknown linker model {name!r}")


MODEL_NAMES = ["logreg", "linear_svm", "gbdt"]


@dataclass
class LinkerExample:
    features: dict[str, float]
    label: int                  # 1 = this candidate is the right field
    group: str                  # writer/client group, for grouped CV
    stem: str = ""
    qualified_key: str = ""


class FieldLinker:
    """Scores (line, field) hypotheses and re-ranks the rule candidates."""

    def __init__(self, model=None, model_name: str = "logreg",
                 blend: float = 0.65, metrics: dict[str, Any] | None = None):
        self.model = model
        self.model_name = model_name
        # How much weight the model's opinion carries against the rule score.
        # Chosen on validation, not guessed at inference time.
        self.blend = blend
        self.metrics = metrics or {}
        self.feature_names = list(FEATURE_NAMES)

    # -- training ------------------------------------------------------
    @staticmethod
    def vectorise(examples: Sequence[LinkerExample]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.vstack([build_vector(e.features, FEATURE_NAMES) for e in examples])
        y = np.array([e.label for e in examples], dtype=np.int32)
        groups = np.array([e.group for e in examples])
        return X, y, groups

    @classmethod
    def select_and_fit(cls, examples: Sequence[LinkerExample],
                       folds: int = 5, seed: int = 0,
                       verbose: bool = True) -> "FieldLinker":
        """Grouped CV over the model families, then refit the winner on all."""
        from sklearn.model_selection import GroupKFold

        X, y, groups = cls.vectorise(examples)
        n_groups = len(set(groups.tolist()))
        folds = max(2, min(folds, n_groups))
        if n_groups < 2 or len(set(y.tolist())) < 2:
            raise ValueError(
                "linker needs at least two groups and both classes present; "
                f"got {n_groups} group(s), classes {sorted(set(y.tolist()))}"
            )

        splitter = GroupKFold(n_splits=folds)
        results: dict[str, dict[str, float]] = {}

        for name in MODEL_NAMES:
            aps, aucs = [], []
            for train_idx, val_idx in splitter.split(X, y, groups):
                if len(set(y[train_idx].tolist())) < 2 or len(set(y[val_idx].tolist())) < 2:
                    continue
                model = build_model(name, seed=seed)
                model.fit(X[train_idx], y[train_idx])
                proba = _proba(model, X[val_idx])
                aps.append(average_precision_score(y[val_idx], proba))
                aucs.append(roc_auc_score(y[val_idx], proba))
            if aps:
                results[name] = {"average_precision": float(np.mean(aps)),
                                 "roc_auc": float(np.mean(aucs)),
                                 "folds": len(aps)}
                if verbose:
                    print(f"  linker/{name:<11} AP={np.mean(aps):.4f} "
                          f"AUC={np.mean(aucs):.4f} over {len(aps)} folds")

        if not results:
            raise ValueError("no linker model could be cross-validated")
        best = max(results, key=lambda k: results[k]["average_precision"])
        if verbose:
            print(f"  linker: selected {best}")

        model = build_model(best, seed=seed)
        model.fit(X, y)
        return cls(model=model, model_name=best,
                   metrics={"cv": results, "selected": best,
                            "n_examples": int(len(y)),
                            "positive_rate": float(y.mean())})

    # -- inference -----------------------------------------------------
    def score(self, matches: Sequence[Any], default_section_id: str) -> np.ndarray:
        if not matches:
            return np.zeros(0, dtype=np.float32)
        X = np.vstack([
            build_vector(linker_features(m, default_section_id), FEATURE_NAMES)
            for m in matches
        ])
        return _proba(self.model, X)

    def rescore(self, line, matches: Sequence[Any],
                default_section_id: str) -> list[Any]:
        """Reorder a line's candidates by the model's opinion.

        The linker decides WHICH field a line is, not WHETHER the value is
        good enough to fill -- that is the confidence model's job. So it
        reorders candidates and records its probability as a feature, but it
        does not overwrite the rule score.

        This matters on real documents. The model is trained on clean text; on
        badly-recognised handwriting it scores every candidate near zero. When
        that probability was blended into the score it dragged every match
        below the assignment floor and the page returned nothing at all, even
        where the label was matched correctly.
        """
        if not matches or self.model is None:
            return list(matches)
        probs = self.score(matches, default_section_id)
        out = []
        for match, p in zip(matches, probs):
            match.linker_score = float(p)
            # Ranking key only; `score` stays the rule score.
            match.rank_score = float(self.blend * p
                                     + (1.0 - self.blend) * match.score)
            out.append(match)
        out.sort(key=lambda m: -getattr(m, "rank_score", m.score))
        return out

    # -- persistence ---------------------------------------------------
    def save(self, path: Path | str) -> Path:
        path = Path(path)
        joblib.dump({
            "model": self.model, "model_name": self.model_name,
            "blend": self.blend, "metrics": self.metrics,
            "feature_names": self.feature_names,
        }, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "FieldLinker":
        blob = joblib.load(Path(path))
        obj = cls(model=blob["model"], model_name=blob.get("model_name", "logreg"),
                  blend=float(blob.get("blend", 0.65)),
                  metrics=blob.get("metrics", {}))
        obj.feature_names = blob.get("feature_names", list(FEATURE_NAMES))
        return obj


def _proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-scores))


__all__ = ["FieldLinker", "LinkerExample", "FEATURE_NAMES", "MODEL_NAMES",
           "build_model"]
