"""CTC decoding and the metrics that actually tell you something.

Reported per run:

* **CER** - character error rate, edit distance over reference length. The
  headline number for handwriting recognition.
* **WER** - word error rate, computed over whitespace tokens. On a word-level
  corpus it collapses to 1 - exact match; on lines it is the number anyone
  comparing against published IAM results will look for.
* **Exact match** - the only metric that matters when the output goes into a
  form field. A salary read as 45009 instead of 45000 has a CER of 0.2 and is
  still simply wrong.

Everything is broken down per dataset, per script and per text length,
because a single pooled CER across Latin characters and Devanagari words is a
number with no meaning.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .charset import UnicodeCharset, script_of
from .data import collate
from .manifest import Sample, load_image
from .models import TORCH_AVAILABLE
from .preprocess import PreprocessConfig

if TORCH_AVAILABLE:
    import torch


def levenshtein(a: Sequence, b: Sequence) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[j - 1] + 1, previous[j] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


@dataclass
class Prediction:
    text: str
    confidence: float


def greedy_decode(log_probs: "torch.Tensor", charset: UnicodeCharset
                  ) -> list[Prediction]:
    """Best-path CTC decode. log_probs is (T, N, C)."""
    best = log_probs.argmax(dim=2)
    frame_confidence = log_probs.exp().max(dim=2).values

    out: list[Prediction] = []
    for n in range(best.size(1)):
        path = best[:, n].tolist()
        confidences = frame_confidence[:, n].tolist()
        indices: list[int] = []
        kept: list[float] = []
        previous = -1
        for index, confidence in zip(path, confidences):
            if index != previous and index != 0:
                indices.append(index)
                kept.append(confidence)
            previous = index
        # The weakest character, not the mean: one badly-read digit ruins a
        # number, and the mean hides it behind the characters that were easy.
        out.append(Prediction(charset.decode_indices(indices),
                              float(min(kept)) if kept else 0.0))
    return out


def _bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 4:
        return "2-4"
    if length <= 10:
        return "5-10"
    if length <= 25:
        return "11-25"
    return "26+"


class _Accumulator:
    def __init__(self) -> None:
        self.distance = 0
        self.characters = 0
        self.word_distance = 0
        self.words = 0
        self.exact = 0
        self.n = 0

    def add(self, truth: str, prediction: str) -> None:
        self.distance += levenshtein(prediction, truth)
        self.characters += max(1, len(truth))
        truth_words, prediction_words = truth.split(), prediction.split()
        self.word_distance += levenshtein(prediction_words, truth_words)
        self.words += max(1, len(truth_words))
        self.exact += int(prediction == truth)
        self.n += 1

    def as_dict(self) -> dict[str, float]:
        return {
            "cer": self.distance / max(1, self.characters),
            "wer": self.word_distance / max(1, self.words),
            "exact_match": self.exact / max(1, self.n),
            "n": self.n,
        }


def evaluate(model, samples: Sequence[Sample], charset: UnicodeCharset, *,
             preprocess: PreprocessConfig | None = None,
             batch_size: int = 64, device: str = "cpu",
             keep_examples: int = 25) -> dict[str, Any]:
    """Run the model over a split and report the breakdowns."""
    from .data import ManifestDataset, make_loader

    if not samples:
        return {"cer": 1.0, "wer": 1.0, "exact_match": 0.0, "n": 0}

    preprocess = preprocess or PreprocessConfig()
    dataset = ManifestDataset(samples, charset, preprocess=preprocess,
                              augment=None)
    # No shuffling and no bucketing, so predictions line up with `samples`.
    loader = make_loader(dataset, batch_size, shuffle=False, workers=0,
                         bucket=False)

    overall = _Accumulator()
    per_dataset: dict[str, _Accumulator] = defaultdict(_Accumulator)
    per_script: dict[str, _Accumulator] = defaultdict(_Accumulator)
    per_length: dict[str, _Accumulator] = defaultdict(_Accumulator)
    worst: list[tuple[float, str, str, str]] = []

    model.eval()
    position = 0
    with torch.no_grad():
        for images, _targets, _lengths in loader:
            predictions = greedy_decode(model(images.to(device)).cpu(), charset)
            for prediction in predictions:
                sample = samples[position]
                position += 1
                truth = charset.filter(sample.text)
                overall.add(truth, prediction.text)
                per_dataset[sample.dataset].add(truth, prediction.text)
                per_script[sample.script].add(truth, prediction.text)
                per_length[_bucket(len(truth))].add(truth, prediction.text)
                if prediction.text != truth and len(worst) < keep_examples:
                    worst.append((prediction.confidence, sample.id, truth,
                                  prediction.text))

    return {
        **overall.as_dict(),
        "per_dataset": {k: v.as_dict() for k, v in sorted(per_dataset.items())},
        "per_script": {k: v.as_dict() for k, v in sorted(per_script.items())},
        "per_length": {k: v.as_dict() for k, v in sorted(per_length.items())},
        "example_errors": [{"id": i, "truth": t, "predicted": p,
                            "confidence": round(c, 3)}
                           for c, i, t, p in worst],
    }


def read_crops(model, crops, charset: UnicodeCharset, *,
               preprocess: PreprocessConfig | None = None,
               device: str = "cpu") -> list[Prediction]:
    """Read raw images - the inference entry point for the trained model."""
    from .data import batch_for_inference
    if not len(crops):
        return []
    batch = batch_for_inference(crops, preprocess)
    model.eval()
    with torch.no_grad():
        log_probs = model(batch.to(device)).cpu()
    return greedy_decode(log_probs, charset)


__all__ = ["evaluate", "greedy_decode", "read_crops", "levenshtein",
           "Prediction"]
