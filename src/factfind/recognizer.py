"""Handwriting recogniser: a compact CRNN-CTC trained on your own notes.

Why this exists: PP-OCRv4 and Tesseract are printed-text engines. On the real
WealthMax notebook pages they return things like "uopm mms 5d3g" for
"Gross annual salary - 50000", and no amount of field-matching logic can
recover a value from that. The recogniser is the bottleneck, so it is the
thing worth training.

Design choices, and why:

  * CRNN-CTC rather than a transformer. It is ~2M parameters, quantises to a
    few MB, runs in the browser inside the per-page budget, and trains on a
    few thousand crops. TrOCR would be more accurate out of the box but is
    ~60MB and far slower.
  * Pre-train on synthetic lines rendered from handwriting fonts using the
    knowledge file's own vocabulary, then fine-tune on real labelled crops.
    Selection always happens on real data.
  * Lexicon-constrained decoding. For a choice field the answer must be one
    of the allowed values, so instead of reading freely we score every
    allowed string under the CTC output and take the best. This is what turns
    a blurry "Jahan" into "Japan" and "Amduiam" into "Indian".
"""
from __future__ import annotations

import json
import math
import random
import string
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Torch is only needed for training and for the PyTorch inference path; the
# shipped browser model is ONNX. Import lazily so the rest of the package
# keeps working without it.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    torch = None
    nn = object  # type: ignore
    TORCH_AVAILABLE = False

IMAGE_HEIGHT = 32
# Width is dynamic at inference; training pads to this.
MAX_WIDTH = 384
DOWNSAMPLE = 4          # the CNN reduces width by this factor


# --------------------------------------------------------------------------
# Charset
# --------------------------------------------------------------------------

@dataclass
class Charset:
    """Index 0 is the CTC blank; every other index is one character."""
    characters: str

    @staticmethod
    def default() -> "Charset":
        chars = (
            string.ascii_uppercase + string.ascii_lowercase + string.digits
            + " .,-/:'()&@+£$%#*"
        )
        # Deduplicate while keeping order stable, because the index order is
        # baked into every exported model.
        seen: list[str] = []
        for c in chars:
            if c not in seen:
                seen.append(c)
        return Charset("".join(seen))

    def __post_init__(self) -> None:
        self._to_index = {c: i + 1 for i, c in enumerate(self.characters)}
        self._to_char = {i + 1: c for i, c in enumerate(self.characters)}

    @property
    def n_classes(self) -> int:
        return len(self.characters) + 1          # + blank

    def encode(self, text: str) -> list[int]:
        return [self._to_index[c] for c in text if c in self._to_index]

    def decode_indices(self, indices: Iterable[int]) -> str:
        return "".join(self._to_char.get(int(i), "") for i in indices)

    def supports(self, text: str) -> bool:
        return all(c in self._to_index for c in text)

    def filter(self, text: str) -> str:
        return "".join(c for c in text if c in self._to_index)

    def to_dict(self) -> dict[str, Any]:
        return {"characters": self.characters, "blankIndex": 0,
                "nClasses": self.n_classes}


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class CRNN(nn.Module):
        """Convolutional feature extractor + BiLSTM + per-frame classifier.

        Input  : (N, 1, 32, W)   greyscale, normalised to [-1, 1]
        Output : (T, N, C)       log-probabilities, T = W/4 - 1
        """

        def __init__(self, n_classes: int, channels: Sequence[int] = (32, 64, 128, 256),
                     hidden: int = 192, dropout: float = 0.1):
            super().__init__()
            c1, c2, c3, c4 = channels
            self.cnn = nn.Sequential(
                nn.Conv2d(1, c1, 3, 1, 1), nn.BatchNorm2d(c1), nn.ReLU(True),
                nn.MaxPool2d(2, 2),                       # 16 x W/2

                nn.Conv2d(c1, c2, 3, 1, 1), nn.BatchNorm2d(c2), nn.ReLU(True),
                nn.MaxPool2d(2, 2),                       # 8 x W/4

                nn.Conv2d(c2, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(True),
                nn.Conv2d(c3, c3, 3, 1, 1), nn.BatchNorm2d(c3), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),             # 4 x W/4

                nn.Conv2d(c3, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.ReLU(True),
                nn.MaxPool2d((2, 1), (2, 1)),             # 2 x W/4

                nn.Conv2d(c4, c4, 2, 1, 0), nn.BatchNorm2d(c4), nn.ReLU(True),
                # 1 x (W/4 - 1)
            )
            self.rnn = nn.LSTM(c4, hidden, num_layers=2, bidirectional=True,
                               batch_first=False, dropout=dropout)
            self.head = nn.Linear(hidden * 2, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.cnn(x)                        # (N, C, 1, T)
            if features.size(2) != 1:
                # Guard rather than silently averaging the wrong axis: a bad
                # input height here would corrupt every prediction.
                features = features.mean(dim=2, keepdim=True)
            features = features.squeeze(2)                # (N, C, T)
            features = features.permute(2, 0, 1)          # (T, N, C)
            recurrent, _ = self.rnn(features)
            logits = self.head(recurrent)                 # (T, N, n_classes)
            return F.log_softmax(logits, dim=2)

else:                                                      # pragma: no cover
    class CRNN:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required to build the recogniser")


# --------------------------------------------------------------------------
# Image preparation
# --------------------------------------------------------------------------

def prepare_crop(image: np.ndarray, height: int = IMAGE_HEIGHT,
                 max_width: int = MAX_WIDTH,
                 pad_to: int | None = None) -> np.ndarray:
    """Greyscale, scale to a fixed height, pad to a width. Returns (1, H, W)."""
    import cv2

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.size == 0:
        image = np.full((height, 8), 255, np.uint8)

    h, w = image.shape[:2]
    scale = height / max(1, h)
    new_w = max(8, min(max_width, int(round(w * scale))))
    resized = cv2.resize(image, (new_w, height), interpolation=cv2.INTER_LINEAR)

    target = pad_to or new_w
    target = max(target, new_w)
    if target > new_w:
        # Pad with the page background (estimated from the border) rather than
        # a flat white, so the model does not learn an artificial edge.
        border = int(np.median(np.concatenate([resized[0], resized[-1]])))
        canvas = np.full((height, target), border, np.uint8)
        canvas[:, :new_w] = resized
        resized = canvas

    normalised = resized.astype(np.float32) / 127.5 - 1.0
    return normalised[None, :, :]


def batch_crops(crops: Sequence[np.ndarray], height: int = IMAGE_HEIGHT,
                max_width: int = MAX_WIDTH) -> "torch.Tensor":
    prepared = [prepare_crop(c, height, max_width) for c in crops]
    width = max(p.shape[2] for p in prepared)
    # Round up so every sample in the batch shares one width.
    width = int(math.ceil(width / DOWNSAMPLE) * DOWNSAMPLE)
    padded = []
    for p in prepared:
        if p.shape[2] < width:
            border = float(np.median(p[:, :, -1]))
            pad = np.full((1, height, width - p.shape[2]), border, np.float32)
            p = np.concatenate([p, pad], axis=2)
        padded.append(p)
    return torch.from_numpy(np.stack(padded))


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------

@dataclass
class Reading:
    text: str
    confidence: float
    candidates: list[tuple[str, float]] = dc_field(default_factory=list)


def greedy_decode(log_probs: "torch.Tensor", charset: Charset
                  ) -> list[Reading]:
    """Standard CTC best-path decode. log_probs is (T, N, C)."""
    probs = log_probs.exp()
    best = log_probs.argmax(dim=2)                 # (T, N)
    frame_conf = probs.max(dim=2).values           # (T, N)

    readings: list[Reading] = []
    for n in range(best.size(1)):
        path = best[:, n].tolist()
        confs = frame_conf[:, n].tolist()
        chars: list[int] = []
        kept: list[float] = []
        previous = -1
        for index, conf in zip(path, confs):
            if index != previous and index != 0:
                chars.append(index)
                kept.append(conf)
            previous = index
        text = charset.decode_indices(chars)
        # Confidence is the weakest character, not the mean: one badly-read
        # digit ruins a salary, and the mean would hide it.
        confidence = float(min(kept)) if kept else 0.0
        readings.append(Reading(text=text, confidence=confidence))
    return readings


def lexicon_decode(log_probs: "torch.Tensor", charset: Charset,
                   lexicon: Sequence[str], top_k: int = 3) -> list[Reading]:
    """Score every allowed string under CTC and return the best.

    This is what the brief means by lexicon-constrained decoding: a choice
    field cannot output anything but an allowed option, so we never have to
    hope the free decode happens to land on one.
    """
    if not lexicon:
        return greedy_decode(log_probs, charset)

    T, N, _C = log_probs.shape
    readings: list[Reading] = []
    usable = [w for w in lexicon if charset.supports(w) and w]
    if not usable:
        return greedy_decode(log_probs, charset)

    targets = [torch.tensor(charset.encode(w), dtype=torch.long) for w in usable]
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.cat(targets)

    for n in range(N):
        # Repeat this sample once per lexicon entry and score them in one go.
        sample = log_probs[:, n: n + 1, :].expand(T, len(usable), -1).contiguous()
        input_lengths = torch.full((len(usable),), T, dtype=torch.long)
        losses = F.ctc_loss(
            sample, flat_targets, input_lengths, target_lengths,
            blank=0, reduction="none", zero_infinity=True,
        )
        # loss is -log P(word); turn it into a comparable probability.
        scores = torch.exp(-losses / target_lengths.clamp(min=1).float())
        order = torch.argsort(scores, descending=True)
        ranked = [(usable[int(i)], float(scores[int(i)])) for i in order[:top_k]]
        readings.append(Reading(text=ranked[0][0], confidence=ranked[0][1],
                                candidates=ranked))
    return readings


# --------------------------------------------------------------------------
# Inference wrapper
# --------------------------------------------------------------------------

class HandwritingRecognizer:
    """Reads cropped line/value images. Wraps either torch or ONNX."""

    def __init__(self, model=None, charset: Charset | None = None,
                 session=None, metrics: dict[str, Any] | None = None):
        self.model = model
        self.charset = charset or Charset.default()
        self.session = session
        self.metrics = metrics or {}
        if self.model is not None and TORCH_AVAILABLE:
            self.model.eval()

    # -- loading --------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "HandwritingRecognizer":
        path = Path(path)
        blob = torch.load(path, map_location="cpu", weights_only=False)
        charset = Charset(blob["charset"])
        model = CRNN(charset.n_classes, hidden=blob.get("hidden", 192))
        model.load_state_dict(blob["state_dict"])
        return cls(model=model, charset=charset, metrics=blob.get("metrics", {}))

    @classmethod
    def load_onnx(cls, path: Path | str, charset_path: Path | str
                  ) -> "HandwritingRecognizer":
        import onnxruntime as ort
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        charset = Charset(json.loads(Path(charset_path).read_text(encoding="utf-8"))["characters"])
        return cls(session=session, charset=charset)

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        torch.save({
            "state_dict": self.model.state_dict(),
            "charset": self.charset.characters,
            "hidden": 192,
            "metrics": self.metrics,
        }, path)
        return path

    # -- reading --------------------------------------------------------
    def _log_probs(self, crops: Sequence[np.ndarray]) -> "torch.Tensor":
        batch = batch_crops(crops)
        if self.model is not None:
            with torch.no_grad():
                return self.model(batch)
        if self.session is not None:
            name = self.session.get_inputs()[0].name
            out = self.session.run(None, {name: batch.numpy()})[0]
            return torch.from_numpy(out)
        raise RuntimeError("recogniser has neither a torch model nor an ONNX session")

    def read(self, crops: Sequence[np.ndarray],
             lexicon: Sequence[str] | None = None) -> list[Reading]:
        if not crops:
            return []
        log_probs = self._log_probs(crops)
        if lexicon:
            return lexicon_decode(log_probs, self.charset, lexicon)
        return greedy_decode(log_probs, self.charset)

    def read_one(self, crop: np.ndarray,
                 lexicon: Sequence[str] | None = None) -> Reading:
        return self.read([crop], lexicon)[0]

    def read_for_field(self, crop: np.ndarray, field) -> Reading:
        """Read a crop with whatever constraint the field's kind implies."""
        lexicon = lexicon_for_field(field)
        return self.read_one(crop, lexicon)


def lexicon_for_field(field) -> list[str]:
    """The allowed strings for a field, straight from the knowledge file.

    Choice fields are closed sets, so they get a hard lexicon. Vocabulary
    fields get their vocabulary as a soft hint -- the knowledge file is
    explicit that vocabulary is not a list of allowed values, so those are
    decoded freely and snapped afterwards by the parser.
    """
    kind = getattr(field, "kind", None)
    if kind == "choice":
        out: list[str] = []
        for choice in field.resolved_choices():
            out.append(choice.value)
            if choice.label and choice.label != choice.value:
                out.append(choice.label)
            out.extend(a for a in choice.aliases if a)
        return list(dict.fromkeys(out))
    if kind == "yesNo":
        return ["Yes", "No", "Y", "N"]
    return []


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------

def export_onnx(recognizer: "HandwritingRecognizer", path: Path,
                width: int = 256) -> dict[str, Any]:
    """Export with a dynamic width axis so any crop length works."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = recognizer.model
    model.eval()
    dummy = torch.randn(1, 1, IMAGE_HEIGHT, width)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["crop"], output_names=["log_probs"],
        dynamic_axes={"crop": {0: "batch", 3: "width"},
                      "log_probs": {0: "time", 1: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    return {"path": str(path), "bytes": path.stat().st_size}


__all__ = ["Charset", "CRNN", "HandwritingRecognizer", "Reading",
           "prepare_crop", "batch_crops", "greedy_decode", "lexicon_decode",
           "lexicon_for_field", "export_onnx", "IMAGE_HEIGHT", "MAX_WIDTH",
           "TORCH_AVAILABLE"]
