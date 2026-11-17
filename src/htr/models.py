"""Model architectures for the sequence recogniser.

The CRNN is **not re-implemented here**: it is imported from
``factfind.recognizer``, which already ships a proven CNN -> BiLSTM -> linear
stack with a dynamic width axis and a working ONNX export. One architecture,
one export path, one thing to keep correct. Only the class count changes, and
that is already a constructor argument.

A transformer encoder variant sits alongside it for when a GPU is available.
Both produce the same ``(T, N, C)`` log-probability tensor, so everything
downstream - CTC loss, decoding, evaluation, export - is shared.

TrOCR was considered and rejected. It is the most accurate option on IAM-style
lines, but it is ~60 MB of encoder-decoder, needs a GPU to fine-tune in any
reasonable time, and cannot run in the browser, which is where this project
ships its models. A ~2M-parameter CRNN that quantises to a few MB is the
constraint the whole repository is built around.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

# src/ is on the path when scripts bootstrap; make the sibling package
# importable when htr is used directly as well.
_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from factfind.recognizer import CRNN, DOWNSAMPLE, IMAGE_HEIGHT, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:                                                    # pragma: no cover
    nn = object


ARCHITECTURES = ("crnn", "crnn-large", "transformer")


if TORCH_AVAILABLE:

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positions over the width axis, (T, N, C)."""

        def __init__(self, dim: int, max_len: int = 2048):
            super().__init__()
            position = torch.arange(max_len).unsqueeze(1).float()
            divisor = torch.exp(torch.arange(0, dim, 2).float()
                                * (-math.log(10000.0) / dim))
            encoding = torch.zeros(max_len, 1, dim)
            encoding[:, 0, 0::2] = torch.sin(position * divisor)
            encoding[:, 0, 1::2] = torch.cos(position * divisor)
            self.register_buffer("encoding", encoding, persistent=False)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            return x + self.encoding[: x.size(0)]

    class CNNTransformer(nn.Module):
        """The CRNN's convolutional trunk with a transformer encoder on top.

        Same contract as CRNN: (N, 1, 32, W) in, (T, N, C) log-probs out.
        """

        def __init__(self, n_classes: int, hidden: int = 256, layers: int = 4,
                     heads: int = 8, dropout: float = 0.1,
                     feedforward: int = 1024):
            super().__init__()
            trunk = CRNN(n_classes, hidden=hidden)
            self.cnn = trunk.cnn
            channels = 256
            self.project = nn.Linear(channels, hidden)
            self.positions = _PositionalEncoding(hidden)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=feedforward,
                dropout=dropout, activation="gelu", batch_first=False,
                norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
            self.head = nn.Linear(hidden, n_classes)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            features = self.cnn(x)
            if features.size(2) != 1:
                features = features.mean(dim=2, keepdim=True)
            features = features.squeeze(2).permute(2, 0, 1)   # (T, N, C)
            features = self.positions(self.project(features))
            encoded = self.encoder(features)
            return F.log_softmax(self.head(encoded), dim=2)

else:                                                    # pragma: no cover
    class CNNTransformer:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required to build the recogniser")


def build_model(architecture: str, n_classes: int, *, hidden: int | None = None,
                dropout: float = 0.1) -> Any:
    """Construct one of the registered architectures."""
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required.\n"
            "  .venv/Scripts/python.exe -m pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu")
    architecture = architecture.lower()
    if architecture == "crnn":
        return CRNN(n_classes, hidden=hidden or 192, dropout=dropout)
    if architecture == "crnn-large":
        return CRNN(n_classes, channels=(64, 128, 256, 256),
                    hidden=hidden or 320, dropout=dropout)
    if architecture == "transformer":
        return CNNTransformer(n_classes, hidden=hidden or 256, dropout=dropout)
    raise ValueError(f"unknown architecture {architecture!r}; "
                     f"known: {', '.join(ARCHITECTURES)}")


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def output_length(width: int) -> int:
    """Frames the model emits for an input of this pixel width."""
    return max(1, width // DOWNSAMPLE - 1)


def minimum_width(text_length: int) -> int:
    """Narrowest input that can still encode a label of this length.

    CTC needs one frame per character, plus a blank between any repeated
    pair. Crops narrower than this make the loss infinite, and the usual
    symptom is a training run whose loss mysteriously stops moving.
    """
    return (2 * max(1, text_length) + 2) * DOWNSAMPLE


__all__ = ["build_model", "count_parameters", "output_length",
           "minimum_width", "ARCHITECTURES", "CRNN", "CNNTransformer",
           "IMAGE_HEIGHT", "DOWNSAMPLE", "TORCH_AVAILABLE"]
