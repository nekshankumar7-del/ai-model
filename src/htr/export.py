"""ONNX export, with a parity check that actually fails when parity fails.

The width axis is dynamic because handwriting crops are not a fixed size, and
the batch axis is dynamic because the browser reads several value crops from
one page. The exported graph is then run against the PyTorch model on random
inputs at several widths: an export that silently bakes in the dummy width is
the classic failure here, and it only shows up in production.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .charset import UnicodeCharset
from .models import IMAGE_HEIGHT, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


def export_onnx(model, path: Path | str, *, width: int = 256,
                opset: int = 17) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    dummy = torch.randn(1, 1, IMAGE_HEIGHT, width)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["crop"], output_names=["log_probs"],
        dynamic_axes={"crop": {0: "batch", 3: "width"},
                      "log_probs": {0: "time", 1: "batch"}},
        opset_version=opset, do_constant_folding=True,
        # The TorchScript exporter, explicitly. From torch 2.9 the default is
        # the torch.export path, which needs onnxscript and, for an LSTM with
        # a dynamic width axis, does not reliably keep that axis dynamic. The
        # parity check below is what proves this choice is the right one.
        dynamo=False,
    )
    return {"path": str(path), "bytes": path.stat().st_size,
            "megabytes": round(path.stat().st_size / 1024 / 1024, 2)}


def check_parity(model, path: Path | str, *, widths=(128, 256, 384),
                 batch: int = 2, tolerance: float = 2e-4) -> dict[str, Any]:
    """Run both and compare. Raises if the graph does not reproduce the model."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    model.eval().cpu()

    results = []
    worst = 0.0
    for width in widths:
        sample = torch.randn(batch, 1, IMAGE_HEIGHT, width)
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = session.run(None, {name: sample.numpy()})[0]
        if expected.shape != actual.shape:
            raise AssertionError(
                f"shape mismatch at width {width}: torch {expected.shape} "
                f"vs onnx {actual.shape} - the width axis did not stay dynamic")
        difference = float(np.abs(expected - actual).max())
        worst = max(worst, difference)
        results.append({"width": width, "max_abs_difference": difference})

    if worst > tolerance:
        raise AssertionError(
            f"ONNX output differs from PyTorch by {worst:.2e} "
            f"(tolerance {tolerance:.0e})")
    return {"passed": True, "max_abs_difference": worst, "per_width": results}


def export_bundle(model, charset: UnicodeCharset, directory: Path | str, *,
                  name: str = "htr-recognizer",
                  metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    """Model + charset + metadata, everything a consumer needs to decode."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    onnx_info = export_onnx(model, directory / f"{name}.onnx")
    charset.save(directory / f"{name}-charset.json")
    parity = check_parity(model, directory / f"{name}.onnx")

    meta = {
        "name": name,
        "input": {"name": "crop", "layout": "NCHW", "height": IMAGE_HEIGHT,
                  "channels": 1, "dtype": "float32",
                  "normalisation": "pixel / 127.5 - 1.0",
                  "width": "dynamic, multiple of 4"},
        "output": {"name": "log_probs", "layout": "TNC",
                   "decoding": "CTC best path, blank index 0"},
        "classes": charset.n_classes,
        "scripts": charset.scripts(),
        "onnx": onnx_info,
        "parity": parity,
        "metrics": metrics or {},
    }
    (directory / f"{name}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta


__all__ = ["export_onnx", "check_parity", "export_bundle"]
