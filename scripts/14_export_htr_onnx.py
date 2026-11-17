"""Step 5 - export a saved checkpoint to ONNX and verify parity.

    python scripts/14_export_htr_onnx.py --name htr-char-crnn
    python scripts/14_export_htr_onnx.py --name htr-word-crnn --out angular/assets/factfind

Writes <name>.onnx, <name>-charset.json and <name>.json (input/output
contract plus metrics) and refuses to call the export good unless the ONNX
graph reproduces the PyTorch model at three different crop widths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from htr.export import export_bundle
from htr.models import TORCH_AVAILABLE
from htr.paths import MODELS_DIR
from htr.train import checkpoint_path, load_checkpoint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="htr-char-crnn")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default=None, help="output directory")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("STOPPING. PyTorch is not installed.")
        return 2

    path = Path(args.checkpoint) if args.checkpoint else checkpoint_path(args.name)
    if not path.exists():
        print(f"no checkpoint at {path}")
        return 1

    model, charset, config, metrics = load_checkpoint(path)
    directory = Path(args.out) if args.out else MODELS_DIR
    bundle = export_bundle(model, charset, directory, name=args.name,
                           metrics=metrics)

    print(f"exported {bundle['onnx']['path']} "
          f"({bundle['onnx']['megabytes']} MB)")
    print(f"  classes {bundle['classes']}, scripts "
          f"{', '.join(bundle['scripts'])}")
    print(f"  parity  max abs difference "
          f"{bundle['parity']['max_abs_difference']:.2e} - passed")
    print(f"  charset {directory / (args.name + '-charset.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
