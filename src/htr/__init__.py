"""Public-dataset handwriting recognition: discover, fetch, prepare, train.

A separate package from ``factfind`` on purpose. ``factfind`` is the shipped
fact-find extraction pipeline with an ASCII charset baked into exported ONNX
models; this one trains sequence recognisers on public corpora with a Unicode
charset derived from the data. They share the CRNN architecture and nothing
else.

Pipeline:

    scripts/10_fetch_datasets.py    discover, download, extract
    scripts/11_build_manifests.py   adapt to one schema, validate, dedupe
    scripts/12_train_htr.py         split, charset, train, evaluate, export
    scripts/13_evaluate_htr.py      score a checkpoint on the held-out split
    scripts/14_export_htr_onnx.py   export and verify ONNX parity

See docs/htr-datasets.md for the inventory and the licence position.
"""
__version__ = "1.0.0"

__all__ = ["registry", "download", "extraction", "adapters", "manifest",
           "preprocess", "augment", "charset", "splits", "data", "models",
           "train", "evaluate", "export", "build", "paths"]
