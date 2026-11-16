"""Step 4.2 - train the handwriting recogniser.

    # stage 1 only: pre-train on synthetic handwriting (no real data needed)
    python scripts/06_train_recognizer.py --pretrain-only

    # both stages: pre-train, then fine-tune on your labelled crops
    python scripts/06_train_recognizer.py

Stage 1 learns to read handwriting-shaped text built from the knowledge file.
Stage 2 adapts it to your advisers' actual hands, using the crops transcribed
in app/label_app.py. Every reported number comes from real crops held out by
writer; synthetic data is never used to judge the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from factfind.knowledge import load_knowledge
from factfind.paths import MODELS_DIR, REPORTS_DIR
from factfind.recognizer import (Charset, HandwritingRecognizer, TORCH_AVAILABLE,
                                 export_onnx)

MIN_REAL_CROPS = 50


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pretrain-epochs", type=int, default=6)
    ap.add_argument("--finetune-epochs", type=int, default=30)
    ap.add_argument("--samples-per-epoch", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--pretrain-only", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip stage 1 and fine-tune the saved model")
    args = ap.parse_args()

    if not TORCH_AVAILABLE:
        print("STOPPING. PyTorch is not installed.")
        print("  .venv/Scripts/python.exe -m pip install torch torchvision "
              "--index-url https://download.pytorch.org/whl/cpu")
        return 2

    from factfind.crops import load_labels, split_by_writer, stats, to_samples
    from factfind.recognizer_train import evaluate, finetune_real, pretrain_synthetic

    kn = load_knowledge()
    charset = Charset.default()
    print(f"charset: {charset.n_classes} classes (including CTC blank)")

    report: dict = {"charset": charset.to_dict()}
    checkpoint = MODELS_DIR / "recognizer.pt"

    # ------------------------------------------------------------ stage 1
    if args.resume and checkpoint.exists():
        print(f"\nresuming from {checkpoint}")
        recognizer = HandwritingRecognizer.load(checkpoint)
        model = recognizer.model
        charset = recognizer.charset
    else:
        print("\n=== Stage 1: pre-train on synthetic handwriting ===")
        model, pretrain_result = pretrain_synthetic(
            kn, charset, epochs=args.pretrain_epochs,
            samples_per_epoch=args.samples_per_epoch,
            batch_size=args.batch_size, seed=args.seed, hidden=args.hidden,
            num_workers=args.workers, checkpoint=checkpoint,
        )
        report["pretrain"] = pretrain_result
        HandwritingRecognizer(model=model, charset=charset,
                              metrics=pretrain_result).save(checkpoint)
        print(f"saved {checkpoint}")

    # ------------------------------------------------------------ stage 2
    records = load_labels()
    crop_stats = stats(records)
    report["crops"] = crop_stats
    print(f"\nreal crops: {crop_stats['n_transcribed']} transcribed "
          f"of {crop_stats['n_crops']} cut, "
          f"{crop_stats['n_writers']} writer(s)")

    if args.pretrain_only:
        print("\n--pretrain-only given; stopping before fine-tuning.")
    elif crop_stats["n_transcribed"] < MIN_REAL_CROPS:
        print(f"\nNOT fine-tuning: only {crop_stats['n_transcribed']} "
              f"transcribed crops, need at least {MIN_REAL_CROPS}.")
        print("  Run: streamlit run app/label_app.py --server.address localhost")
        print("  The synthetic-only model is saved, but it has NOT been shown "
              "to read your handwriting. Do not trust it until stage 2 runs.")
    else:
        done = [r for r in records if r.text.strip()]
        train_records, val_records = split_by_writer(done, seed=args.seed)
        train_samples = to_samples(train_records)
        val_samples = to_samples(val_records)

        if crop_stats["n_writers"] < 2:
            print("  WARNING: one writer only, so validation is held out by "
                  "page, not by hand. It cannot show the model generalises "
                  "to a new adviser.")

        print(f"\n=== Stage 2: fine-tune on real crops ===")
        print(f"  {len(train_samples)} train / {len(val_samples)} validation")

        before = evaluate(HandwritingRecognizer(model=model, charset=charset),
                          val_samples)
        print(f"  before fine-tuning: CER={before['cer']:.4f} "
              f"exact={before['exact_match']:.4f}")

        result = finetune_real(model, train_samples, val_samples, charset,
                               epochs=args.finetune_epochs,
                               batch_size=min(args.batch_size, 16),
                               seed=args.seed)

        after = evaluate(HandwritingRecognizer(model=model, charset=charset),
                         val_samples)
        print(f"  after  fine-tuning: CER={after['cer']:.4f} "
              f"exact={after['exact_match']:.4f}")

        report["finetune"] = {"history": result["history"],
                              "before": before, "after": after,
                              "n_train": len(train_samples),
                              "n_validation": len(val_samples)}

        if after["cer"] > before["cer"]:
            print("  Fine-tuning made it worse on held-out real crops. "
                  "Keeping the pre-trained weights.")
        else:
            HandwritingRecognizer(model=model, charset=charset,
                                  metrics=report).save(checkpoint)
            print(f"  saved {checkpoint}")

    # ------------------------------------------------------------ export
    recognizer = HandwritingRecognizer(model=model, charset=charset)
    export_dir = Path("angular/assets/factfind")
    onnx_info = export_onnx(recognizer, export_dir / "recognizer.onnx")
    (export_dir / "recognizer-charset.json").write_text(
        json.dumps(charset.to_dict(), indent=2), encoding="utf-8")
    report["onnx"] = onnx_info
    print(f"\nexported {onnx_info['path']} "
          f"({onnx_info['bytes'] / 1024 / 1024:.2f} MB)")

    out = REPORTS_DIR / "recognizer.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
