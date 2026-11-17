"""The training loop.

Decisions that are easy to get wrong and are therefore made explicitly:

* **Padding is kept small by bucketing, not hidden by it.** Every crop in a
  batch is padded to one width, so the CTC input length really is the same
  for all of them; what bucketing buys is that the shared width is close to
  each crop's own, instead of the widest crop in the corpus.
* **Non-finite batches are skipped, and counted.** ``zero_infinity`` turns an
  impossible alignment into a zero gradient. Skipping quietly means a run can
  train on nothing and report a healthy-looking loss, so the count is printed
  and stored; if it is large, the crops are too narrow for their labels.
* **Selection is on validation CER, never on loss.** CTC loss and CER
  disagree often enough to matter, and CER is the thing anyone cares about.
* **Every improvement is checkpointed immediately.** Training this on CPU
  takes hours; losing it to an interrupted session is not an acceptable
  failure mode.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .charset import UnicodeCharset
from .evaluate import evaluate
from .manifest import Sample
from .models import TORCH_AVAILABLE, build_model, count_parameters
from .paths import MODELS_DIR
from .preprocess import PreprocessConfig

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn


@dataclass
class TrainConfig:
    architecture: str = "crnn"
    level: str = "word"
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_batches: int = 300
    patience: int = 5
    hidden: int | None = None
    dropout: float = 0.1
    grad_clip: float = 5.0
    workers: int = 0
    seed: int = 0
    device: str = "cpu"
    augment_strength: float = 1.0
    image_height: int = 32
    max_width: int = 512
    preprocess_level: str = "light"
    max_validation: int = 4000
    label_smoothing: float = 0.0

    def preprocess(self) -> PreprocessConfig:
        return PreprocessConfig(level=self.preprocess_level,
                                height=self.image_height,
                                max_width=self.max_width)


@dataclass
class TrainResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    best_cer: float = float("inf")
    best_epoch: int = -1
    stopped_early: bool = False
    parameters: int = 0
    skipped_batches: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def checkpoint_path(name: str = "htr") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_DIR / f"{name}.pt"


def save_checkpoint(path: Path, model, charset: UnicodeCharset,
                    config: TrainConfig, metrics: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "charset": list(charset.characters),
        "config": asdict(config),
        "metrics": metrics,
    }, path)
    return path


def load_checkpoint(path: Path | str, device: str = "cpu"):
    blob = torch.load(Path(path), map_location=device, weights_only=False)
    charset = UnicodeCharset(tuple(blob["charset"]))
    config = TrainConfig(**{k: v for k, v in blob["config"].items()
                            if k in TrainConfig.__dataclass_fields__})
    model = build_model(config.architecture, charset.n_classes,
                        hidden=config.hidden, dropout=config.dropout)
    model.load_state_dict(blob["state_dict"])
    model.to(device).eval()
    return model, charset, config, blob.get("metrics", {})


def train(model, loader, validation: Sequence[Sample], charset: UnicodeCharset,
          config: TrainConfig, *, checkpoint: Path | None = None,
          log_every: int = 50, stage: str = "train") -> TrainResult:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to train")

    device = config.device
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_batches = max(1, len(loader)) * config.epochs
    def learning_rate_at(step: int) -> float:
        if step < config.warmup_batches:
            return (step + 1) / max(1, config.warmup_batches)
        progress = (step - config.warmup_batches) / max(
            1, total_batches - config.warmup_batches)
        return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate_at)

    result = TrainResult(parameters=count_parameters(model))
    best_state = None
    since_improved = 0
    started = time.time()
    step = 0

    holdout = list(validation)[: config.max_validation]

    for epoch in range(1, config.epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        epoch_started = time.time()

        for images, targets, target_lengths in loader:
            images = images.to(device)
            log_probs = model(images)                       # (T, N, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                result.skipped_batches += 1
                step += 1
                scheduler.step()
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimiser.step()
            scheduler.step()
            step += 1
            running += float(loss.detach())
            batches += 1
            if log_every and batches % log_every == 0:
                rate = batches / max(1e-6, time.time() - epoch_started)
                print(f"    {stage} epoch {epoch} batch {batches}/{len(loader)} "
                      f"loss={running / batches:.4f} "
                      f"lr={optimiser.param_groups[0]['lr']:.2e} "
                      f"({rate:.1f} batch/s)", flush=True)

        metrics = evaluate(model, holdout, charset,
                           preprocess=config.preprocess(),
                           batch_size=max(16, config.batch_size),
                           device=device) if holdout else {}
        row = {
            "epoch": epoch,
            "loss": running / max(1, batches),
            "cer": metrics.get("cer"),
            "wer": metrics.get("wer"),
            "exact_match": metrics.get("exact_match"),
            "seconds": round(time.time() - epoch_started, 1),
        }
        result.history.append(row)
        print(f"  {stage} epoch {epoch}: loss={row['loss']:.4f}"
              + (f" cer={row['cer']:.4f} wer={row['wer']:.4f} "
                 f"exact={row['exact_match']:.4f}" if metrics else "")
              + f" [{row['seconds']:.0f}s]", flush=True)

        current = row["cer"] if row["cer"] is not None else row["loss"]
        if current < result.best_cer - 1e-4:
            result.best_cer = float(current)
            result.best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            since_improved = 0
            if checkpoint is not None:
                save_checkpoint(checkpoint, model, charset, config,
                                {"stage": stage, "epoch": epoch, **metrics})
                print(f"    checkpoint saved (cer={result.best_cer:.4f})",
                      flush=True)
        else:
            since_improved += 1
            if since_improved >= config.patience:
                print(f"  {stage}: no improvement for {config.patience} "
                      f"epochs, stopping early")
                result.stopped_early = True
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.seconds = round(time.time() - started, 1)
    if result.skipped_batches:
        print(f"  WARNING: {result.skipped_batches} batches skipped as "
              f"non-finite - crops too narrow for their labels")
    return result


def build_and_train(train_samples: Sequence[Sample],
                    validation: Sequence[Sample],
                    charset: UnicodeCharset, config: TrainConfig, *,
                    composed: Any = None, checkpoint: Path | None = None,
                    model=None) -> tuple[Any, TrainResult]:
    """Assemble loaders and run one training stage."""
    from .augment import AugmentConfig
    from .data import ManifestDataset, make_loader

    augment = AugmentConfig(strength=config.augment_strength) \
        if config.augment_strength > 0 else None

    dataset: Any = ManifestDataset(train_samples, charset,
                                   preprocess=config.preprocess(),
                                   augment=augment, seed=config.seed)
    if composed is not None:
        dataset = torch.utils.data.ConcatDataset([dataset, composed])

    loader = make_loader(dataset, config.batch_size, shuffle=True,
                         workers=config.workers, seed=config.seed,
                         bucket=composed is None)

    if model is None:
        model = build_model(config.architecture, charset.n_classes,
                            hidden=config.hidden, dropout=config.dropout)
    print(f"  model: {config.architecture}, "
          f"{count_parameters(model):,} parameters, "
          f"{charset.n_classes} classes")

    result = train(model, loader, validation, charset, config,
                   checkpoint=checkpoint)
    return model, result


__all__ = ["TrainConfig", "TrainResult", "train", "build_and_train",
           "save_checkpoint", "load_checkpoint", "checkpoint_path"]
