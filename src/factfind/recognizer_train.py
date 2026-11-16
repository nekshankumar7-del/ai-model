"""Training for the handwriting recogniser.

Two stages, as the brief requires:

  1. Pre-train on synthetic line images rendered from handwriting fonts, using
     the knowledge file's own labels, choices and vocabulary so the model sees
     the strings it will actually meet.
  2. Fine-tune on real labelled crops from your notes. All model selection and
     every reported number comes from real data.

Augmentation is applied to the training split only.
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from .knowledge import Knowledge, Field
from .recognizer import (Charset, CRNN, HandwritingRecognizer, IMAGE_HEIGHT,
                         MAX_WIDTH, TORCH_AVAILABLE, batch_crops, greedy_decode,
                         prepare_crop)
from .synth import HAND_FONTS, Writer, _load_font, available_writers

if TORCH_AVAILABLE:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
else:                                                   # pragma: no cover
    Dataset = object


# --------------------------------------------------------------------------
# Corpus: what strings the recogniser should be able to read
# --------------------------------------------------------------------------

def knowledge_corpus(kn: Knowledge, rng: random.Random,
                     n_values: int = 6000) -> list[str]:
    """Strings drawn from the knowledge file plus realistic value shapes."""
    from .synth import (COMPANIES, FIRST_NAMES, LAST_NAMES, generate_value,
                        surface_form)

    corpus: list[str] = []

    # Every label and alias: the recogniser must read the left-hand side too.
    for fld in kn.fields:
        corpus.append(fld.label)
        corpus.extend(fld.aliases)
        for choice in fld.resolved_choices():
            corpus.append(choice.label)
            corpus.append(choice.value)
            corpus.extend(choice.aliases)
        corpus.extend(fld.vocabulary)
    for section in kn.sections:
        corpus.append(section.title)
        corpus.extend(section.headings)

    # Values, in the forms people write them.
    fields = kn.fields
    for _ in range(n_values):
        fld = rng.choice(fields)
        value = generate_value(fld, rng, {})
        if value is None:
            continue
        corpus.append(str(surface_form(fld, value, rng)))

    # Whole "label - value" lines, which is what a detected line actually is.
    for _ in range(n_values // 2):
        fld = rng.choice(fields)
        value = generate_value(fld, rng, {})
        if value is None:
            continue
        label = rng.choice([fld.label, *fld.aliases]) if fld.aliases else fld.label
        sep = rng.choices([" - ", " -", ": ", " : ", " – ", " "],
                          weights=[40, 15, 20, 10, 8, 7], k=1)[0]
        corpus.append(f"{label}{sep}{surface_form(fld, value, rng)}")

    corpus.extend(FIRST_NAMES)
    corpus.extend(LAST_NAMES)
    corpus.extend(COMPANIES)
    corpus.extend(f"{a} {b}" for a, b in
                  zip(rng.sample(FIRST_NAMES, 30), rng.sample(LAST_NAMES, 30)))

    cleaned = []
    seen = set()
    for item in corpus:
        text = " ".join(str(item).split())
        if text and text not in seen and len(text) <= 60:
            seen.add(text)
            cleaned.append(text)
    return cleaned


# --------------------------------------------------------------------------
# Synthetic line rendering
# --------------------------------------------------------------------------

def render_line(text: str, writer: Writer, rng: random.Random,
                height: int = IMAGE_HEIGHT * 2) -> np.ndarray:
    """Render one line of handwriting as a greyscale crop."""
    from PIL import Image, ImageDraw

    font_size = int(height * rng.uniform(0.55, 0.78))
    font = _load_font(writer.hand_font, font_size)

    probe = Image.new("L", (10, 10), 255)
    width = int(ImageDraw.Draw(probe).textlength(text, font=font))
    width = max(16, width + rng.randint(10, 40))

    image = Image.new("L", (width, height), rng.randint(238, 253))
    draw = ImageDraw.Draw(image)

    # Optional ruled line, as on the real notepad pages.
    if rng.random() < 0.45:
        y = int(height * rng.uniform(0.74, 0.92))
        draw.line([(0, y), (width, y)], fill=rng.randint(170, 205), width=1)

    x = rng.randint(2, 12)
    baseline = int(height * 0.5 - font_size * 0.55)
    ink = rng.randint(10, 70)
    for ch in text:
        dy = int(rng.uniform(-writer.jitter, writer.jitter))
        draw.text((x + rng.uniform(-0.6, 0.9), baseline + dy), ch,
                  font=font, fill=max(0, ink + rng.randint(-18, 18)))
        x += draw.textlength(ch, font=font) + rng.uniform(-0.4, 1.0)

    return np.array(image)


def augment_crop(image: np.ndarray, rng: random.Random,
                 strength: float = 1.0) -> np.ndarray:
    """Degradations a phone photo of a notepad actually produces."""
    out = image.copy()
    h, w = out.shape[:2]

    if rng.random() < 0.5 * strength:                      # slant / shear
        shear = rng.uniform(-0.25, 0.25) * strength
        M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.45 * strength:                     # rotation
        angle = rng.uniform(-3.5, 3.5) * strength
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        out = cv2.warpAffine(out, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < 0.5 * strength:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.2))

    if rng.random() < 0.4 * strength:                      # uneven lighting
        gradient = np.linspace(rng.uniform(0.75, 1.0), rng.uniform(0.75, 1.0), w)
        out = np.clip(out.astype(np.float32) * gradient[None, :], 0, 255).astype(np.uint8)

    if rng.random() < 0.5 * strength:
        gain = rng.uniform(0.82, 1.18)
        bias = rng.uniform(-20, 14)
        out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if rng.random() < 0.5 * strength:
        noise = np.random.normal(0, rng.uniform(2, 10) * strength, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.4 * strength:                      # JPEG
        ok, buf = cv2.imencode(".jpg", out,
                               [int(cv2.IMWRITE_JPEG_QUALITY), int(rng.uniform(35, 85))])
        if ok:
            out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

    if rng.random() < 0.3 * strength:                      # resolution loss
        scale = rng.uniform(0.55, 0.9)
        small = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

@dataclass
class CropSample:
    """One labelled crop: either a path on disk or an in-memory image."""
    text: str
    image_path: str | None = None
    image: np.ndarray | None = None
    writer: str = "unknown"
    source: str = "real"

    def load(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        data = cv2.imdecode(np.fromfile(str(self.image_path), dtype=np.uint8),
                            cv2.IMREAD_GRAYSCALE)
        if data is None:
            raise ValueError(f"cannot read crop {self.image_path}")
        return data


if TORCH_AVAILABLE:

    class CropDataset(Dataset):
        def __init__(self, samples: Sequence[CropSample], charset: Charset,
                     augment: bool = False, seed: int = 0):
            self.samples = list(samples)
            self.charset = charset
            self.augment = augment
            self.rng = random.Random(seed)

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            image = sample.load()
            if self.augment:
                image = augment_crop(image, self.rng)
            crop = prepare_crop(image)
            return crop, self.charset.encode(self.charset.filter(sample.text))

    class SyntheticLineDataset(Dataset):
        """Renders a fresh image every epoch, so the model never memorises one."""

        def __init__(self, corpus: Sequence[str], charset: Charset,
                     writers: Sequence[Writer], length: int = 20000,
                     seed: int = 0, augment: bool = True):
            self.corpus = list(corpus)
            self.charset = charset
            self.writers = list(writers)
            self.length = length
            self.seed = seed
            self.augment = augment

        def __len__(self) -> int:
            return self.length

        def __getitem__(self, index: int):
            rng = random.Random(self.seed * 1_000_003 + index)
            text = self.charset.filter(rng.choice(self.corpus))
            if not text:
                text = "Yes"
            writer = rng.choice(self.writers)
            image = render_line(text, writer, rng)
            if self.augment:
                image = augment_crop(image, rng, strength=rng.uniform(0.5, 1.2))
            return prepare_crop(image), self.charset.encode(text)

    def collate(batch):
        """Pad crops to a common width and flatten targets for CTC."""
        crops = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        width = max(c.shape[2] for c in crops)
        width = int(np.ceil(width / 4) * 4)
        padded = []
        for c in crops:
            if c.shape[2] < width:
                border = float(np.median(c[:, :, -1]))
                pad = np.full((1, c.shape[1], width - c.shape[2]), border, np.float32)
                c = np.concatenate([c, pad], axis=2)
            padded.append(c)
        images = torch.from_numpy(np.stack(padded))
        target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
        return images, flat, target_lengths


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
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


def evaluate(recognizer: HandwritingRecognizer, samples: Sequence[CropSample],
             batch_size: int = 32) -> dict[str, float]:
    """Character error rate and exact-match rate."""
    if not samples:
        return {"cer": 1.0, "exact_match": 0.0, "n": 0}

    total_chars = 0
    total_distance = 0
    exact = 0
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        crops = [s.load() for s in chunk]
        readings = recognizer.read(crops)
        for sample, reading in zip(chunk, readings):
            truth = recognizer.charset.filter(sample.text)
            total_distance += levenshtein(reading.text, truth)
            total_chars += max(1, len(truth))
            exact += int(reading.text == truth)
    return {
        "cer": total_distance / max(1, total_chars),
        "exact_match": exact / len(samples),
        "n": len(samples),
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(model, loader, validation: Sequence[CropSample], charset: Charset,
          epochs: int, learning_rate: float = 3e-4, device: str = "cpu",
          log_every: int = 50, stage: str = "train",
          patience: int = 3, checkpoint: Path | str | None = None) -> dict[str, Any]:
    """One training stage. Returns history and leaves the best weights loaded."""
    model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    ctc = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    history: list[dict[str, Any]] = []
    best_cer = float("inf")
    best_state = None
    since_improved = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        batches = 0
        for images, targets, target_lengths in loader:
            images = images.to(device)
            log_probs = model(images)                      # (T, N, C)
            input_lengths = torch.full((images.size(0),), log_probs.size(0),
                                       dtype=torch.long)
            loss = ctc(log_probs, targets, input_lengths, target_lengths)
            if not torch.isfinite(loss):
                continue                                   # skip a bad batch
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
            running += float(loss.detach())
            batches += 1
            if log_every and batches % log_every == 0:
                print(f"    {stage} epoch {epoch} batch {batches} "
                      f"loss={running / batches:.4f}", flush=True)
        scheduler.step()

        recognizer = HandwritingRecognizer(model=model, charset=charset)
        metrics = evaluate(recognizer, validation) if validation else {}
        model.train()
        row = {"epoch": epoch, "loss": running / max(1, batches), **metrics}
        history.append(row)
        print(f"  {stage} epoch {epoch}: loss={row['loss']:.4f}"
              + (f" cer={metrics['cer']:.4f} exact={metrics['exact_match']:.4f}"
                 if metrics else ""), flush=True)

        current = metrics.get("cer", row["loss"])
        if current < best_cer - 1e-4:
            best_cer = current
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since_improved = 0
            # Save every time it improves, not only at the end. Training this
            # on CPU takes hours, and losing all of it to an interruption --
            # or to someone stopping the job one epoch early -- is not an
            # acceptable failure mode.
            if checkpoint is not None:
                try:
                    torch.save({
                        "state_dict": best_state,
                        "charset": charset.characters,
                        "hidden": getattr(model.rnn, "hidden_size", 192),
                        "metrics": {"stage": stage, "epoch": epoch, **metrics},
                    }, Path(checkpoint))
                    print(f"    checkpoint saved (cer={best_cer:.4f})", flush=True)
                except OSError as exc:
                    print(f"    could not save checkpoint: {exc}", flush=True)
        else:
            since_improved += 1
            if since_improved >= patience:
                print(f"  {stage}: no improvement for {patience} epochs, stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "best_cer": best_cer}


def pretrain_synthetic(kn: Knowledge, charset: Charset, epochs: int = 6,
                       checkpoint: Path | str | None = None,
                       samples_per_epoch: int = 12000, batch_size: int = 32,
                       n_writers: int = 16, seed: int = 0,
                       hidden: int = 192,
                       num_workers: int = 4) -> tuple[Any, dict[str, Any]]:
    """Stage 1: learn to read handwriting-like text from the knowledge file."""
    rng = random.Random(seed)
    corpus = knowledge_corpus(kn, rng)
    writers = available_writers(n_writers, seed=seed)
    print(f"  corpus: {len(corpus)} distinct strings, {len(writers)} writer styles")

    dataset = SyntheticLineDataset(corpus, charset, writers,
                                   length=samples_per_epoch, seed=seed)
    # Rendering each line with PIL is the bottleneck on CPU, not the forward
    # pass, so the workers matter far more here than the batch size.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate, num_workers=num_workers,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=4 if num_workers > 0 else None)

    # A small held-out synthetic set, purely to watch the stage converge.
    holdout_rng = random.Random(seed + 999)
    holdout_writers = available_writers(4, seed=seed + 5000)
    holdout = []
    for _ in range(240):
        text = charset.filter(holdout_rng.choice(corpus)) or "Yes"
        writer = holdout_rng.choice(holdout_writers)
        holdout.append(CropSample(text=text,
                                  image=render_line(text, writer, holdout_rng),
                                  source="synthetic"))

    model = CRNN(charset.n_classes, hidden=hidden)
    result = train(model, loader, holdout, charset, epochs=epochs,
                   stage="pretrain", patience=3, checkpoint=checkpoint)
    return model, result


def finetune_real(model, train_samples: Sequence[CropSample],
                  validation: Sequence[CropSample], charset: Charset,
                  epochs: int = 30, batch_size: int = 16,
                  learning_rate: float = 1e-4, seed: int = 0) -> dict[str, Any]:
    """Stage 2: adapt to your advisers' actual handwriting."""
    dataset = CropDataset(train_samples, charset, augment=True, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, num_workers=0)
    return train(model, loader, validation, charset, epochs=epochs,
                 learning_rate=learning_rate, stage="finetune", patience=5)


__all__ = ["CropSample", "knowledge_corpus", "render_line", "augment_crop",
           "pretrain_synthetic", "finetune_real", "evaluate", "train",
           "levenshtein"]
