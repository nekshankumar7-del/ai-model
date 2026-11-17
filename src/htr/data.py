"""Torch datasets over the unified manifests.

Three things here are worth knowing about:

1. **Every crop is widened to at least ``minimum_width(len(label))``.** CTC
   needs one frame per character plus a blank between repeats; a crop that is
   too narrow makes the loss infinite. ``zero_infinity`` hides that by turning
   the batch into a no-op, which looks exactly like a model that has stopped
   learning. Padding instead means the sample still trains.

2. **Batches are bucketed by width.** Handwriting crops run from 8 to 512
   pixels wide. Batching at random means padding almost every batch to its
   widest member and spending most of the compute on blank padding.

3. **Character corpora can be composed into words.** EMNIST and NIST SD19 are
   isolated glyphs, and training a CTC model on single glyphs teaches it a
   degenerate alignment. ``ComposedWordDataset`` stitches real glyphs from one
   writer into word images with realistic spacing and baseline jitter, which
   is how character data earns its place in a word-level model. It is opt-in.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Iterator, Sequence

import numpy as np

from .augment import AugmentConfig, augment as apply_augment
from .charset import UnicodeCharset
from .manifest import Sample, load_image
from .models import TORCH_AVAILABLE, minimum_width
from .preprocess import (PreprocessConfig, normalise, pad_to_width,
                         to_tensor_array)

if TORCH_AVAILABLE:
    import torch
    from torch.utils.data import DataLoader, Dataset, Sampler
else:                                                    # pragma: no cover
    Dataset = object
    Sampler = object


def _prepare(image: np.ndarray, text: str, config: PreprocessConfig
             ) -> np.ndarray:
    """Normalise to (1, H, W) and guarantee enough frames for the label."""
    crop = normalise(image, config)
    array = to_tensor_array(crop)
    needed = min(config.max_width, minimum_width(len(text)))
    if array.shape[2] < needed:
        array = pad_to_width(array, needed)
    return array


if TORCH_AVAILABLE:

    class ManifestDataset(Dataset):
        """One manifest row per item."""

        def __init__(self, samples: Sequence[Sample], charset: UnicodeCharset,
                     *, preprocess: PreprocessConfig | None = None,
                     augment: AugmentConfig | None = None, seed: int = 0):
            self.samples = list(samples)
            self.charset = charset
            self.preprocess = preprocess or PreprocessConfig()
            self.augment = augment
            self.seed = seed

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            image = load_image(sample)
            if self.augment is not None:
                rng = random.Random((self.seed * 1_000_003 + index) & 0xFFFFFFFF)
                image = apply_augment(image, rng, self.augment)
            text = self.charset.filter(sample.text)
            return _prepare(image, text, self.preprocess), self.charset.encode(text)

        def widths(self) -> list[int]:
            """A cheap width estimate for bucketing, without decoding images."""
            out = []
            for sample in self.samples:
                if sample.height and sample.width:
                    scaled = int(sample.width * self.preprocess.height
                                 / max(1, sample.height))
                else:
                    scaled = self.preprocess.height
                out.append(max(scaled, minimum_width(len(sample.text))))
            return out

    class ComposedWordDataset(Dataset):
        """Words built on the fly from isolated character crops.

        Glyphs are drawn from one writer per word where writer identity
        exists, so a composed word is one hand rather than a ransom note.
        Spacing, baseline jitter and per-glyph scale are randomised within
        the range real handwriting shows.
        """

        def __init__(self, samples: Sequence[Sample], charset: UnicodeCharset,
                     vocabulary: Sequence[str], *, length: int = 20000,
                     preprocess: PreprocessConfig | None = None,
                     augment: AugmentConfig | None = None, seed: int = 0):
            self.charset = charset
            self.vocabulary = [v for v in vocabulary if v]
            self.length = length
            self.preprocess = preprocess or PreprocessConfig()
            self.augment = augment
            self.seed = seed

            self.by_writer_char: dict[str, dict[str, list[Sample]]] = defaultdict(
                lambda: defaultdict(list))
            self.by_char: dict[str, list[Sample]] = defaultdict(list)
            for sample in samples:
                text = sample.text
                if len(text) != 1:
                    continue
                self.by_writer_char[sample.writer][text].append(sample)
                self.by_char[text].append(sample)
            self.writers = [w for w, chars in self.by_writer_char.items()
                            if len(chars) >= 8]
            self.available = set(self.by_char)

        def __len__(self) -> int:
            return self.length

        def _pick(self, char: str, writer: str | None,
                  rng: random.Random) -> Sample | None:
            if writer is not None:
                pool = self.by_writer_char.get(writer, {}).get(char)
                if pool:
                    return rng.choice(pool)
            pool = self.by_char.get(char)
            return rng.choice(pool) if pool else None

        def _word(self, rng: random.Random) -> tuple[np.ndarray, str] | None:
            for _ in range(8):
                text = rng.choice(self.vocabulary)
                if all(c in self.available or c == " " for c in text):
                    break
            else:
                return None

            writer = rng.choice(self.writers) if self.writers else None
            height = self.preprocess.height * 2
            glyphs: list[np.ndarray] = []
            kept: list[str] = []
            for char in text:
                if char == " ":
                    gap = rng.randint(int(height * 0.25), int(height * 0.5))
                    glyphs.append(np.full((height, gap), 255, np.uint8))
                    kept.append(" ")
                    continue
                sample = self._pick(char, writer, rng)
                if sample is None:
                    continue
                from .preprocess import normalise_polarity, to_grayscale, trim_margins
                glyph = trim_margins(normalise_polarity(to_grayscale(
                    load_image(sample))), pad=1)
                scale = rng.uniform(0.74, 1.0)
                target_h = max(6, int(height * scale))
                ratio = target_h / max(1, glyph.shape[0])
                import cv2
                glyph = cv2.resize(glyph,
                                   (max(3, int(glyph.shape[1] * ratio)), target_h),
                                   interpolation=cv2.INTER_AREA)
                glyphs.append(glyph)
                kept.append(char)
            if not any(c != " " for c in kept):
                return None

            # Lay the glyphs out on a shared baseline with a small overlap,
            # which is what makes joined-up writing look joined up.
            total = sum(g.shape[1] for g in glyphs)
            canvas = np.full((height, total + 2 * len(glyphs) + 8),
                             rng.randint(242, 255), np.uint8)
            x = 4
            baseline = int(height * 0.78)
            for glyph in glyphs:
                gh, gw = glyph.shape
                y = max(0, min(height - gh, baseline - gh + rng.randint(-2, 3)))
                region = canvas[y:y + gh, x:x + gw]
                canvas[y:y + gh, x:x + gw] = np.minimum(region, glyph)
                x += gw + rng.randint(-1, 3)
            return canvas[:, :x + 4], "".join(kept).strip()

        def __getitem__(self, index: int):
            rng = random.Random((self.seed * 1_000_003 + index) & 0xFFFFFFFF)
            made = self._word(rng)
            if made is None:
                image = np.full((self.preprocess.height * 2, 48), 250, np.uint8)
                text = ""
            else:
                image, text = made
            if self.augment is not None:
                image = apply_augment(image, rng, self.augment)
            text = self.charset.filter(text)
            return _prepare(image, text, self.preprocess), self.charset.encode(text)

    class BucketBatchSampler(Sampler):
        """Group similar widths into a batch; shuffle the batches, not the rows."""

        def __init__(self, widths: Sequence[int], batch_size: int,
                     shuffle: bool = True, seed: int = 0,
                     bucket_multiplier: int = 32):
            self.widths = list(widths)
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.seed = seed
            self.pool = batch_size * bucket_multiplier
            self.epoch = 0

        def __iter__(self) -> Iterator[list[int]]:
            order = list(range(len(self.widths)))
            rng = random.Random(self.seed + self.epoch)
            if self.shuffle:
                rng.shuffle(order)
            batches: list[list[int]] = []
            for start in range(0, len(order), self.pool):
                chunk = sorted(order[start:start + self.pool],
                               key=lambda i: self.widths[i])
                for b in range(0, len(chunk), self.batch_size):
                    batches.append(chunk[b:b + self.batch_size])
            if self.shuffle:
                rng.shuffle(batches)
            self.epoch += 1
            return iter(batches)

        def __len__(self) -> int:
            return (len(self.widths) + self.batch_size - 1) // self.batch_size

    def collate(batch):
        """Pad to a shared width; flatten targets the way CTC wants them."""
        crops = [b[0] for b in batch]
        targets = [b[1] for b in batch]
        width = max(c.shape[2] for c in crops)
        width = int(np.ceil(width / 4) * 4)
        padded = [pad_to_width(c, width) for c in crops]
        images = torch.from_numpy(np.stack(padded))
        lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
        flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
        return images, flat, lengths

    def make_loader(dataset, batch_size: int, *, shuffle: bool = True,
                    workers: int = 0, seed: int = 0, bucket: bool = True):
        if bucket and hasattr(dataset, "widths"):
            sampler = BucketBatchSampler(dataset.widths(), batch_size,
                                         shuffle=shuffle, seed=seed)
            return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate,
                              num_workers=workers,
                              persistent_workers=workers > 0,
                              prefetch_factor=4 if workers > 0 else None)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          collate_fn=collate, num_workers=workers,
                          persistent_workers=workers > 0,
                          prefetch_factor=4 if workers > 0 else None)


def batch_for_inference(images: Sequence[np.ndarray],
                        config: PreprocessConfig | None = None):
    """Prepare raw crops for a forward pass without any label context."""
    config = config or PreprocessConfig()
    arrays = [to_tensor_array(normalise(image, config)) for image in images]
    width = int(np.ceil(max(a.shape[2] for a in arrays) / 4) * 4)
    stacked = np.stack([pad_to_width(a, width) for a in arrays])
    return torch.from_numpy(stacked) if TORCH_AVAILABLE else stacked


__all__ = ["ManifestDataset", "ComposedWordDataset", "BucketBatchSampler",
           "collate", "make_loader", "batch_for_inference"]
