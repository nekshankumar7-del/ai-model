"""Augmentation for handwriting crops.

Every operation here is label-preserving by construction. Nothing flips,
mirrors, crops into the ink, or shuffles anything: a horizontal flip turns a
'b' into a 'd' and a crop turns "1000" into "100", and either one quietly
poisons the training set.

What the operations stand in for:

  shear + rotation      different slants; the same hand writing uphill
  elastic distortion    writing speed - fast hands wobble the baseline
  stroke thickness      pen type: fineliner, ballpoint, gel, marker
  scale + translation   writing size, and where in the crop it sits
  brightness / contrast / background texture / blur / JPEG
                        the camera, the lighting and the paper

Strength is a single dial so that a model can be trained on clean data with
light augmentation and fine-tuned on photographs with heavy augmentation,
without rewriting the policy.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AugmentConfig:
    strength: float = 1.0
    probability: float = 0.9      # chance that any augmentation runs at all
    allow_elastic: bool = True
    allow_perspective: bool = True
    allow_thickness: bool = True


DEFAULT = AugmentConfig()
LIGHT = AugmentConfig(strength=0.5, probability=0.7)
HEAVY = AugmentConfig(strength=1.4, probability=1.0)


def _border_value(image: np.ndarray) -> int:
    border = np.concatenate([image[0], image[-1]])
    return int(np.median(border))


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------

def shear(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    h, w = image.shape[:2]
    amount = rng.uniform(-0.3, 0.3) * strength
    matrix = np.float32([[1, amount, -amount * h / 2], [0, 1, 0]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_border_value(image))


def rotate(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    h, w = image.shape[:2]
    angle = rng.uniform(-4.0, 4.0) * strength
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_border_value(image))


def scale_translate(image: np.ndarray, rng: random.Random, strength: float
                    ) -> np.ndarray:
    h, w = image.shape[:2]
    factor = 1.0 + rng.uniform(-0.12, 0.12) * strength
    dx = rng.uniform(-0.04, 0.04) * w * strength
    dy = rng.uniform(-0.08, 0.08) * h * strength
    matrix = np.float32([[factor, 0, dx], [0, factor, dy]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT,
                          borderValue=_border_value(image))


def perspective(image: np.ndarray, rng: random.Random, strength: float
                ) -> np.ndarray:
    """A slight camera tilt. Kept small: a strong warp changes glyph identity."""
    h, w = image.shape[:2]
    m = 0.035 * strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[rng.uniform(0, m) * w, rng.uniform(0, m) * h],
                      [w - rng.uniform(0, m) * w, rng.uniform(0, m) * h],
                      [w - rng.uniform(0, m) * w, h - rng.uniform(0, m) * h],
                      [rng.uniform(0, m) * w, h - rng.uniform(0, m) * h]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=_border_value(image))


def elastic(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    """Smooth local warping - the baseline wobble of a hand moving quickly.

    Displacements are low-pass filtered so strokes bend rather than tear.
    """
    h, w = image.shape[:2]
    alpha = rng.uniform(2.0, 6.0) * strength
    sigma = max(3.0, min(h, w) / 6.0)
    seed = rng.randint(0, 2 ** 31 - 1)
    generator = np.random.default_rng(seed)
    dx = cv2.GaussianBlur(generator.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur(generator.uniform(-1, 1, (h, w)).astype(np.float32),
                          (0, 0), sigma) * alpha * 0.6
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
    return cv2.remap(image, grid_x + dx, grid_y + dy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


# --------------------------------------------------------------------------
# Ink and paper
# --------------------------------------------------------------------------

def stroke_thickness(image: np.ndarray, rng: random.Random, strength: float
                     ) -> np.ndarray:
    """Thicken or thin the ink - a marker versus a fineliner.

    Erode/dilate run on the inverted image because the ink is the dark side.
    """
    size = 2 if rng.random() < 0.3 * strength else 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size * 2 + 1,) * 2)
    ink = 255 - image
    ink = cv2.dilate(ink, kernel) if rng.random() < 0.5 else cv2.erode(ink, kernel)
    return 255 - ink


def brightness_contrast(image: np.ndarray, rng: random.Random, strength: float
                        ) -> np.ndarray:
    gain = 1.0 + rng.uniform(-0.22, 0.22) * strength
    bias = rng.uniform(-25, 18) * strength
    return np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def uneven_light(image: np.ndarray, rng: random.Random, strength: float
                 ) -> np.ndarray:
    h, w = image.shape[:2]
    left, right = rng.uniform(0.72, 1.05), rng.uniform(0.72, 1.05)
    ramp = np.linspace(left, right, w, dtype=np.float32)
    top, bottom = rng.uniform(0.85, 1.05), rng.uniform(0.85, 1.05)
    column = np.linspace(top, bottom, h, dtype=np.float32)
    field = 1.0 + (np.outer(column, ramp) - 1.0) * strength
    return np.clip(image.astype(np.float32) * field, 0, 255).astype(np.uint8)


def background_noise(image: np.ndarray, rng: random.Random, strength: float
                     ) -> np.ndarray:
    """Paper grain, sensor noise, and sometimes a ruled line."""
    h, w = image.shape[:2]
    out = image.astype(np.float32)
    out += np.random.default_rng(rng.randint(0, 2 ** 31 - 1)).normal(
        0, rng.uniform(2, 11) * strength, (h, w))
    if rng.random() < 0.18 * strength:
        y = rng.randint(int(h * 0.7), h - 1)
        out[y:y + 1, :] = np.minimum(out[y:y + 1, :], rng.uniform(150, 200))
    return np.clip(out, 0, 255).astype(np.uint8)


def blur(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    sigma = rng.uniform(0.3, 1.1) * strength
    if sigma < 0.1:
        return image
    return cv2.GaussianBlur(image, (0, 0), sigma)


def jpeg(image: np.ndarray, rng: random.Random, strength: float) -> np.ndarray:
    quality = int(rng.uniform(88 - 45 * strength, 92))
    ok, buffer = cv2.imencode(".jpg", image,
                              [int(cv2.IMWRITE_JPEG_QUALITY), max(15, quality)])
    return cv2.imdecode(buffer, cv2.IMREAD_GRAYSCALE) if ok else image


def resolution_loss(image: np.ndarray, rng: random.Random, strength: float
                    ) -> np.ndarray:
    h, w = image.shape[:2]
    factor = 1.0 - rng.uniform(0.1, 0.45) * strength
    small = cv2.resize(image, (max(4, int(w * factor)), max(4, int(h * factor))),
                       interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

def augment(image: np.ndarray, rng: random.Random,
            config: AugmentConfig = DEFAULT) -> np.ndarray:
    """Apply a random, realistic subset. Never changes what the crop says."""
    if rng.random() > config.probability:
        return image

    s = config.strength
    out = image.copy()

    # Geometry first, so later photometric noise is not itself warped.
    if rng.random() < 0.45:
        out = shear(out, rng, s)
    if rng.random() < 0.40:
        out = rotate(out, rng, s)
    if rng.random() < 0.35:
        out = scale_translate(out, rng, s)
    if config.allow_perspective and rng.random() < 0.18 * s:
        out = perspective(out, rng, s)
    if config.allow_elastic and rng.random() < 0.30 * s:
        out = elastic(out, rng, s)
    if config.allow_thickness and rng.random() < 0.30 * s:
        out = stroke_thickness(out, rng, s)

    # Then the camera and the paper.
    if rng.random() < 0.50:
        out = brightness_contrast(out, rng, s)
    if rng.random() < 0.35:
        out = uneven_light(out, rng, s)
    if rng.random() < 0.45:
        out = background_noise(out, rng, s)
    if rng.random() < 0.35:
        out = blur(out, rng, s)
    if rng.random() < 0.30:
        out = jpeg(out, rng, s)
    if rng.random() < 0.20:
        out = resolution_loss(out, rng, s)
    return out


__all__ = ["AugmentConfig", "DEFAULT", "LIGHT", "HEAVY", "augment", "elastic",
           "stroke_thickness", "shear", "rotate", "perspective",
           "brightness_contrast", "background_noise", "blur"]
