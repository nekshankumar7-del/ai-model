"""One normalisation path for crops that arrive from a dozen different places.

EMNIST is white ink on black, 28x28, already centred. NIST SD19 is black ink
on white, 128x128, bilevel. IIIT word images are grey photographs of varying
height. IAM lines are scanned at 300 dpi. They have to end up looking like the
same thing to the model, without being scrubbed so hard that the handwriting
stops being that person's handwriting.

The levels exist for exactly that reason:

  ``none``   geometry only - greyscale and resize.
  ``light``  (default) ink polarity, background flattening, contrast
             stretching. Reversible-looking operations that do not touch
             stroke shape.
  ``full``   adds deskew, denoise and adaptive thresholding. Use it for
             photographed pages; it is too aggressive for clean scans, where
             it eats thin strokes.

Thresholding is deliberately not in the default. A binarised stroke has lost
its pressure and its edge quality, and a CTC model trained on binarised crops
reads greyscale ones noticeably worse.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

Level = Literal["none", "light", "full"]


@dataclass
class PreprocessConfig:
    level: Level = "light"
    height: int = 32
    max_width: int = 512
    min_width: int = 8
    pad: int = 2
    deskew_limit: float = 12.0
    invert_if_dark_background: bool = True


DEFAULT = PreprocessConfig()


# --------------------------------------------------------------------------
# Individual operations
# --------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def normalise_polarity(image: np.ndarray) -> np.ndarray:
    """Make it dark ink on a light background, whatever it arrived as.

    The border is the honest estimate of background: ink does not usually
    touch all four edges, but it does dominate the centre of an EMNIST crop.
    """
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    if float(np.median(border)) < 128:
        return 255 - image
    return image


def flatten_background(image: np.ndarray, radius: int = 0) -> np.ndarray:
    """Divide out slow illumination changes - shadows, page curl, flash.

    Keeps stroke contrast where it is, which a global contrast stretch does
    not when one corner of the photograph is in shadow.
    """
    h, w = image.shape[:2]
    if radius <= 0:
        radius = max(15, (max(h, w) // 8) | 1)
    if radius % 2 == 0:
        radius += 1
    background = cv2.medianBlur(image, min(radius, 51))
    background = cv2.GaussianBlur(background.astype(np.float32), (0, 0), radius / 3.0)
    background = np.maximum(background, 1.0)
    flat = image.astype(np.float32) / background
    flat = np.clip(flat * 200.0, 0, 255)
    return flat.astype(np.uint8)


def stretch_contrast(image: np.ndarray, low: float = 2.0, high: float = 98.0
                     ) -> np.ndarray:
    """Percentile stretch. Percentiles, not min/max, so one dust speck cannot
    set the black point."""
    lo, hi = np.percentile(image, [low, high])
    if hi - lo < 8:
        return image
    out = (image.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def denoise(image: np.ndarray) -> np.ndarray:
    """Remove salt-and-pepper without rounding off stroke ends."""
    return cv2.medianBlur(image, 3)


def estimate_skew(image: np.ndarray, limit: float = 12.0) -> float:
    """Skew angle in degrees, from the minimum-area rectangle over the ink."""
    ink = 255 - image
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None or len(coords) < 24:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90
    if abs(angle) > limit:
        return 0.0
    return float(angle)


def deskew(image: np.ndarray, limit: float = 12.0) -> np.ndarray:
    angle = estimate_skew(image, limit)
    if abs(angle) < 0.25:
        return image
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def threshold(image: np.ndarray) -> np.ndarray:
    block = max(11, (min(image.shape[:2]) // 2) | 1)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, min(block, 51), 10)


def trim_margins(image: np.ndarray, pad: int = 2) -> np.ndarray:
    """Crop to the ink, then give it a small uniform margin back.

    Different corpora pad their crops differently; without this the model has
    to learn that an EMNIST 'A' and a NIST 'A' are the same letter at two
    different scales.
    """
    ink = 255 - image
    _, mask = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image
    x, y, w, h = cv2.boundingRect(coords)
    if w < 3 or h < 3:
        return image
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    return image[y0:y1, x0:x1]


def resize_keep_aspect(image: np.ndarray, height: int, max_width: int,
                       min_width: int = 8) -> np.ndarray:
    h, w = image.shape[:2]
    scale = height / max(1, h)
    new_w = int(round(w * scale))
    new_w = max(min_width, min(max_width, new_w))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, height), interpolation=interp)


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------

def normalise(image: np.ndarray, config: PreprocessConfig = DEFAULT,
              *, trim: bool = True) -> np.ndarray:
    """Greyscale, dark-on-light, fixed height, aspect preserved."""
    out = to_grayscale(image)
    if out.size == 0:
        return np.full((config.height, config.min_width), 255, np.uint8)

    if config.level != "none":
        if config.invert_if_dark_background:
            out = normalise_polarity(out)
        out = flatten_background(out)
        out = stretch_contrast(out)

    if config.level == "full":
        out = denoise(out)
        out = deskew(out, config.deskew_limit)
        out = threshold(out)

    if trim:
        out = trim_margins(out, config.pad)
    return resize_keep_aspect(out, config.height, config.max_width,
                              config.min_width)


def to_tensor_array(image: np.ndarray) -> np.ndarray:
    """(1, H, W) float32 in [-1, 1] - the model's input convention."""
    return (image.astype(np.float32) / 127.5 - 1.0)[None, :, :]


def pad_to_width(array: np.ndarray, width: int) -> np.ndarray:
    """Pad a (1, H, W) array on the right with its own border value."""
    if array.shape[2] >= width:
        return array[:, :, :width]
    border = float(np.median(array[:, :, -1]))
    pad = np.full((array.shape[0], array.shape[1], width - array.shape[2]),
                  border, np.float32)
    return np.concatenate([array, pad], axis=2)


__all__ = ["PreprocessConfig", "DEFAULT", "normalise", "to_tensor_array",
           "pad_to_width", "resize_keep_aspect", "trim_margins", "deskew",
           "estimate_skew", "flatten_background", "stretch_contrast",
           "normalise_polarity", "to_grayscale", "threshold", "denoise"]
