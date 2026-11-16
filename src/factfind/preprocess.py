"""Page preprocessing: get from a phone photo or PDF page to a clean,
deskewed, binarised image the OCR engine can read.

Steps, in order: load -> page crop (find the paper) -> perspective warp ->
deskew -> ruled-line removal -> adaptive threshold.

Each step is defensive: if it cannot find what it is looking for it returns
the input unchanged rather than mangling the page. `describe_quality` returns
the image-quality features the confidence model consumes.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

MAX_SIDE = 2200          # cap work size; bigger buys nothing for OCR
MIN_SIDE = 600


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_pages(path: Path | str, dpi: int = 200) -> list[np.ndarray]:
    """Load an image or every page of a PDF as BGR arrays."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, dpi=dpi)
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        # Fall back to PIL for formats OpenCV declines (some PNG/HEIC variants).
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception as exc:
            raise ValueError(f"cannot read image: {path} ({exc})") from exc
    return [img]


def load_pages_from_bytes(data: bytes, filename: str, dpi: int = 200) -> list[np.ndarray]:
    """Same as load_pages but for an in-memory upload (used by Streamlit)."""
    if filename.lower().endswith(".pdf"):
        import fitz
        pages = []
        with fitz.open(stream=data, filetype="pdf") as doc:
            for page in doc:
                pm = page.get_pixmap(dpi=dpi)
                arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
                if pm.n == 4:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
                elif pm.n == 3:
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                else:
                    arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
                pages.append(arr)
        return pages
    pil = Image.open(io.BytesIO(data)).convert("RGB")
    return [cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)]


def _load_pdf(path: Path, dpi: int) -> list[np.ndarray]:
    import fitz
    pages = []
    with fitz.open(str(path)) as doc:
        for page in doc:
            pm = page.get_pixmap(dpi=dpi)
            arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.h, pm.w, pm.n)
            if pm.n == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
            elif pm.n == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            pages.append(arr)
    return pages


def resize_to_work(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
    elif longest < MIN_SIDE:
        scale = MIN_SIDE / longest
    else:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


# --------------------------------------------------------------------------
# Page crop + perspective
# --------------------------------------------------------------------------

def find_page_quad(img: np.ndarray) -> np.ndarray | None:
    """Find the four corners of the sheet of paper, or None."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    page_area = h * w
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:6]:
        area = cv2.contourArea(cnt)
        # Must cover a real part of the frame but not be the whole frame
        # (which means we just found the image border).
        if area < page_area * 0.28 or area > page_area * 0.995:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order corners tl, tr, br, bl."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect


def warp_page(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    rect = order_quad(quad)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width < 50 or height < 50:
        return img
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                   dtype=np.float32)
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (width, height))


# --------------------------------------------------------------------------
# Deskew
# --------------------------------------------------------------------------

def estimate_skew(gray: np.ndarray) -> float:
    """Skew angle in degrees, from the dominant near-horizontal text lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120,
                            minLineLength=max(60, gray.shape[1] // 6), maxLineGap=12)
    if lines is None or len(lines) == 0:
        return 0.0
    # OpenCV 4 returns (N, 1, 4); OpenCV 5 returns (N, 4). Normalise both.
    segments = np.asarray(lines).reshape(-1, 4)
    angles = []
    for x1, y1, x2, y2 in segments:
        if x2 == x1:
            continue
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -20 < ang < 20:            # near-horizontal only
            angles.append(ang)
    if len(angles) < 5:
        return 0.0
    return float(np.median(angles))


def deskew(img: np.ndarray, max_angle: float = 15.0) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    angle = estimate_skew(gray)
    if abs(angle) < 0.25 or abs(angle) > max_angle:
        return img, 0.0
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle


# --------------------------------------------------------------------------
# Ruled lines and shadows
# --------------------------------------------------------------------------

def remove_ruled_lines(gray: np.ndarray) -> np.ndarray:
    """Erase long horizontal/vertical rules (lined paper, table borders)
    without eating the strokes of the handwriting that sits on them."""
    inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                cv2.THRESH_BINARY_INV, 25, 15)
    h, w = gray.shape[:2]
    out = gray.copy()
    for axis, length in (("h", max(25, w // 14)), ("v", max(25, h // 14))):
        kernel_size = (length, 1) if axis == "h" else (1, length)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        found = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
        # Thicken slightly so the whole rule goes, then inpaint from
        # surrounding paper rather than painting flat white over ink.
        found = cv2.dilate(found, np.ones((3, 3), np.uint8), iterations=1)
        if found.sum() == 0:
            continue
        out = cv2.inpaint(out, found, 3, cv2.INPAINT_TELEA)
    return out


def flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Remove the soft shadow gradient a phone photo leaves behind."""
    background = cv2.medianBlur(gray, 31)
    background = cv2.GaussianBlur(background, (0, 0), 15)
    flat = cv2.divide(gray, background, scale=255)
    return flat


def binarise(gray: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 31, 12)


# --------------------------------------------------------------------------
# Quality features
# --------------------------------------------------------------------------

@dataclass
class PageQuality:
    """Image-quality features fed to the confidence model."""
    width: int
    height: int
    blur_score: float        # variance of Laplacian; low = blurry
    contrast: float          # std of grey levels
    brightness: float        # mean grey level 0..255
    ink_ratio: float         # proportion of dark pixels after binarising
    skew_angle: float
    page_found: bool
    shadow_spread: float     # spread of local background level

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def describe_quality(gray: np.ndarray, skew_angle: float, page_found: bool) -> PageQuality:
    h, w = gray.shape[:2]
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    binary = binarise(gray)
    ink = float((binary < 128).mean())
    background = cv2.medianBlur(gray, 31)
    return PageQuality(
        width=int(w), height=int(h),
        blur_score=blur,
        contrast=float(gray.std()),
        brightness=float(gray.mean()),
        ink_ratio=ink,
        skew_angle=float(skew_angle),
        page_found=bool(page_found),
        shadow_spread=float(background.std()),
    )


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

@dataclass
class PreparedPage:
    colour: np.ndarray       # deskewed, warped BGR -- what crops are cut from
    gray: np.ndarray         # illumination-flattened, de-ruled greyscale
    binary: np.ndarray       # binarised, for layout work
    quality: PageQuality


def prepare_page(img: np.ndarray,
                 do_page_crop: bool = True,
                 do_deskew: bool = True,
                 do_line_removal: bool = True) -> PreparedPage:
    """Run the full preprocessing chain over one page image."""
    img = resize_to_work(img)

    page_found = False
    if do_page_crop:
        quad = find_page_quad(img)
        if quad is not None:
            img = warp_page(img, quad)
            img = resize_to_work(img)
            page_found = True

    angle = 0.0
    if do_deskew:
        img, angle = deskew(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = flatten_illumination(gray)
    if do_line_removal:
        gray = remove_ruled_lines(gray)

    quality = describe_quality(gray, angle, page_found)
    return PreparedPage(colour=img, gray=gray, binary=binarise(gray), quality=quality)


def prepare_file(path: Path | str, **kwargs) -> list[PreparedPage]:
    return [prepare_page(p, **kwargs) for p in load_pages(path)]


__all__ = ["PreparedPage", "PageQuality", "prepare_page", "prepare_file",
           "load_pages", "load_pages_from_bytes", "describe_quality", "binarise"]
