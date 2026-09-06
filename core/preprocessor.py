# -*- coding: utf-8 -*-
"""Image Ingestion & Edge Quality Gate (Module 1).

Implements FR-1.1 through FR-1.4 from the AgriDoc-PK PRD:
  - Client-side resizing to max 1024×1024 px
  - EXIF metadata sanitization (GPS/device serial stripping)
  - Laplacian-variance blur detection (threshold 100.0)
  - Mean luminance exposure guards (reject < 40 or > 225)
"""

from __future__ import annotations

import io
import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urdu rejection messages (field-friendly, conversational tone)
# ---------------------------------------------------------------------------
MSG_BLURRY = "تصویر واضح نہیں ہے۔ کیمرہ ساکت رکھ کر دوبارہ تصویر بنائیں۔"
MSG_TOO_DARK = "تصویر بہت تاریک ہے۔ روشنی میں دوبارہ تصویر لیں۔"
MSG_TOO_BRIGHT = "تصویر بہت زیادہ روشن ہے۔ سورج کی چمک سے ہٹ کر تصویر لیں۔"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MAX_SIZE: int = 1024
BLUR_THRESHOLD: float = 100.0
EXPOSURE_LOW: int = 40
EXPOSURE_HIGH: int = 225
MAX_UPLOAD_DIMENSION: int = 4096
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024  # 10 MB


def resize_image(img: np.ndarray, max_size: int = DEFAULT_MAX_SIZE) -> np.ndarray:
    """Resize image so its longest side is at most *max_size* pixels.

    Preserves aspect ratio using high-quality INTER_AREA downscaling.
    If the image is already within bounds it is returned unchanged.

    Args:
        img: BGR NumPy array (H×W×3).
        max_size: Maximum allowed dimension in pixels.

    Returns:
        Resized BGR NumPy array.
    """
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img

    scale = max_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.debug("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)
    return resized


def strip_exif(image_bytes: bytes) -> bytes:
    """Remove all EXIF, GPS, and device metadata from JPEG/PNG bytes.

    Re-encodes the image through Pillow which drops all auxiliary
    metadata chunks, preventing geolocation or device serial leaks
    (Security requirement from PRD Section 8.1).

    Args:
        image_bytes: Raw image file bytes.

    Returns:
        Sanitized JPEG bytes with no EXIF tags.
    """
    pil_img = Image.open(io.BytesIO(image_bytes))
    # Create a fresh image by copying pixel data only (no metadata).
    # Using .copy() on a new canvas avoids the deprecated getdata() API.
    clean = Image.new(pil_img.mode, pil_img.size)
    clean.paste(pil_img)

    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=92)
    buf.seek(0)
    logger.debug("EXIF metadata stripped from %d byte image", len(image_bytes))
    return buf.read()


def check_blur(img: np.ndarray, threshold: float = BLUR_THRESHOLD) -> Tuple[bool, float]:
    """Detect image blur using Laplacian variance.

    Computes Var(∇²I) — the variance of the Laplacian. A low variance
    indicates the image lacks sharp edges and is likely out-of-focus.

    Args:
        img: BGR NumPy array.
        threshold: Minimum acceptable Laplacian variance (default 100.0).

    Returns:
        Tuple of (is_sharp: bool, variance: float).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = float(laplacian.var())
    is_sharp = variance >= threshold
    logger.debug("Blur check: variance=%.2f, threshold=%.1f, sharp=%s",
                 variance, threshold, is_sharp)
    return is_sharp, variance


def check_exposure(
    img: np.ndarray,
    low: int = EXPOSURE_LOW,
    high: int = EXPOSURE_HIGH,
) -> Tuple[bool, float, str | None]:
    """Validate image exposure via mean grayscale luminance.

    Rejects images that are either too dark (underexposed, field shade)
    or too bright (overexposed, direct sunlight wash).

    Args:
        img: BGR NumPy array.
        low: Minimum acceptable mean luminance (default 40).
        high: Maximum acceptable mean luminance (default 225).

    Returns:
        Tuple of (is_valid: bool, mean_luminance: float, rejection_reason_ur: str | None).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_luminance = float(gray.mean())

    if mean_luminance < low:
        logger.debug("Exposure FAIL: too dark (mean=%.1f < %d)", mean_luminance, low)
        return False, mean_luminance, MSG_TOO_DARK
    if mean_luminance > high:
        logger.debug("Exposure FAIL: too bright (mean=%.1f > %d)", mean_luminance, high)
        return False, mean_luminance, MSG_TOO_BRIGHT

    logger.debug("Exposure OK: mean=%.1f", mean_luminance)
    return True, mean_luminance, None


def validate_image(img: np.ndarray) -> Tuple[bool, str | None]:
    """Run the full Edge Quality Gate pipeline on an image.

    Orchestrates blur detection and exposure validation in sequence.
    Returns early on first failure with the appropriate Urdu message.

    Args:
        img: BGR NumPy array (raw camera/upload frame).

    Returns:
        Tuple of (is_valid: bool, rejection_reason_ur: str | None).
        If is_valid is True, rejection_reason_ur is None.
    """
    # Gate 1: Blur detection (FR-1.3)
    is_sharp, variance = check_blur(img)
    if not is_sharp:
        logger.info("Image rejected: blurry (variance=%.2f)", variance)
        return False, MSG_BLURRY

    # Gate 2: Exposure validation (FR-1.4)
    is_exposed, mean_lum, reason = check_exposure(img)
    if not is_exposed:
        logger.info("Image rejected: exposure (mean=%.1f, reason=%s)", mean_lum, reason)
        return False, reason

    logger.info("Image passed quality gate (blur_var=%.1f, lum=%.1f)", variance, mean_lum)
    return True, None
