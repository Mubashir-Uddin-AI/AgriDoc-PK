# -*- coding: utf-8 -*-
"""Quantitative Severity Assessment Engine (Module 3).

Implements FR-3.1 and FR-3.2 from the AgriDoc-PK PRD:
  - CLAHE shadow compensation on the L-channel for Pakistani field lighting
  - HSV color thresholding to segment total leaf area and necrotic lesions
  - Surface damage percentage calculation
  - Three-tier triage classification (Mild / Moderate / Severe)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Triage thresholds (FR-3.2)
# ---------------------------------------------------------------------------
MILD_UPPER: float = 10.0
MODERATE_UPPER: float = 30.0

# ---------------------------------------------------------------------------
# HSV range definitions for Pakistani crop leaf segmentation
# ---------------------------------------------------------------------------

# Total vegetative leaf area: green + yellow-green + light brown tissue
# These ranges capture healthy and diseased vegetative pixels
LEAF_HSV_RANGES = [
    # Green foliage (healthy tissue)
    {"lower": np.array([25, 30, 30]), "upper": np.array([95, 255, 255])},
    # Yellow / chlorotic tissue
    {"lower": np.array([15, 30, 40]), "upper": np.array([25, 255, 255])},
    # Brown / necrotic tissue (still part of leaf surface)
    {"lower": np.array([5, 30, 20]), "upper": np.array([20, 255, 200])},
]

# Necrotic / diseased lesion tissue: rust pustules, blight spots, chlorosis
LESION_HSV_RANGES = [
    # Brown/rust pustules (wheat rust, rice blast lesions)
    {"lower": np.array([5, 50, 30]), "upper": np.array([20, 255, 200])},
    # Dark necrotic spots (bacterial blight, severe fungal)
    {"lower": np.array([0, 30, 10]), "upper": np.array([10, 255, 120])},
    # Yellowish chlorosis (viral chlorosis — CLCuV, early symptoms)
    {"lower": np.array([18, 80, 80]), "upper": np.array([30, 255, 255])},
]

# ---------------------------------------------------------------------------
# CLAHE configuration
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT: float = 3.0
CLAHE_TILE_SIZE: tuple = (8, 8)

# Morphological kernel for noise reduction
MORPH_KERNEL_SIZE: int = 5


@dataclass
class SeverityResult:
    """Structured severity assessment output."""

    infected_percentage: float
    level: str
    level_ur: str
    leaf_pixel_count: int
    lesion_pixel_count: int


def _apply_clahe_compensation(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE shadow compensation to normalize field lighting.

    Converts to LAB color space, applies CLAHE to the L-channel to
    equalize brightness variations caused by harsh Pakistani sunlight
    and deep canopy shadows, then converts back to BGR.

    Args:
        img_bgr: Input BGR image.

    Returns:
        Shadow-compensated BGR image.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE,
    )
    l_equalized = clahe.apply(l_channel)

    lab_equalized = cv2.merge([l_equalized, a_channel, b_channel])
    result = cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)
    logger.debug("CLAHE shadow compensation applied (clip=%.1f)", CLAHE_CLIP_LIMIT)
    return result


def _create_multi_range_mask(
    hsv: np.ndarray,
    ranges: list[dict],
) -> np.ndarray:
    """Create a binary mask by combining multiple HSV range thresholds.

    Args:
        hsv: HSV image array.
        ranges: List of dicts with 'lower' and 'upper' numpy arrays.

    Returns:
        Combined binary mask (uint8, values 0 or 255).
    """
    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for r in ranges:
        mask = cv2.inRange(hsv, r["lower"], r["upper"])
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Apply morphological operations to reduce noise in a binary mask.

    Uses opening (erosion then dilation) to remove small noise pixels,
    followed by closing (dilation then erosion) to fill small holes
    within genuine tissue regions.

    Args:
        mask: Binary mask (uint8).

    Returns:
        Cleaned binary mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    # Opening: remove small noise specks
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    # Closing: fill small holes within tissue
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def _classify_severity(percentage: float) -> tuple[str, str]:
    """Map infection percentage to triage level (FR-3.2).

    Args:
        percentage: Surface damage percentage (0.0 to 100.0).

    Returns:
        Tuple of (level_en, level_ur).
    """
    if percentage < MILD_UPPER:
        return "Mild", "ہلکا نقصان"
    elif percentage <= MODERATE_UPPER:
        return "Moderate", "درمیانی نقصان"
    else:
        return "Severe", "شدید نقصان"


def calculate_severity(img_bgr: np.ndarray) -> Dict[str, object]:
    """Calculate quantitative leaf disease severity.

    Full pipeline:
      1. CLAHE shadow compensation
      2. Convert to HSV
      3. Segment total leaf area (green + yellow + brown vegetative tissue)
      4. Segment necrotic/chlorotic lesions
      5. Compute S_pct = (lesion_pixels / leaf_pixels) × 100%
      6. Classify into Mild / Moderate / Severe

    Args:
        img_bgr: BGR NumPy image array of the leaf.

    Returns:
        Dictionary with keys:
          - infected_percentage (float): Damage ratio as percentage
          - level (str): English triage level
          - level_ur (str): Urdu triage level
          - leaf_pixel_count (int): Total leaf surface pixels
          - lesion_pixel_count (int): Necrotic/chlorotic pixels
    """
    # Step 1: CLAHE shadow compensation
    compensated = _apply_clahe_compensation(img_bgr)

    # Step 2: Convert to HSV color space
    hsv = cv2.cvtColor(compensated, cv2.COLOR_BGR2HSV)

    # Step 3: Segment total leaf area
    leaf_mask = _create_multi_range_mask(hsv, LEAF_HSV_RANGES)
    leaf_mask = _clean_mask(leaf_mask)

    # Step 4: Segment necrotic/chlorotic lesions
    lesion_mask = _create_multi_range_mask(hsv, LESION_HSV_RANGES)
    lesion_mask = _clean_mask(lesion_mask)

    # Ensure lesions are a subset of leaf area
    lesion_mask = cv2.bitwise_and(lesion_mask, leaf_mask)

    # Step 5: Count pixels and compute damage percentage
    leaf_pixels = int(np.count_nonzero(leaf_mask))
    lesion_pixels = int(np.count_nonzero(lesion_mask))

    if leaf_pixels == 0:
        logger.warning("No leaf tissue detected in image — returning 0%% severity")
        infected_pct = 0.0
    else:
        infected_pct = (lesion_pixels / leaf_pixels) * 100.0

    # Step 6: Classify triage level
    level, level_ur = _classify_severity(infected_pct)

    logger.info(
        "Severity: %.1f%% (%s) — leaf_px=%d, lesion_px=%d",
        infected_pct, level, leaf_pixels, lesion_pixels,
    )

    return {
        "infected_percentage": round(infected_pct, 1),
        "level": level,
        "level_ur": level_ur,
        "leaf_pixel_count": leaf_pixels,
        "lesion_pixel_count": lesion_pixels,
    }
