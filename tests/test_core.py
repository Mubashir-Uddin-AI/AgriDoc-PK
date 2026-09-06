# -*- coding: utf-8 -*-
"""Unit tests for AgriDoc-PK core modules.

Covers acceptance criteria:
  TC-01: Quality Gate — blurred image rejection
  TC-02: Quality Gate — dark/bright image rejection
  TC-05: Severity Engine — 50% necrotic leaf → 45-55% severity
  TC-07: Guardrail — hallucinated dosage override

Run with: python -m pytest tests/test_core.py -v
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import (
    MSG_BLURRY,
    MSG_TOO_BRIGHT,
    MSG_TOO_DARK,
    check_blur,
    check_exposure,
    resize_image,
    strip_exif,
    validate_image,
)
from core.severity_engine import calculate_severity
from core.guardrails import (
    enforce_guardrail,
    load_whitelist,
    validate_chemical,
    _normalize_ingredient,
    _extract_dose_value,
)


# ========================================================================
# FIXTURES — Synthetic test images
# ========================================================================


@pytest.fixture
def sharp_green_leaf():
    """Create a synthetic sharp green leaf image (passes all quality gates).

    Generates a 300×300 green image with strong edge patterns so that
    Laplacian variance is well above the blur threshold of 100.
    """
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Fill with mid-green (typical leaf color)
    img[:, :] = [30, 120, 50]  # BGR: dark-ish green

    # Add sharp high-frequency patterns to boost Laplacian variance
    for i in range(0, 300, 6):
        cv2.line(img, (i, 0), (i, 299), (10, 80, 30), 1)
    for j in range(0, 300, 6):
        cv2.line(img, (0, j), (299, j), (10, 80, 30), 1)

    # Add some bright spots to push mean luminance into valid range
    cv2.circle(img, (150, 150), 40, (60, 180, 80), -1)
    return img


@pytest.fixture
def blurry_image():
    """Create an intentionally blurry image (Laplacian var < 60).

    Applies extreme Gaussian blur to eliminate all edges.
    """
    img = np.full((300, 300, 3), 128, dtype=np.uint8)
    # Add some content then blur it away
    cv2.rectangle(img, (50, 50), (250, 250), (100, 160, 80), -1)
    # Heavy blur to push variance well below 100
    blurred = cv2.GaussianBlur(img, (51, 51), 25)
    return blurred


@pytest.fixture
def black_image():
    """Create a blank black image (mean luminance < 20)."""
    return np.zeros((300, 300, 3), dtype=np.uint8)


@pytest.fixture
def white_image():
    """Create a near-white overexposed image (mean luminance > 225)."""
    img = np.full((300, 300, 3), 240, dtype=np.uint8)
    return img


@pytest.fixture
def half_necrotic_leaf():
    """Create a synthetic leaf that is exactly 50% green and 50% brown/necrotic.

    Left half: healthy green leaf tissue (HSV green range).
    Right half: brown necrotic tissue (HSV brown/rust range).

    This directly tests TC-05: severity should be between 45% and 55%.
    """
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Left half: healthy green (H≈60, S≈150, V≈120 in HSV)
    # In BGR: approximately (30, 120, 30) → green leaf
    img[:, :150] = [30, 120, 30]   # BGR green

    # Right half: brown/necrotic (H≈12, S≈150, V≈100 in HSV)
    # In BGR: approximately (20, 60, 140) → brownish rust color
    img[:, 150:] = [20, 60, 140]   # BGR brown/rust

    return img


@pytest.fixture
def whitelist_data():
    """Load the PARC whitelist for guardrail tests."""
    return load_whitelist()


# ========================================================================
# TC-01: Quality Gate — Blur Detection
# ========================================================================


class TestBlurDetection:
    """TC-01: Intentionally blurred leaf photo must be rejected."""

    def test_blurry_image_rejected(self, blurry_image):
        """Blurred image (Var < 60) → rejected with Urdu steady-camera message."""
        is_valid, reason = validate_image(blurry_image)
        assert is_valid is False, "Blurry image should be rejected"
        assert reason == MSG_BLURRY

    def test_blurry_image_low_variance(self, blurry_image):
        """Verify the Laplacian variance is indeed below threshold."""
        is_sharp, variance = check_blur(blurry_image)
        assert is_sharp is False
        assert variance < 100.0, f"Expected variance < 100, got {variance}"

    def test_sharp_image_accepted(self, sharp_green_leaf):
        """Sharp image with strong edges should pass the blur gate."""
        is_sharp, variance = check_blur(sharp_green_leaf)
        assert is_sharp is True, f"Sharp image should pass (variance={variance})"
        assert variance >= 100.0


# ========================================================================
# TC-02: Quality Gate — Exposure (Dark / Bright)
# ========================================================================


class TestExposureDetection:
    """TC-02: Blank black photo must be rejected as too dark."""

    def test_dark_image_rejected(self, black_image):
        """Black image (μ < 20) → rejected by the quality gate.

        Note: A solid black image has zero Laplacian variance, so the
        blur gate catches it before the exposure gate. The critical
        assertion is that the image IS rejected. The exposure-specific
        check is tested separately via check_exposure.
        """
        is_valid, reason = validate_image(black_image)
        assert is_valid is False, "Black image should be rejected"
        # Black image may be rejected by blur (zero variance) or exposure
        assert reason in (MSG_BLURRY, MSG_TOO_DARK)

    def test_dark_image_exposure_direct(self, black_image):
        """Directly verify check_exposure rejects dark image (μ < 40)."""
        is_valid, mean_lum, reason = check_exposure(black_image)
        assert is_valid is False
        assert mean_lum < 40
        assert reason == MSG_TOO_DARK

    def test_overexposed_image_rejected(self, white_image):
        """White image (μ > 225) → rejected with 'too bright' warning."""
        is_valid, mean_lum, reason = check_exposure(white_image)
        assert is_valid is False
        assert mean_lum > 225
        assert reason == MSG_TOO_BRIGHT

    def test_normal_exposure_accepted(self, sharp_green_leaf):
        """Normal green leaf image should pass exposure check."""
        is_valid, mean_lum, reason = check_exposure(sharp_green_leaf)
        assert is_valid is True
        assert reason is None
        assert 40 <= mean_lum <= 225


# ========================================================================
# TC-01 + TC-02: Combined Quality Gate Validation
# ========================================================================


class TestFullQualityGate:
    """Combined quality gate pipeline tests."""

    def test_valid_image_passes_all_gates(self, sharp_green_leaf):
        """A clear, well-exposed image passes the full quality gate."""
        is_valid, reason = validate_image(sharp_green_leaf)
        assert is_valid is True, f"Valid image should pass (reason={reason})"
        assert reason is None

    def test_blur_checked_before_exposure(self, blurry_image):
        """Blur check runs first — even if exposure is fine, blur rejects."""
        is_valid, reason = validate_image(blurry_image)
        assert is_valid is False
        assert reason == MSG_BLURRY  # Blur caught first, not exposure


# ========================================================================
# TC-05: Severity Engine — 50% Necrotic Leaf
# ========================================================================


class TestSeverityEngine:
    """TC-05: Leaf with ~50% necrotic spots → 45-55% severity."""

    def test_severity_50_percent(self, half_necrotic_leaf):
        """Synthetic 50/50 green-brown leaf → severity between 45% and 55%."""
        result = calculate_severity(half_necrotic_leaf)

        pct = result["infected_percentage"]
        assert 45.0 <= pct <= 55.0, (
            f"Expected severity between 45-55%, got {pct}%"
        )

    def test_severity_50_percent_is_severe(self, half_necrotic_leaf):
        """50% infection should classify as 'Severe' (> 30% threshold)."""
        result = calculate_severity(half_necrotic_leaf)
        assert result["level"] == "Severe"
        assert result["level_ur"] == "شدید نقصان"

    def test_severity_triage_mild(self):
        """A mostly green image should classify as 'Mild'."""
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        img[:, :] = [30, 120, 30]  # All green
        result = calculate_severity(img)
        assert result["level"] == "Mild"
        assert result["infected_percentage"] < 10.0

    def test_severity_returns_pixel_counts(self, half_necrotic_leaf):
        """Severity result must include raw pixel counts."""
        result = calculate_severity(half_necrotic_leaf)
        assert "leaf_pixel_count" in result
        assert "lesion_pixel_count" in result
        assert result["leaf_pixel_count"] > 0

    def test_severity_empty_image(self):
        """An image with no leaf tissue → 0% severity."""
        # Pure blue image (no leaf HSV ranges match)
        img = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        result = calculate_severity(img)
        assert result["infected_percentage"] == 0.0


# ========================================================================
# TC-07: Guardrail Validator — Hallucinated Dosage Override
# ========================================================================


class TestGuardrailValidator:
    """TC-07: Hallucinated 500g Nativo → overridden to 65g max safe limit."""

    def test_hallucinated_dosage_overridden(self, whitelist_data):
        """500g Nativo for wheat_leaf_rust → clamped to 65g max."""
        hallucinated = {
            "product_name": "Nativo 75 WG",
            "active_ingredient": "Tebuconazole 50% + Trifloxystrobin 25%",
            "dosage_per_acre": "500 grams per 100 liters water",
        }

        result = validate_chemical(
            hallucinated,
            disease_id="wheat_leaf_rust",
            whitelist=whitelist_data,
        )

        assert result["guardrail_verified"] is True
        assert result["guardrail_action"] == "dosage_clamped"
        # Verify dose was clamped to 65
        assert "65" in result["dosage_per_acre"]

    def test_valid_dosage_approved(self, whitelist_data):
        """Valid 50g Nativo passes guardrail without modification."""
        valid_rec = {
            "product_name": "Nativo 75 WG",
            "active_ingredient": "Tebuconazole 50% + Trifloxystrobin 25%",
            "dosage_per_acre": "50 grams per 100 liters water",
        }

        result = validate_chemical(
            valid_rec,
            disease_id="wheat_leaf_rust",
            whitelist=whitelist_data,
        )

        assert result["guardrail_verified"] is True
        assert result["guardrail_action"] == "approved"

    def test_unknown_chemical_overridden(self, whitelist_data):
        """Unapproved chemical → overridden with baseline approved chemical."""
        fake_chemical = {
            "product_name": "FakePesticide 9000",
            "active_ingredient": "Nonexistentium 99%",
            "dosage_per_acre": "100 grams",
        }

        result = validate_chemical(
            fake_chemical,
            disease_id="wheat_leaf_rust",
            whitelist=whitelist_data,
        )

        assert result["guardrail_verified"] is True
        assert result["guardrail_action"] == "ingredient_overridden"
        # Must be replaced with the first baseline chemical (Nativo)
        assert result["active_ingredient"] == "Tebuconazole 50% + Trifloxystrobin 25%"

    def test_enforce_guardrail_full_pipeline(self, whitelist_data):
        """Full enforce_guardrail pipeline with hallucinated LLM output."""
        llm_output = {
            "summary_ur": "Test advisory",
            "chemical_treatment": {
                "product_name": "Nativo 75 WG",
                "active_ingredient": "Tebuconazole 50% + Trifloxystrobin 25%",
                "dosage_per_acre": "500 grams",
            },
        }

        result = enforce_guardrail(
            llm_output,
            disease_id="wheat_leaf_rust",
            whitelist=whitelist_data,
        )

        chem = result["chemical_treatment"]
        assert chem["guardrail_verified"] is True
        assert chem["guardrail_action"] == "dosage_clamped"
        assert "65" in chem["dosage_per_acre"]

    def test_enforce_guardrail_missing_chemical_section(self, whitelist_data):
        """LLM output with no chemical_treatment → baseline injected."""
        llm_output = {"summary_ur": "Test advisory"}

        result = enforce_guardrail(
            llm_output,
            disease_id="wheat_leaf_rust",
            whitelist=whitelist_data,
        )

        assert "chemical_treatment" in result
        chem = result["chemical_treatment"]
        assert chem["guardrail_verified"] is True
        assert chem["guardrail_action"] == "baseline_injected"


# ========================================================================
# Utility function tests
# ========================================================================


class TestUtilities:
    """Tests for preprocessor and guardrail utility functions."""

    def test_resize_large_image(self):
        """Image larger than 1024px gets resized."""
        big = np.zeros((2000, 3000, 3), dtype=np.uint8)
        resized = resize_image(big, max_size=1024)
        h, w = resized.shape[:2]
        assert max(h, w) <= 1024

    def test_resize_small_image_unchanged(self):
        """Image within bounds is returned unchanged."""
        small = np.zeros((200, 300, 3), dtype=np.uint8)
        resized = resize_image(small, max_size=1024)
        assert resized.shape == small.shape

    def test_exif_stripping(self):
        """EXIF strip produces valid JPEG bytes without metadata."""
        # Create a minimal JPEG in memory
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        success, buf = cv2.imencode(".jpg", img)
        assert success

        cleaned = strip_exif(buf.tobytes())
        assert len(cleaned) > 0
        # Verify it's valid JPEG (starts with FF D8)
        assert cleaned[:2] == b'\xff\xd8'

    def test_normalize_ingredient(self):
        """Ingredient normalization strips percentages and whitespace."""
        assert _normalize_ingredient("Tebuconazole 50%") == "tebuconazole"
        assert _normalize_ingredient("  Azoxystrobin 20% + Difenoconazole 12.5%  ") == "azoxystrobin + difenoconazole"

    def test_extract_dose_value(self):
        """Dose extraction handles various format strings."""
        assert _extract_dose_value("65 grams per 100 liters") == 65.0
        assert _extract_dose_value("200 ml") == 200.0
        assert _extract_dose_value(65) == 65.0
        assert _extract_dose_value("invalid") is None
