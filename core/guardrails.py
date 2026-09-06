# -*- coding: utf-8 -*-
"""Agrochemical Safety Guardrail Validator (FR-4.4).

Implements the deterministic post-generation guardrail that enforces a
0.00% chemical hallucination policy. Every chemical recommendation
produced by the LLM is cross-checked against the PARC-verified
whitelist before reaching the farmer.

Validation rules:
  1. Active ingredient MUST exist in parc_whitelist.json for the
     diagnosed disease.
  2. Dosage MUST NOT exceed the maximum allowable per acre.
  3. If validation fails, the chemical section is hard-overridden
     with the verified baseline record and an audit event is logged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Whitelist path resolution
# ---------------------------------------------------------------------------
_WHITELIST_PATH = (
    Path(__file__).resolve().parent.parent
    / "knowledge_base"
    / "parc_whitelist.json"
)

# Module-level cache
_whitelist_cache: Optional[Dict[str, Any]] = None


def load_whitelist(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and cache the PARC agrochemical whitelist.

    Args:
        path: Optional override path to the whitelist JSON file.
              Defaults to knowledge_base/parc_whitelist.json.

    Returns:
        Parsed whitelist dictionary.

    Raises:
        FileNotFoundError: If the whitelist file does not exist.
        json.JSONDecodeError: If the JSON is malformed.
    """
    global _whitelist_cache

    if _whitelist_cache is not None and path is None:
        return _whitelist_cache

    whitelist_path = Path(path) if path else _WHITELIST_PATH
    if not whitelist_path.exists():
        raise FileNotFoundError(
            f"PARC whitelist not found at: {whitelist_path}. "
            "The guardrail cannot operate without the verified chemical registry."
        )

    with open(whitelist_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if path is None:
        _whitelist_cache = data

    logger.info("PARC whitelist loaded from %s (%d diseases)",
                whitelist_path, len(data.get("diseases", {})))
    return data


def _normalize_ingredient(ingredient: str) -> str:
    """Normalize an active ingredient string for comparison.

    Strips whitespace, converts to lowercase, and removes percentage
    annotations so that 'Tebuconazole 50%' matches 'tebuconazole'.

    Args:
        ingredient: Raw active ingredient string.

    Returns:
        Normalized lowercase string.
    """
    normalized = ingredient.lower().strip()
    # Remove percentage patterns like "50%", "20 %"
    normalized = re.sub(r"\s*\d+(\.\d+)?\s*%", "", normalized)
    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_dose_value(dose_str: str) -> Optional[float]:
    """Extract the numeric dosage value from a dose string.

    Handles formats like:
      - "65 grams per 100 liters water"
      - "200 ml"
      - "500g"
      - "65"

    Args:
        dose_str: Dosage string or numeric value.

    Returns:
        Extracted numeric value, or None if unparseable.
    """
    if isinstance(dose_str, (int, float)):
        return float(dose_str)

    dose_str = str(dose_str).strip()
    match = re.match(r"^(\d+(?:\.\d+)?)", dose_str)
    if match:
        return float(match.group(1))
    return None


def _find_matching_chemical(
    active_ingredient: str,
    approved_chemicals: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find a whitelisted chemical matching the given active ingredient.

    Performs normalized substring matching — the LLM may produce
    partial ingredient names (e.g., 'Tebuconazole' instead of
    'Tebuconazole 50% + Trifloxystrobin 25%'), so we check if any
    normalized component of the whitelist entry matches.

    Args:
        active_ingredient: Active ingredient from the LLM output.
        approved_chemicals: List of approved chemical records.

    Returns:
        Matching whitelist chemical record, or None.
    """
    normalized_input = _normalize_ingredient(active_ingredient)

    for chemical in approved_chemicals:
        whitelist_ingredient = chemical.get("active_ingredient", "")
        normalized_whitelist = _normalize_ingredient(whitelist_ingredient)

        # Check exact match
        if normalized_input == normalized_whitelist:
            return chemical

        # Check if input is a subset component (e.g., "tebuconazole" in
        # "tebuconazole + trifloxystrobin")
        whitelist_components = [
            c.strip() for c in normalized_whitelist.split("+")
        ]
        if normalized_input in whitelist_components:
            return chemical

        # Check if the whitelist ingredient contains the input
        if normalized_input in normalized_whitelist:
            return chemical

        # Check product name match
        product_name = chemical.get("product_name", "").lower().strip()
        if normalized_input in product_name or product_name in normalized_input:
            return chemical

    return None


def validate_chemical(
    recommendation: Dict[str, Any],
    disease_id: str,
    whitelist: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate a single chemical recommendation against the PARC whitelist.

    Cross-checks:
      1. The active ingredient exists in the approved list for the disease.
      2. The dosage does not exceed the maximum allowable per acre.

    If validation fails, the recommendation is overridden with the
    first approved baseline chemical and an audit event is logged.

    Args:
        recommendation: LLM-generated chemical recommendation dict with
            at minimum 'active_ingredient' and 'dosage_per_acre' keys.
        disease_id: The classified disease identifier (e.g., 'wheat_leaf_rust').
        whitelist: Pre-loaded whitelist dict. If None, loads from disk.

    Returns:
        Validated (possibly overridden) recommendation dict with an
        added 'guardrail_verified' boolean and optional 'guardrail_action' field.
    """
    if whitelist is None:
        whitelist = load_whitelist()

    result = deepcopy(recommendation)
    diseases = whitelist.get("diseases", {})

    # If disease_id not in whitelist, override with audit
    if disease_id not in diseases:
        logger.warning(
            "GUARDRAIL: disease_id '%s' not found in whitelist. "
            "Cannot verify chemical recommendations.", disease_id
        )
        result["guardrail_verified"] = False
        result["guardrail_action"] = "disease_not_in_registry"
        return result

    disease_record = diseases[disease_id]
    approved_chemicals = disease_record.get("approved_chemicals", [])

    if not approved_chemicals:
        logger.warning("GUARDRAIL: No approved chemicals for disease '%s'", disease_id)
        result["guardrail_verified"] = False
        result["guardrail_action"] = "no_approved_chemicals"
        return result

    # Extract LLM's recommended active ingredient
    llm_ingredient = recommendation.get("active_ingredient", "")
    llm_dosage_raw = recommendation.get("dosage_per_acre", "")

    # Step 1: Check if active ingredient is in the approved list
    matched_chemical = _find_matching_chemical(llm_ingredient, approved_chemicals)

    if matched_chemical is None:
        # OVERRIDE: Unapproved chemical — replace with first baseline
        baseline = approved_chemicals[0]
        logger.warning(
            "GUARDRAIL OVERRIDE: LLM recommended unapproved ingredient '%s' "
            "for disease '%s'. Replacing with baseline '%s'.",
            llm_ingredient, disease_id, baseline["product_name"],
        )
        result["product_name"] = baseline["product_name"]
        result["active_ingredient"] = baseline["active_ingredient"]
        result["dosage_per_acre"] = f"{baseline['max_dose_per_acre']} {baseline['dose_unit']}"
        result["dose_unit"] = baseline["dose_unit"]
        result["pre_harvest_interval_days"] = baseline["pre_harvest_interval_days"]
        result["guardrail_verified"] = True
        result["guardrail_action"] = "ingredient_overridden"
        return result

    # Step 2: Check dosage is within maximum allowable
    llm_dose = _extract_dose_value(llm_dosage_raw)
    max_dose = matched_chemical["max_dose_per_acre"]

    if llm_dose is not None and llm_dose > max_dose:
        # OVERRIDE: Dosage exceeds safe limit — clamp to maximum
        logger.warning(
            "GUARDRAIL OVERRIDE: LLM dosage %.1f exceeds max %.1f %s for '%s'. "
            "Clamping to safe maximum.",
            llm_dose, max_dose, matched_chemical["dose_unit"],
            matched_chemical["product_name"],
        )
        result["dosage_per_acre"] = (
            f"{max_dose} {matched_chemical['dose_unit']}"
        )
        result["guardrail_verified"] = True
        result["guardrail_action"] = "dosage_clamped"
        return result

    # All checks passed
    result["guardrail_verified"] = True
    result["guardrail_action"] = "approved"
    logger.info(
        "GUARDRAIL PASS: '%s' at dose '%s' verified for '%s'",
        llm_ingredient, llm_dosage_raw, disease_id,
    )
    return result


def enforce_guardrail(
    llm_output: Dict[str, Any],
    disease_id: str,
    whitelist: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Full post-generation guardrail enforcement on LLM advisory output.

    Validates the chemical_treatment section of the LLM output. If the
    LLM generated an unapproved chemical or out-of-bounds dosage, the
    guardrail overrides the chemical section with the verified PARC
    baseline record.

    Args:
        llm_output: Complete LLM-generated advisory dictionary. Expected
            to contain a 'chemical_treatment' key with 'active_ingredient'
            and 'dosage_per_acre' sub-keys.
        disease_id: Classified disease identifier.
        whitelist: Pre-loaded whitelist dict. If None, loads from disk.

    Returns:
        Guardrail-validated advisory dictionary with verified chemical
        recommendations.
    """
    if whitelist is None:
        whitelist = load_whitelist()

    result = deepcopy(llm_output)

    # Extract the chemical treatment section
    chemical_section = result.get("chemical_treatment", {})

    if not chemical_section:
        logger.warning(
            "GUARDRAIL: No chemical_treatment section in LLM output for '%s'. "
            "Injecting baseline recommendation.", disease_id,
        )
        diseases = whitelist.get("diseases", {})
        if disease_id in diseases:
            approved = diseases[disease_id].get("approved_chemicals", [])
            if approved:
                baseline = approved[0]
                result["chemical_treatment"] = {
                    "product_name": baseline["product_name"],
                    "active_ingredient": baseline["active_ingredient"],
                    "dosage_per_acre": f"{baseline['max_dose_per_acre']} {baseline['dose_unit']}",
                    "dose_unit": baseline["dose_unit"],
                    "pre_harvest_interval_days": baseline["pre_harvest_interval_days"],
                    "guardrail_verified": True,
                    "guardrail_action": "baseline_injected",
                }
        return result

    # Validate the chemical recommendation
    validated = validate_chemical(chemical_section, disease_id, whitelist)
    result["chemical_treatment"] = validated

    return result
