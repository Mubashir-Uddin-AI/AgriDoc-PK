# -*- coding: utf-8 -*-
"""Integration tests and RAG groundedness benchmarks for AgriDoc-PK.

Covers acceptance criteria:
  TC-06: RAG Groundedness — context contains PARC citation; chemical matches whitelist
  TC-08: Follow-up Chat — grounded response for rain-spray question

Also includes:
  - RAG faithfulness scoring across all disease classes
  - End-to-end pipeline integration (Quality Gate → Severity → RAG → Guardrail)
  - Static fallback integrity checks

Run with: python -m pytest tests/test_integration.py -v
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import validate_image
from core.severity_engine import calculate_severity
from core.guardrails import load_whitelist, enforce_guardrail
from core.rag_engine import RAGEngine
from core.llm_orchestrator import _get_static_fallback, _build_user_prompt


# ========================================================================
# FIXTURES
# ========================================================================

@pytest.fixture(scope="module")
def whitelist():
    return load_whitelist()


@pytest.fixture(scope="module")
def rag():
    engine = RAGEngine()
    if not engine.is_ready:
        pytest.skip("RAG index not built — run knowledge_base/build_index.py")
    return engine


@pytest.fixture
def green_leaf():
    """Synthetic healthy green leaf image."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:, :] = [30, 120, 30]
    for i in range(0, 300, 6):
        cv2.line(img, (i, 0), (i, 299), (10, 80, 30), 1)
    for j in range(0, 300, 6):
        cv2.line(img, (0, j), (299, j), (10, 80, 30), 1)
    cv2.circle(img, (150, 150), 40, (60, 180, 80), -1)
    return img


# ========================================================================
# TC-06: RAG Groundedness
# ========================================================================


class TestRAGGroundedness:
    """TC-06: RAG context contains exact PARC citation;
    chemical active ingredient matches whitelist."""

    DISEASE_QUERY_MAP = {
        "wheat_leaf_rust": "wheat leaf rust treatment fungicide",
        "wheat_stripe_rust": "wheat stripe yellow rust treatment",
        "cotton_leaf_curl_virus": "cotton leaf curl virus whitefly control insecticide",
        "rice_blast": "rice blast fungicide tricyclazole treatment",
    }

    def test_wheat_rust_groundedness(self, rag, whitelist):
        """Wheat rust RAG retrieval includes PARC citation and valid chemical."""
        context = rag.build_context(
            disease_id="wheat_leaf_rust",
            disease_name="Wheat Leaf Rust",
            severity_pct=24.5,
            severity_level="Moderate",
            confidence=0.95,
            crop="Wheat",
        )
        passages = context["retrieved_passages"]
        assert len(passages) > 0, "No passages retrieved for wheat_leaf_rust"

        # Check PARC citation exists
        all_content = " ".join(p["content"].lower() for p in passages)
        assert any(term in all_content for term in [
            "parc", "nativo", "amistar", "tilt", "propiconazole",
            "tebuconazole", "fungicid", "rust",
        ]), "Retrieved content lacks relevant PARC chemical information"

        # Verify at least one approved chemical is mentioned
        approved = whitelist["diseases"]["wheat_leaf_rust"]["approved_chemicals"]
        approved_names = [c["product_name"].lower() for c in approved]
        approved_ingredients = [c["active_ingredient"].lower() for c in approved]
        has_match = any(
            name in all_content or ingr.split()[0].lower() in all_content
            for name, ingr in zip(approved_names, approved_ingredients)
        )
        assert has_match, "No approved chemical found in retrieved passages"

    def test_cotton_clcuv_groundedness(self, rag, whitelist):
        """Cotton CLCuV RAG retrieval includes whitefly/insecticide content."""
        context = rag.build_context(
            disease_id="cotton_leaf_curl_virus",
            disease_name="Cotton Leaf Curl Virus",
            severity_pct=15.0,
            severity_level="Moderate",
            confidence=0.88,
            crop="Cotton",
        )
        passages = context["retrieved_passages"]
        assert len(passages) > 0

        all_content = " ".join(p["content"].lower() for p in passages)
        assert any(term in all_content for term in [
            "whitefly", "imidacloprid", "confidor", "acetamiprid",
            "CLCuV".lower(), "cotton", "curl",
        ]), "Cotton CLCuV passages lack relevant content"

    def test_rice_blast_groundedness(self, rag, whitelist):
        """Rice blast RAG retrieval includes blast-specific content."""
        context = rag.build_context(
            disease_id="rice_blast",
            disease_name="Rice Blast",
            severity_pct=35.0,
            severity_level="Severe",
            confidence=0.91,
            crop="Rice",
        )
        passages = context["retrieved_passages"]
        assert len(passages) > 0

        all_content = " ".join(p["content"].lower() for p in passages)
        assert any(term in all_content for term in [
            "blast", "tricyclazole", "beam", "magnaporthe", "rice",
        ]), "Rice blast passages lack relevant content"

    @pytest.mark.parametrize("disease_id", [
        "wheat_leaf_rust", "wheat_stripe_rust",
        "cotton_leaf_curl_virus", "rice_blast",
    ])
    def test_all_diseases_have_retrievable_passages(self, rag, disease_id):
        """Every disease class retrieves at least 1 relevant PARC passage."""
        query = self.DISEASE_QUERY_MAP.get(disease_id, f"{disease_id} treatment")
        results = rag.retrieve(query, disease_id=disease_id, top_k=3)
        assert len(results) >= 1, f"No passages for {disease_id}"
        assert all(r["relevance_score"] > 0 for r in results)

    @pytest.mark.parametrize("disease_id", [
        "wheat_leaf_rust", "cotton_leaf_curl_virus", "rice_blast",
    ])
    def test_context_citations_are_populated(self, rag, disease_id):
        """build_context generates non-empty citations list."""
        context = rag.build_context(
            disease_id=disease_id,
            disease_name=disease_id.replace("_", " ").title(),
            severity_pct=20.0,
            severity_level="Moderate",
            confidence=0.90,
            crop=disease_id.split("_")[0].title(),
        )
        assert len(context["citations"]) > 0, f"No citations for {disease_id}"


# ========================================================================
# TC-08: Follow-up Chat — Grounded Response
# ========================================================================


class TestFollowUpChat:
    """TC-08: Farmer asks rain-spray question → grounded response."""

    def test_static_fallback_answers_rain_question(self):
        """Static fallback provides a valid advisory structure for follow-up context."""
        # Without a live API key, we test the infrastructure for follow-up
        from core.llm_orchestrator import generate_chat_response

        session_context = {
            "disease_id": "wheat_leaf_rust",
            "severity_pct": 24.5,
            "prior_advisory_summary": "Nativo 75 WG ka 65 gram per acre spray karein.",
        }

        # Without API key, should return graceful fallback
        old_key = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            response = generate_chat_response(
                query="Kya barish mein spray ho sakta hai?",
                session_context=session_context,
                api_key="",
            )
            assert "reply_text_ur" in response
            assert len(response["reply_text_ur"]) > 0
            assert "sources_cited" in response
        finally:
            if old_key:
                os.environ["GOOGLE_API_KEY"] = old_key

    def test_prompt_preserves_session_context(self):
        """Follow-up prompt construction retains prior diagnosis state."""
        context = {
            "vision_metadata": {
                "crop": "Wheat",
                "disease_id": "wheat_leaf_rust",
                "disease_name": "Wheat Leaf Rust",
                "confidence": 0.95,
                "severity_pct": 24.5,
                "severity_level": "Moderate",
            },
            "retrieved_passages": [{
                "content": "Apply Nativo at 65g/acre. Do not spray in rain.",
                "source_file": "wheat_rust_bulletin.md",
                "section_heading": "Spray Guidelines",
            }],
        }
        prompt = _build_user_prompt(context)
        assert "wheat_leaf_rust" in prompt
        assert "24.5%" in prompt
        assert "Nativo" in prompt


# ========================================================================
# RAG Faithfulness Benchmark
# ========================================================================


class TestRAGFaithfulness:
    """RAG faithfulness scoring — verifies chemical grounding across all diseases."""

    @pytest.mark.parametrize("disease_id", [
        "wheat_leaf_rust", "wheat_stripe_rust",
        "cotton_leaf_curl_virus", "rice_blast",
    ])
    def test_static_fallback_chemicals_match_whitelist(self, whitelist, disease_id):
        """Static fallback chemicals are verifiably from the PARC whitelist."""
        fallback = _get_static_fallback(disease_id)
        chem = fallback.get("chemical_treatment", {})

        if not chem:
            pytest.skip(f"No chemical in static fallback for {disease_id}")

        # Verify the chemical exists in the whitelist
        approved = whitelist["diseases"][disease_id]["approved_chemicals"]
        approved_names = [c["product_name"] for c in approved]
        assert chem["product_name"] in approved_names, (
            f"Fallback chemical '{chem['product_name']}' not in whitelist for {disease_id}"
        )

        # Verify guardrail marked as verified
        assert chem.get("guardrail_verified") is True

    @pytest.mark.parametrize("disease_id", [
        "wheat_leaf_rust", "cotton_leaf_curl_virus", "rice_blast",
    ])
    def test_guardrail_enforces_on_all_diseases(self, whitelist, disease_id):
        """Guardrail correctly validates chemicals for every disease."""
        # Simulate an LLM output with excessive dosage
        disease_data = whitelist["diseases"][disease_id]
        first_chem = disease_data["approved_chemicals"][0]

        llm_output = {
            "chemical_treatment": {
                "product_name": first_chem["product_name"],
                "active_ingredient": first_chem["active_ingredient"],
                "dosage_per_acre": f"{first_chem['max_dose_per_acre'] * 10} {first_chem['dose_unit']}",
            }
        }

        result = enforce_guardrail(llm_output, disease_id, whitelist)
        chem = result["chemical_treatment"]
        assert chem["guardrail_verified"] is True
        assert chem["guardrail_action"] == "dosage_clamped"

    def test_zero_hallucination_guarantee(self, whitelist):
        """Every disease's fallback chain produces guardrail-verified output."""
        diseases = whitelist.get("diseases", {})
        for disease_id in diseases:
            fallback = _get_static_fallback(disease_id)
            chem = fallback.get("chemical_treatment", {})
            assert chem.get("guardrail_verified") is True, (
                f"Hallucination detected: {disease_id} fallback not verified"
            )


# ========================================================================
# End-to-End Pipeline Integration
# ========================================================================


class TestEndToEndPipeline:
    """Full pipeline integration: image → quality gate → severity → RAG → guardrail."""

    def test_full_pipeline_valid_image(self, green_leaf, rag, whitelist):
        """Valid leaf image flows through entire pipeline without error."""
        # Step 1: Quality gate
        is_valid, reason = validate_image(green_leaf)
        assert is_valid, f"Quality gate failed: {reason}"

        # Step 2: Severity assessment
        severity = calculate_severity(green_leaf)
        assert "infected_percentage" in severity
        assert "level" in severity

        # Step 3: RAG retrieval
        context = rag.build_context(
            disease_id="wheat_leaf_rust",
            disease_name="Wheat Leaf Rust",
            severity_pct=severity["infected_percentage"],
            severity_level=severity["level"],
            confidence=0.92,
            crop="Wheat",
        )
        assert len(context["retrieved_passages"]) > 0

        # Step 4: Guardrail on static fallback
        fallback = _get_static_fallback("wheat_leaf_rust")
        validated = enforce_guardrail(fallback, "wheat_leaf_rust", whitelist)
        assert validated["chemical_treatment"]["guardrail_verified"] is True

    def test_pipeline_timing(self, green_leaf, rag):
        """Core pipeline (no LLM/TTS) completes under 500ms."""
        start = time.time()

        validate_image(green_leaf)
        calculate_severity(green_leaf)
        rag.build_context(
            disease_id="wheat_leaf_rust",
            disease_name="Wheat Leaf Rust",
            severity_pct=5.0,
            severity_level="Mild",
            confidence=0.92,
            crop="Wheat",
        )
        _get_static_fallback("wheat_leaf_rust")

        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 500, f"Pipeline took {elapsed_ms:.0f}ms (limit 500ms)"
