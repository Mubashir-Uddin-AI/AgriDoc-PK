# -*- coding: utf-8 -*-
"""Unit tests for AgriDoc-PK Phase 3 modules.

Covers:
  - Classifier: model architecture, preprocessing, class mappings
  - RAG Engine: ChromaDB retrieval, context assembly
  - LLM Orchestrator: static fallback, prompt construction
  - TTS Engine: synthesis integration (TC-09)
  - Build Index: chunking logic

Run with: python -m pytest tests/test_phase3.py -v
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_base.build_index import _chunk_text, _extract_sections


# ========================================================================
# Index Builder — Chunking & Section Extraction
# ========================================================================


class TestIndexBuilder:
    """Tests for the PARC document chunking and section extraction."""

    def test_chunk_short_text(self):
        """Text shorter than chunk_size stays as one chunk."""
        text = "This is a short paragraph about wheat rust."
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_long_text_creates_overlaps(self):
        """Long text is split into multiple overlapping chunks."""
        text = "Sentence one about fungicide. " * 30  # ~900 chars
        chunks = _chunk_text(text, chunk_size=200, overlap=50)
        assert len(chunks) > 1
        # Verify overlap: end of chunk N should appear in start of chunk N+1
        for i in range(len(chunks) - 1):
            last_words = chunks[i][-30:]
            assert any(w in chunks[i + 1][:80] for w in last_words.split()[:3])

    def test_extract_sections_from_markdown(self):
        """Markdown headings are correctly extracted as sections."""
        md_text = """# Main Title
Some intro text.

## Section One
Content of section one.

### Subsection
More detailed content.

## Section Two
Final content here.
"""
        sections = _extract_sections(md_text)
        assert len(sections) >= 3
        headings = [s["heading"] for s in sections]
        assert "Section One" in headings
        assert "Section Two" in headings

    def test_extract_sections_preserves_content(self):
        """Section content is preserved without the heading line."""
        md_text = """## Treatment
Apply Nativo 75 WG at 65 grams per acre.
"""
        sections = _extract_sections(md_text)
        treatment = [s for s in sections if s["heading"] == "Treatment"]
        assert len(treatment) == 1
        assert "Nativo 75 WG" in treatment[0]["content"]
        assert "65 grams" in treatment[0]["content"]


# ========================================================================
# Classifier — Architecture & Class Mappings
# ========================================================================


class TestClassifier:
    """Tests for the MobileNetV3 classifier module."""

    def test_class_names_count(self):
        """Verify all 10 classes are defined per PRD FR-2.1."""
        from core.classifier import CLASS_NAMES
        assert len(CLASS_NAMES) == 10

    def test_all_classes_have_urdu_labels(self):
        """Every class must have a corresponding Urdu label."""
        from core.classifier import CLASS_NAMES, CLASS_LABELS_UR
        for cls in CLASS_NAMES:
            assert cls in CLASS_LABELS_UR, f"Missing Urdu label for {cls}"
            assert len(CLASS_LABELS_UR[cls]) > 0

    def test_all_classes_have_crop_mapping(self):
        """Every class maps to a (crop_en, crop_ur) tuple."""
        from core.classifier import CLASS_NAMES, CROP_FROM_CLASS
        for cls in CLASS_NAMES:
            assert cls in CROP_FROM_CLASS
            crop_en, crop_ur = CROP_FROM_CLASS[cls]
            assert crop_en in ("Wheat", "Cotton", "Rice")

    def test_model_initialization(self):
        """MobileNetV3-Small initializes with correct output size."""
        from core.classifier import initialize_model
        model = initialize_model(num_classes=10, pretrained=False)
        # Verify final classifier outputs 10 classes
        final_layer = model.classifier[-1]
        assert final_layer.out_features == 10

    def test_preprocessing_output_shape(self):
        """Preprocessing produces correct tensor shape (1, 3, 224, 224)."""
        from core.classifier import CropDiseaseClassifier
        # Create classifier without weights (will warn but work)
        classifier = CropDiseaseClassifier(weights_path="nonexistent.pth")
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensor = classifier.preprocess(img)
        assert tensor.shape == (1, 3, 224, 224)

    def test_predict_returns_required_keys(self):
        """Predict returns all required diagnosis keys."""
        from core.classifier import CropDiseaseClassifier
        classifier = CropDiseaseClassifier(weights_path="nonexistent.pth")
        img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        result = classifier.predict(img)
        required_keys = [
            "disease_id", "disease_name", "disease_name_ur",
            "crop", "crop_ur", "confidence", "pathogen",
            "is_ood", "ood_message_ur",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"
        assert 0.0 <= result["confidence"] <= 1.0


# ========================================================================
# RAG Engine — ChromaDB Retrieval
# ========================================================================


class TestRAGEngine:
    """Tests for the ChromaDB-backed RAG retrieval engine."""

    @pytest.fixture(scope="class")
    def rag_engine(self):
        """Initialize RAG engine (uses the index built by build_index.py)."""
        from core.rag_engine import RAGEngine
        engine = RAGEngine()
        return engine

    def test_rag_engine_is_ready(self, rag_engine):
        """RAG engine should have a populated index from build_index.py."""
        assert rag_engine.is_ready, (
            "RAG engine not ready — run 'python knowledge_base/build_index.py' first"
        )

    def test_retrieve_wheat_rust(self, rag_engine):
        """Query for wheat rust returns relevant PARC passages."""
        if not rag_engine.is_ready:
            pytest.skip("RAG index not built")
        results = rag_engine.retrieve(
            "wheat leaf rust treatment fungicide dosage",
            disease_id="wheat_leaf_rust",
            top_k=3,
        )
        assert len(results) > 0
        # Check at least one passage mentions relevant content
        all_content = " ".join(r["content"].lower() for r in results)
        assert any(term in all_content for term in ["rust", "nativo", "fungicid", "wheat"])

    def test_retrieve_cotton_clcuv(self, rag_engine):
        """Query for CLCuV returns relevant PARC passages."""
        if not rag_engine.is_ready:
            pytest.skip("RAG index not built")
        results = rag_engine.retrieve(
            "cotton leaf curl virus whitefly control",
            disease_id="cotton_leaf_curl_virus",
        )
        assert len(results) > 0
        all_content = " ".join(r["content"].lower() for r in results)
        assert any(term in all_content for term in ["cotton", "whitefly", "virus", "curl"])

    def test_retrieve_rice_blast(self, rag_engine):
        """Query for rice blast returns relevant PARC passages."""
        if not rag_engine.is_ready:
            pytest.skip("RAG index not built")
        results = rag_engine.retrieve(
            "rice blast fungicide tricyclazole treatment",
            disease_id="rice_blast",
        )
        assert len(results) > 0

    def test_context_assembly(self, rag_engine):
        """build_context produces complete context dictionary."""
        if not rag_engine.is_ready:
            pytest.skip("RAG index not built")
        context = rag_engine.build_context(
            disease_id="wheat_leaf_rust",
            disease_name="Wheat Leaf Rust",
            severity_pct=24.5,
            severity_level="Moderate",
            confidence=0.95,
            crop="Wheat",
        )
        assert "vision_metadata" in context
        assert "retrieved_passages" in context
        assert "citations" in context
        assert context["vision_metadata"]["disease_id"] == "wheat_leaf_rust"
        assert len(context["retrieved_passages"]) > 0

    def test_retrieval_has_relevance_scores(self, rag_engine):
        """Retrieved passages include relevance scores between 0 and 1."""
        if not rag_engine.is_ready:
            pytest.skip("RAG index not built")
        results = rag_engine.retrieve("wheat rust nativo dosage")
        for r in results:
            assert "relevance_score" in r
            assert 0.0 <= r["relevance_score"] <= 1.0


# ========================================================================
# LLM Orchestrator — Static Fallback & Prompt Construction
# ========================================================================


class TestLLMOrchestrator:
    """Tests for the Gemini Flash LLM orchestrator."""

    def test_static_fallback_wheat_rust(self):
        """Static fallback generates valid advisory for wheat_leaf_rust."""
        from core.llm_orchestrator import _get_static_fallback
        fallback = _get_static_fallback("wheat_leaf_rust")
        assert fallback["is_fallback"] is True
        assert "chemical_treatment" in fallback
        chem = fallback["chemical_treatment"]
        assert chem["product_name"] == "Nativo 75 WG"
        assert chem["guardrail_verified"] is True
        assert "voice_script_ur" in fallback
        assert len(fallback["voice_script_ur"]) > 0

    def test_static_fallback_cotton_clcuv(self):
        """Static fallback for CLCuV recommends whitefly insecticide."""
        from core.llm_orchestrator import _get_static_fallback
        fallback = _get_static_fallback("cotton_leaf_curl_virus")
        chem = fallback["chemical_treatment"]
        assert "Imidacloprid" in chem["active_ingredient"]

    def test_static_fallback_rice_blast(self):
        """Static fallback for rice blast recommends Tricyclazole."""
        from core.llm_orchestrator import _get_static_fallback
        fallback = _get_static_fallback("rice_blast")
        chem = fallback["chemical_treatment"]
        assert "Tricyclazole" in chem["active_ingredient"]

    def test_static_fallback_unknown_disease(self):
        """Unknown disease returns graceful fallback."""
        from core.llm_orchestrator import _get_static_fallback
        fallback = _get_static_fallback("unknown_disease_xyz")
        assert fallback["is_fallback"] is True
        assert "sources_cited" in fallback

    def test_generate_advisory_no_api_key(self):
        """Without API key, generate_advisory uses static fallback."""
        from core.llm_orchestrator import generate_advisory
        # Ensure no API key is set
        old_key = os.environ.pop("GOOGLE_API_KEY", None)
        try:
            context = {
                "vision_metadata": {
                    "crop": "Wheat",
                    "disease_id": "wheat_leaf_rust",
                    "disease_name": "Wheat Leaf Rust",
                    "confidence": 0.95,
                    "severity_pct": 25.0,
                    "severity_level": "Moderate",
                },
                "retrieved_passages": [],
                "citations": [],
            }
            result = generate_advisory(context, "wheat_leaf_rust", api_key="")
            assert result["is_fallback"] is True
            assert result["chemical_treatment"]["guardrail_verified"] is True
        finally:
            if old_key:
                os.environ["GOOGLE_API_KEY"] = old_key

    def test_prompt_construction(self):
        """_build_user_prompt produces formatted prompt with vision context."""
        from core.llm_orchestrator import _build_user_prompt
        context = {
            "vision_metadata": {
                "crop": "Wheat",
                "disease_id": "wheat_leaf_rust",
                "disease_name": "Wheat Leaf Rust",
                "confidence": 0.95,
                "severity_pct": 24.5,
                "severity_level": "Moderate",
            },
            "retrieved_passages": [
                {
                    "content": "Apply Nativo 75 WG at 65 grams per acre.",
                    "source_file": "wheat_rust_bulletin.md",
                    "section_heading": "Treatment",
                }
            ],
        }
        prompt = _build_user_prompt(context)
        assert "Wheat Leaf Rust" in prompt
        assert "24.5%" in prompt
        assert "Nativo 75 WG" in prompt
        assert "PARC Passage 1" in prompt


# ========================================================================
# TTS Engine — Voice Synthesis (TC-09)
# ========================================================================


class TestTTSEngine:
    """TC-09: Voice Synthesizer — Urdu text → playable MP3 bytes."""

    @pytest.mark.skipif(
        os.environ.get("SKIP_TTS_TESTS", "0") == "1",
        reason="TTS tests skipped (requires network)",
    )
    def test_urdu_synthesis_produces_mp3(self):
        """Urdu text synthesizes to non-empty MP3 bytes."""
        from core.tts_engine import synthesize_urdu
        text = "آپ کی گندم کی فصل میں زنگ پایا گیا ہے۔"
        try:
            audio = synthesize_urdu(text, use_cache=False)
        except RuntimeError as e:
            if "403" in str(e) or "handshake" in str(e).lower():
                pytest.skip("Edge-TTS service temporarily unavailable (403)")
            raise
        assert isinstance(audio, bytes)
        assert len(audio) > 1000, f"Audio too small: {len(audio)} bytes"
        # MP3 starts with ID3 tag or MPEG sync bytes
        assert audio[:3] in (b"ID3", b"\xff\xfb", b"\xff\xf3")

    @pytest.mark.skipif(
        os.environ.get("SKIP_TTS_TESTS", "0") == "1",
        reason="TTS tests skipped (requires network)",
    )
    def test_tts_caching(self):
        """Repeated synthesis of same text returns cached result."""
        from core.tts_engine import synthesize_urdu, clear_cache
        clear_cache()
        text = "یہ ایک مختصر پیغام ہے۔"
        try:
            audio1 = synthesize_urdu(text, use_cache=True)
        except RuntimeError as e:
            if "403" in str(e) or "handshake" in str(e).lower():
                pytest.skip("Edge-TTS service temporarily unavailable (403)")
            raise
        audio2 = synthesize_urdu(text, use_cache=True)
        # Cached bytes should be identical
        assert audio1 == audio2

    def test_empty_text_returns_empty(self):
        """Empty text input returns empty bytes without error."""
        from core.tts_engine import synthesize_urdu
        result = synthesize_urdu("", use_cache=False)
        assert result == b""

    def test_cache_clear(self):
        """Cache clear returns count and empties the cache."""
        from core.tts_engine import clear_cache, _audio_cache
        clear_cache()
        assert len(_audio_cache) == 0
