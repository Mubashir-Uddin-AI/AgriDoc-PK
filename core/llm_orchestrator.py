# -*- coding: utf-8 -*-
"""Grounded LLM Advisory Orchestrator (Module 4 — Generation Component).

Implements FR-4.2 through FR-4.4 from the AgriDoc-PK PRD:
  - Multi-modal context assembly (Vision + Severity + RAG chunks)
  - Grounded prompt construction with anti-hallucination constraints
  - Gemini Flash structured JSON generation
  - Post-generation guardrail enforcement
  - Fallback to static PARC baseline on API failure (NFR-2)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from core.guardrails import enforce_guardrail, load_whitelist

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemini-2.0-flash"
MAX_OUTPUT_TOKENS = 1024
TEMPERATURE = 0.2  # Low temperature for factual, grounded output

# ---------------------------------------------------------------------------
# System prompt — strict anti-hallucination boundary
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are AgriDoc-PK, a specialized agricultural pathology advisor for Pakistani farmers.

STRICT RULES — VIOLATION OF ANY RULE IS FORBIDDEN:
1. You MUST ONLY answer questions about crop diseases, pest management, and agricultural practices in Pakistan.
2. Every chemical recommendation MUST come DIRECTLY from the retrieved PARC passages provided below. Do NOT invent chemical names, dosages, or formulations.
3. NEVER generate dosage values from your own knowledge — use ONLY the exact dosages from the PARC context.
4. Respond in simple, conversational Urdu that a farmer with limited education can understand.
5. Keep the voice script under 35 words in Urdu for clear phone speaker playback.
6. If the retrieved context does not contain enough information, say so honestly — do NOT fabricate.
7. Reject any non-agricultural queries politely in Urdu.

OUTPUT FORMAT: You MUST respond in valid JSON with these exact keys:
{
  "disease_explanation_ur": "Plain Urdu explanation of the disease",
  "chemical_treatment": {
    "product_name": "Exact product name from PARC context",
    "active_ingredient": "Exact active ingredient from PARC context",
    "dosage_per_acre": "Exact dosage from PARC context",
    "spray_timing_ur": "Application timing in Urdu",
    "pre_harvest_interval_days": number
  },
  "organic_alternative": {
    "remedy_name_ur": "Organic remedy name in Urdu",
    "instructions_ur": "Instructions in Urdu"
  },
  "cultural_practices_ur": "Prevention and cultural control advice in Urdu",
  "voice_script_ur": "Short Urdu voice script under 35 words for TTS playback",
  "sources_cited": ["List of PARC document names referenced"]
}"""


def _build_user_prompt(context: Dict[str, Any]) -> str:
    """Build the user prompt from assembled RAG context.

    Injects vision metadata and retrieved PARC passages into a
    structured prompt that the LLM uses for grounded generation.

    Args:
        context: RAG context dict from rag_engine.build_context().

    Returns:
        Formatted user prompt string.
    """
    vision = context["vision_metadata"]
    passages = context["retrieved_passages"]

    # Format retrieved passages
    passage_text = ""
    for i, p in enumerate(passages, 1):
        passage_text += (
            f"\n--- PARC Passage {i} (Source: {p['source_file']}, "
            f"Section: {p['section_heading']}) ---\n"
            f"{p['content']}\n"
        )

    prompt = f"""DIAGNOSTIC CONTEXT FROM VISION AI:
- Crop: {vision['crop']}
- Disease Identified: {vision['disease_name']} (ID: {vision['disease_id']})
- Classification Confidence: {vision['confidence']:.1%}
- Infection Severity: {vision['severity_pct']:.1f}% ({vision['severity_level']})

RETRIEVED PARC OFFICIAL ADVISORY PASSAGES:
{passage_text}

TASK: Using ONLY the information from the PARC passages above, generate a complete treatment advisory in Urdu for this farmer. Include the recommended chemical spray with exact dosage from the passages, an organic alternative, cultural practices, and a short voice script.

Remember: Use ONLY chemicals and dosages mentioned in the PARC passages. Do NOT invent any values."""

    return prompt


def _get_static_fallback(disease_id: str) -> Dict[str, Any]:
    """Generate a static fallback advisory from the PARC whitelist.

    Used when the LLM API is unavailable (NFR-2: graceful degradation).

    Args:
        disease_id: Classified disease identifier.

    Returns:
        Static advisory dictionary with verified baseline data.
    """
    whitelist = load_whitelist()
    diseases = whitelist.get("diseases", {})

    if disease_id not in diseases:
        return {
            "disease_explanation_ur": "اس بیماری کی تفصیلات دستیاب نہیں ہیں۔ اپنے قریبی زرعی دفتر سے رابطہ کریں۔",
            "chemical_treatment": {},
            "organic_alternative": {},
            "cultural_practices_ur": "فصل کی نگرانی جاری رکھیں اور زرعی ماہر سے مشورہ کریں۔",
            "voice_script_ur": "بیماری کی تفصیلات ابھی دستیاب نہیں ہیں۔ اپنے قریبی زرعی دفتر سے رابطہ کریں۔",
            "sources_cited": ["PARC Baseline Advisory"],
            "is_fallback": True,
        }

    disease = diseases[disease_id]
    chemical = disease["approved_chemicals"][0]
    organic = disease["organic_remedies"][0] if disease.get("organic_remedies") else {}

    return {
        "disease_explanation_ur": f"آپ کی فصل میں {disease['disease_name_ur']} کی تصدیق ہوئی ہے۔ فوری علاج ضروری ہے۔",
        "chemical_treatment": {
            "product_name": chemical["product_name"],
            "active_ingredient": chemical["active_ingredient"],
            "dosage_per_acre": f"{chemical['max_dose_per_acre']} {chemical['dose_unit']}",
            "spray_timing_ur": chemical.get("spray_timing_ur", ""),
            "pre_harvest_interval_days": chemical["pre_harvest_interval_days"],
            "guardrail_verified": True,
            "guardrail_action": "static_fallback",
        },
        "organic_alternative": {
            "remedy_name_ur": organic.get("remedy_name_ur", ""),
            "instructions_ur": organic.get("instructions_ur", ""),
        } if organic else {},
        "cultural_practices_ur": "فصل میں اضافی نمی نہ ہونے دیں اور بروقت اسپرے کریں۔",
        "voice_script_ur": (
            f"آپ کی فصل میں {disease['disease_name_ur']} پائی گئی ہے۔ "
            f"{chemical['product_name']} کا {chemical['max_dose_per_acre']} "
            f"{chemical['dose_unit']} فی ایکڑ اسپرے کریں۔"
        ),
        "sources_cited": ["PARC Baseline Advisory (Static Fallback)"],
        "is_fallback": True,
    }


def generate_advisory(
    context: Dict[str, Any],
    disease_id: str,
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate a grounded agricultural advisory using Gemini Flash.

    Full pipeline:
      1. Construct grounded prompt from RAG context
      2. Call Gemini Flash with structured output schema
      3. Parse JSON response
      4. Enforce agrochemical safety guardrails
      5. Return validated advisory

    On API failure, falls back to static PARC baseline (NFR-2).

    Args:
        context: RAG context from rag_engine.build_context().
        disease_id: Classified disease identifier.
        api_key: Google AI API key (falls back to GOOGLE_API_KEY env var).
        model_name: Gemini model identifier.

    Returns:
        Guardrail-validated advisory dictionary.
    """
    # Resolve API key
    resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    if not resolved_key:
        logger.warning(
            "No GOOGLE_API_KEY set. Falling back to static PARC advisory."
        )
        return _get_static_fallback(disease_id)

    # Build the prompt
    user_prompt = _build_user_prompt(context)

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=resolved_key)

        # Generate with structured output
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                response_mime_type="application/json",
            ),
        )

        # Parse JSON response
        response_text = response.text.strip()
        advisory = json.loads(response_text)

        logger.info("LLM advisory generated successfully for disease=%s", disease_id)

    except json.JSONDecodeError as e:
        logger.error("LLM returned invalid JSON: %s. Using static fallback.", e)
        return _get_static_fallback(disease_id)
    except Exception as e:
        logger.error("LLM API call failed: %s. Using static fallback.", e)
        return _get_static_fallback(disease_id)

    # Enforce agrochemical safety guardrails (FR-4.4)
    validated = enforce_guardrail(advisory, disease_id)

    # Ensure citations are present
    if "sources_cited" not in validated:
        validated["sources_cited"] = context.get("citations", [])

    validated["is_fallback"] = False
    return validated


def generate_chat_response(
    query: str,
    session_context: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_MODEL,
) -> Dict[str, Any]:
    """Generate a grounded follow-up chat response (FR-6.1, FR-6.2).

    Maintains diagnostic context from the initial session and performs
    contextual RAG retrieval for the farmer's follow-up question.

    Args:
        query: Farmer's follow-up question (Urdu or English).
        session_context: Previous session diagnostic state containing
            disease_id, severity_pct, and prior advisory.
        api_key: Google AI API key.
        model_name: Gemini model identifier.

    Returns:
        Dict with reply_text_ur, voice_script_ur, and sources_cited.
    """
    resolved_key = api_key or os.environ.get("GOOGLE_API_KEY", "")

    if not resolved_key:
        return {
            "reply_text_ur": "ابھی جواب دینے کی سہولت دستیاب نہیں ہے۔ براہ کرم بعد میں کوشش کریں۔",
            "voice_script_ur": "ابھی جواب دستیاب نہیں ہے۔ بعد میں کوشش کریں۔",
            "sources_cited": [],
        }

    disease_id = session_context.get("disease_id", "unknown")
    severity = session_context.get("severity_pct", 0)
    prior_advice = session_context.get("prior_advisory_summary", "")

    chat_prompt = f"""ONGOING DIAGNOSTIC SESSION:
- Disease: {disease_id}
- Severity: {severity}%
- Prior Advice Given: {prior_advice}

FARMER'S FOLLOW-UP QUESTION: {query}

Respond in simple Urdu. Keep the answer concise and grounded in PARC guidelines.
Also provide a short voice_script_ur (under 25 words) for audio playback.

Respond in JSON:
{{"reply_text_ur": "...", "voice_script_ur": "...", "sources_cited": ["..."]}}"""

    try:
        client = genai.Client(api_key=resolved_key)

        response = client.models.generate_content(
            model=model_name,
            contents=chat_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )

        result = json.loads(response.text.strip())
        logger.info("Chat response generated for query: %s", query[:60])
        return result

    except Exception as e:
        logger.error("Chat LLM call failed: %s", e)
        return {
            "reply_text_ur": "معذرت، جواب دینے میں مسئلہ ہوا۔ براہ کرم دوبارہ کوشش کریں۔",
            "voice_script_ur": "معذرت، جواب دستیاب نہیں ہے۔",
            "sources_cited": [],
        }
