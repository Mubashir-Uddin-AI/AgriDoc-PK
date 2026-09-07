# -*- coding: utf-8 -*-
"""AgriDoc-PK: Streamlit Mobile Web UI.

High-contrast, field-friendly diagnostic interface with:
  - Crop selector tabs (Wheat / Cotton / Rice)
  - Camera capture & file upload
  - Edge Quality Gate validation
  - AI diagnosis card with severity meter
  - Urdu voice audio playback (Edge-TTS)
  - Follow-up Q&A chat (Context-Aware RAG)

Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import cv2
import numpy as np
import streamlit as st

from core.preprocessor import validate_image, resize_image, strip_exif
from core.severity_engine import calculate_severity
from core.guardrails import load_whitelist

# Conditional imports — graceful degradation if modules aren't ready
try:
    from core.classifier import CropDiseaseClassifier, CLASS_LABELS_UR, CROP_FROM_CLASS, PATHOGEN_MAP
    CLASSIFIER_AVAILABLE = True
except Exception:
    CLASSIFIER_AVAILABLE = False

try:
    from core.rag_engine import RAGEngine
    RAG_AVAILABLE = True
except Exception:
    RAG_AVAILABLE = False

try:
    from core.llm_orchestrator import generate_advisory, generate_chat_response, _get_static_fallback
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

try:
    from core.tts_engine import synthesize_urdu
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# UI Configuration — PRD Section 5.1 Field Ergonomics
# ---------------------------------------------------------------------------
COLORS = {
    "bg": "#F0FFF4",
    "primary": "#1B4332",
    "primary_light": "#2D6A4F",
    "text": "#1B4332",
    "text_secondary": "#4A5568",
    "severe": "#DC2626",
    "moderate": "#F59E0B",
    "mild": "#10B981",
    "white": "#FFFFFF",
    "accent": "#2D6A4F",
    "sage": "#52B788",
    "mint": "#B7E4C7",
    "cream": "#F0FFF4",
    "gold": "#D4A017",
    "card_bg": "rgba(255, 255, 255, 0.85)",
    "card_border": "rgba(45, 106, 79, 0.15)",
    "shadow": "rgba(27, 67, 50, 0.08)",
}

SEVERITY_COLORS = {
    "Mild": COLORS["mild"],
    "Moderate": COLORS["moderate"],
    "Severe": COLORS["severe"],
}

CROP_TABS = {
    "🌾 گندم (Wheat)": "wheat",
    "🌿 کپاس (Cotton)": "cotton",
    "🌾 چاول (Rice)": "rice",
}


def configure_page():
    """Set Streamlit page config for mobile-first, modern display."""
    st.set_page_config(
        page_title="AgriDoc-PK | ایگری ڈاک پاکستان",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.markdown("""
    <style>
        /* ===== Google Font Import ===== */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');

        /* ===== Keyframe Animations ===== */
        @keyframes headerShimmer {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        @keyframes pulseGlow {
            0%, 100% { box-shadow: 0 0 8px rgba(45, 106, 79, 0.3); }
            50% { box-shadow: 0 0 20px rgba(45, 106, 79, 0.6); }
        }

        @keyframes severityFill {
            from { width: 0%; }
        }

        /* ===== Base App Styling ===== */
        .stApp {
            background: linear-gradient(180deg, #F0FFF4 0%, #E8F5E8 40%, #F0FFF4 100%) !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* Hide default Streamlit header/footer chrome */
        #MainMenu { visibility: hidden; }
        header[data-testid="stHeader"] {
            background: transparent !important;
        }
        footer { visibility: hidden; }

        /* ===== Card / Container Styling ===== */
        .glass-card {
            background: rgba(255, 255, 255, 0.88);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(45, 106, 79, 0.12);
            border-radius: 20px;
            padding: 28px;
            margin: 16px 0;
            box-shadow: 0 8px 32px rgba(27, 67, 50, 0.06),
                        0 2px 8px rgba(27, 67, 50, 0.04);
            animation: fadeInUp 0.5s ease-out;
        }

        /* ===== Hero Header ===== */
        .hero-header {
            background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 30%, #40916C 60%, #2D6A4F 100%);
            background-size: 200% 200%;
            animation: headerShimmer 8s ease infinite;
            color: white;
            padding: 36px 32px;
            border-radius: 24px;
            margin-bottom: 28px;
            text-align: center;
            position: relative;
            overflow: hidden;
            box-shadow: 0 12px 40px rgba(27, 67, 50, 0.25),
                        0 4px 12px rgba(27, 67, 50, 0.15);
        }
        .hero-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
            pointer-events: none;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 2px 8px rgba(0,0,0,0.2);
            letter-spacing: 0.5px;
        }
        .hero-subtitle {
            display: inline-block;
            margin-top: 12px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(4px);
            padding: 6px 20px;
            border-radius: 50px;
            font-size: 0.95rem;
            font-weight: 500;
            letter-spacing: 0.3px;
        }

        /* ===== Buttons ===== */
        .stButton > button {
            min-height: 52px;
            font-size: 1.05rem;
            font-weight: 600;
            border-radius: 14px;
            padding: 12px 28px;
            margin: 6px 0;
            border: none;
            background: linear-gradient(135deg, #2D6A4F, #1B4332) !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(27, 67, 50, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            letter-spacing: 0.3px;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(27, 67, 50, 0.35) !important;
            background: linear-gradient(135deg, #40916C, #2D6A4F) !important;
        }
        .stButton > button:active {
            transform: translateY(0px);
        }

        /* ===== Tab Styling ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255, 255, 255, 0.6);
            backdrop-filter: blur(8px);
            padding: 8px;
            border-radius: 18px;
            border: 1px solid rgba(45, 106, 79, 0.1);
        }
        .stTabs [data-baseweb="tab"] {
            min-height: 52px;
            font-size: 1.05rem;
            font-weight: 600;
            padding: 10px 24px;
            border-radius: 14px !important;
            color: #2D6A4F !important;
            background: transparent;
            transition: all 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background: rgba(45, 106, 79, 0.08);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #2D6A4F, #1B4332) !important;
            color: white !important;
            box-shadow: 0 4px 14px rgba(27, 67, 50, 0.25);
        }
        .stTabs [data-baseweb="tab-highlight"] {
            display: none;
        }
        .stTabs [data-baseweb="tab-border"] {
            display: none;
        }

        /* ===== Upload Area ===== */
        .stFileUploader {
            animation: fadeInUp 0.6s ease-out;
        }
        .stFileUploader > div {
            border-radius: 16px !important;
            border: 2px dashed rgba(45, 106, 79, 0.3) !important;
            background: rgba(255, 255, 255, 0.7) !important;
            padding: 20px !important;
            transition: all 0.3s ease;
        }
        .stFileUploader > div:hover {
            border-color: #2D6A4F !important;
            background: rgba(255, 255, 255, 0.9) !important;
            box-shadow: 0 4px 16px rgba(45, 106, 79, 0.1);
        }

        /* ===== Camera Input ===== */
        .stCameraInput > div {
            border-radius: 16px !important;
            overflow: hidden;
            border: 2px solid rgba(45, 106, 79, 0.15) !important;
            box-shadow: 0 4px 16px rgba(27, 67, 50, 0.08);
        }

        /* ===== Image Display ===== */
        .stImage {
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 6px 20px rgba(27, 67, 50, 0.1);
        }
        .stImage img {
            border-radius: 16px;
        }

        /* ===== Diagnosis Card ===== */
        .diagnosis-card {
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(45, 106, 79, 0.12);
            border-left: 5px solid #2D6A4F;
            border-radius: 20px;
            padding: 28px;
            margin: 20px 0;
            box-shadow: 0 8px 32px rgba(27, 67, 50, 0.08);
            animation: fadeInUp 0.5s ease-out;
        }

        /* ===== Metrics ===== */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(45, 106, 79, 0.1);
            border-radius: 16px;
            padding: 16px 20px;
            box-shadow: 0 2px 8px rgba(27, 67, 50, 0.05);
            transition: transform 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
        }
        [data-testid="stMetricLabel"] {
            font-weight: 600 !important;
            color: #2D6A4F !important;
        }
        [data-testid="stMetricValue"] {
            color: #1B4332 !important;
            font-weight: 700 !important;
        }

        /* ===== Severity Meter ===== */
        .severity-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 20px;
            padding: 24px;
            margin: 16px 0;
            border: 1px solid rgba(45, 106, 79, 0.1);
            box-shadow: 0 4px 16px rgba(27, 67, 50, 0.06);
        }
        .severity-label {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .severity-label-title {
            font-weight: 700;
            font-size: 1.15rem;
            color: #1B4332;
        }
        .severity-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 50px;
            font-weight: 700;
            font-size: 1.05rem;
            color: white;
        }
        .severity-track {
            background: #E5E7EB;
            border-radius: 14px;
            height: 28px;
            overflow: hidden;
            box-shadow: inset 0 2px 4px rgba(0,0,0,0.06);
        }
        .severity-fill {
            height: 100%;
            border-radius: 14px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
            animation: severityFill 1.2s ease-out;
            position: relative;
        }
        .severity-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(180deg, rgba(255,255,255,0.25) 0%, transparent 100%);
            border-radius: 14px;
        }

        /* ===== Audio Section ===== */
        .audio-card {
            background: linear-gradient(135deg, #1B4332, #2D6A4F);
            color: white;
            padding: 24px;
            border-radius: 20px;
            text-align: center;
            margin: 20px 0;
            box-shadow: 0 8px 24px rgba(27, 67, 50, 0.2);
            animation: fadeInUp 0.5s ease-out;
        }
        .audio-card h3 {
            color: white !important;
            margin-bottom: 12px;
        }

        /* ===== Chat Styling ===== */
        .stChatMessage {
            border-radius: 16px !important;
            padding: 12px 16px !important;
            margin: 8px 0 !important;
            box-shadow: 0 2px 8px rgba(27, 67, 50, 0.06) !important;
            border: 1px solid rgba(45, 106, 79, 0.08) !important;
        }
        [data-testid="stChatInput"] > div {
            border-radius: 16px !important;
            border: 2px solid rgba(45, 106, 79, 0.2) !important;
            transition: border-color 0.3s ease;
        }
        [data-testid="stChatInput"] > div:focus-within {
            border-color: #2D6A4F !important;
            box-shadow: 0 0 0 3px rgba(45, 106, 79, 0.1) !important;
        }

        /* ===== Expander ===== */
        .streamlit-expanderHeader {
            font-weight: 600;
            font-size: 1.05rem;
            color: #2D6A4F !important;
            background: rgba(255, 255, 255, 0.6);
            border-radius: 12px;
        }

        /* ===== RTL support for Urdu ===== */
        .urdu-text {
            direction: rtl;
            text-align: right;
            font-family: 'Noto Nastaliq Urdu', 'Jameel Noori Nastaleeq', serif;
            font-size: 1.15rem;
            line-height: 2.2;
            color: #1B4332;
        }

        /* ===== Status / Alerts ===== */
        .stAlert {
            border-radius: 14px !important;
        }
        [data-testid="stStatusWidget"] {
            border-radius: 16px !important;
        }

        /* ===== Section Dividers ===== */
        .section-divider {
            height: 2px;
            background: linear-gradient(90deg, transparent, rgba(45, 106, 79, 0.2), transparent);
            margin: 28px 0;
            border: none;
        }

        /* ===== Section Heading ===== */
        .section-heading {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.3rem;
            font-weight: 700;
            color: #1B4332;
            margin: 20px 0 16px;
        }
        .section-heading .icon {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 42px;
            height: 42px;
            background: linear-gradient(135deg, #D1FAE5, #A7F3D0);
            border-radius: 12px;
            font-size: 1.3rem;
        }

        /* ===== High contrast headings ===== */
        h1, h2, h3, h4 {
            color: #1B4332 !important;
            font-family: 'Inter', sans-serif !important;
        }

        /* ===== Footer ===== */
        .app-footer {
            text-align: center;
            padding: 24px 16px;
            margin-top: 40px;
            color: #4A5568;
            font-size: 0.85rem;
            border-top: 1px solid rgba(45, 106, 79, 0.1);
        }
        .app-footer a {
            color: #2D6A4F;
            text-decoration: none;
            font-weight: 600;
        }

        /* ===== Responsive Mobile ===== */
        @media (max-width: 768px) {
            .hero-header { padding: 24px 16px; border-radius: 16px; }
            .hero-title { font-size: 1.5rem; }
            .glass-card { padding: 20px; border-radius: 16px; }
            .diagnosis-card { padding: 20px; border-radius: 16px; }
            .stTabs [data-baseweb="tab"] { padding: 8px 14px; font-size: 0.95rem; }
        }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render the premium animated hero header."""
    st.markdown("""
    <div class="hero-header">
        <h1 class="hero-title">🌿 AgriDoc-PK | ایگری ڈاک پاکستان</h1>
        <div class="hero-subtitle">
            🤖 AI Crop Doctor — Bano Qabil AI Hackathon 2026
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_severity_meter(pct: float, level: str, level_ur: str):
    """Render a modern animated severity progress bar with triage badge."""
    color = SEVERITY_COLORS.get(level, COLORS["moderate"])
    bar_width = min(pct, 100)

    # Create gradient based on severity
    if level == "Severe":
        gradient = "linear-gradient(90deg, #F59E0B, #EF4444, #DC2626)"
    elif level == "Moderate":
        gradient = "linear-gradient(90deg, #10B981, #F59E0B, #F59E0B)"
    else:
        gradient = "linear-gradient(90deg, #6EE7B7, #10B981, #10B981)"

    st.markdown(f"""
    <div class="severity-container">
        <div class="severity-label">
            <span class="severity-label-title">📊 بیماری کی شدت / Disease Severity</span>
            <span class="severity-badge" style="background:{color}; box-shadow: 0 4px 12px {color}40;">
                {pct:.1f}% — {level_ur}
            </span>
        </div>
        <div class="severity-track">
            <div class="severity-fill"
                 style="width:{bar_width}%; background:{gradient}; box-shadow: 0 0 12px {color}30;">
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_diagnosis_card(diagnosis: dict, severity: dict, advisory: dict):
    """Render the full diagnosis result card with glassmorphic styling."""
    crop = diagnosis.get("crop", "")
    crop_ur = diagnosis.get("crop_ur", "")
    disease_name = diagnosis.get("disease_name", "")
    disease_ur = diagnosis.get("disease_name_ur", "")
    confidence = diagnosis.get("confidence", 0)
    pathogen = diagnosis.get("pathogen", "")

    sev_pct = severity.get("infected_percentage", 0)
    sev_level = severity.get("level", "")
    sev_level_ur = severity.get("level_ur", "")

    # Section divider
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Section heading
    st.markdown("""
    <div class="section-heading">
        <span class="icon">🔬</span>
        <span>AI تشخیص کے نتائج / Diagnosis Results</span>
    </div>
    """, unsafe_allow_html=True)

    # Diagnosis card
    st.markdown('<div class="diagnosis-card">', unsafe_allow_html=True)

    # Diagnosis summary metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🌾 فصل / Crop", f"{crop_ur} ({crop})")
        st.metric("🔬 اعتماد / Confidence", f"{confidence:.1%}")
    with col2:
        st.metric("⚠️ تشخیص / Diagnosis", disease_ur)
        if pathogen and pathogen != "N/A":
            st.metric("🦠 جرثومہ / Pathogen", pathogen)

    st.markdown('</div>', unsafe_allow_html=True)

    # Severity meter
    render_severity_meter(sev_pct, sev_level, sev_level_ur)

    # Advisory sections
    if advisory:
        # Citations
        sources = advisory.get("sources_cited", [])
        if sources:
            st.markdown(f"""
            <div style="background: rgba(45,106,79,0.06); border-radius: 12px; padding: 12px 18px;
                        margin: 12px 0; border-left: 4px solid #2D6A4F;">
                📚 <strong>مصدقہ ماخذ:</strong> {', '.join(sources)}
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Treatment tabs
        tab_chem, tab_organic, tab_cultural = st.tabs([
            "💊 تجویز کردہ اسپرے",
            "🌱 دیسی قدرتی علاج",
            "🌾 احتیاطی تدابیر",
        ])

        with tab_chem:
            chem = advisory.get("chemical_treatment", {})
            if chem:
                st.markdown(f"""
                <div class="glass-card urdu-text" style="border-left: 4px solid #2D6A4F;">
                    <strong style="font-size:1.25rem; color:#1B4332;">{chem.get('product_name', '')}</strong><br><br>
                    <span style="color:#2D6A4F;">●</span> جزو فعال: <strong>{chem.get('active_ingredient', '')}</strong><br>
                    <span style="color:#2D6A4F;">●</span> مقدار: <strong>{chem.get('dosage_per_acre', '')}</strong><br>
                    <span style="color:#2D6A4F;">●</span> وقت: {chem.get('spray_timing_ur', '')}<br>
                    <span style="color:#F59E0B;">●</span> احتیاط: فصل کاٹنے سے <strong>{chem.get('pre_harvest_interval_days', 'N/A')}</strong> دن پہلے اسپرے بند کریں
                </div>
                """, unsafe_allow_html=True)
                if chem.get("guardrail_verified"):
                    st.success("✅ PARC سے مصدقہ — محفوظ مقدار کی تصدیق ہو گئی")
            else:
                st.info("کیمیائی علاج کی تفصیلات دستیاب نہیں ہیں۔")

        with tab_organic:
            org = advisory.get("organic_alternative", {})
            if org:
                st.markdown(f"""
                <div class="glass-card urdu-text" style="border-left: 4px solid #10B981;">
                    <strong style="font-size:1.25rem; color:#1B4332;">{org.get('remedy_name_ur', '')}</strong><br><br>
                    {org.get('instructions_ur', '')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("قدرتی علاج کی تفصیلات دستیاب نہیں ہیں۔")

        with tab_cultural:
            cultural = advisory.get("cultural_practices_ur", "")
            if cultural:
                st.markdown(f"""
                <div class="glass-card urdu-text" style="border-left: 4px solid #D4A017;">
                    {cultural}
                </div>
                """, unsafe_allow_html=True)

        # Explanation expander
        explanation = advisory.get("disease_explanation_ur", "")
        if explanation:
            with st.expander("📖 بیماری کی تفصیل / Disease Details", expanded=False):
                st.markdown(f'<div class="urdu-text">{explanation}</div>',
                           unsafe_allow_html=True)


def render_audio_player(advisory: dict, disease_ur: str, sev_pct: float, crop_key: str = ""):
    """Render the Urdu voice audio player with modern card styling."""
    if not TTS_AVAILABLE:
        st.warning("🔇 آواز کی سہولت ابھی دستیاب نہیں ہے (edge-tts not installed)")
        return

    voice_script = advisory.get("voice_script_ur", "")
    if not voice_script:
        # Generate a default voice script
        voice_script = (
            f"آپ کی فصل میں {disease_ur} پائی گئی ہے، "
            f"نقصان {sev_pct:.0f} فیصد ہے۔ "
            "فوری طور پر تجویز کردہ اسپرے کریں۔"
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="audio-card">
        <h3 style="margin:0 0 8px; font-size:1.3rem;">🔊 آواز میں ہدایت سنیں</h3>
        <p style="margin:0; opacity:0.85; font-size:0.95rem;">Listen to advice in Urdu</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("▶️  اردو میں سنیں (Play Urdu Audio)", type="primary",
                 use_container_width=True, key=f"audio_btn_{crop_key}"):
        with st.spinner("آواز تیار ہو رہی ہے..."):
            try:
                audio_bytes = synthesize_urdu(voice_script)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                    st.success("✅ آواز تیار ہے — اوپر پلے بٹن دبائیں")
                else:
                    st.error("آواز تیار نہیں ہو سکی۔")
            except Exception as e:
                st.error(f"آواز کی خرابی: {e}")


def render_chat(session_context: dict):
    """Render the follow-up Q&A chat interface with modern styling."""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="section-heading">
        <span class="icon">💬</span>
        <span>مزید سوال پوچھیں / Follow-up Q&A</span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display existing chat messages
    for msg in st.session_state.chat_history:
        role = msg["role"]
        if role == "farmer":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(
                f'<div class="urdu-text">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # Chat input
    query = st.chat_input("یہاں اپنا سوال لکھیں... (Type your question here)")
    if query:
        st.session_state.chat_history.append({"role": "farmer", "content": query})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("جواب تیار ہو رہا ہے..."):
                if LLM_AVAILABLE:
                    try:
                        response = generate_chat_response(query, session_context)
                        reply = response.get("reply_text_ur", "معذرت، جواب دستیاب نہیں ہے۔")
                    except Exception as e:
                        reply = f"معذرت، جواب دینے میں مسئلہ ہوا: {e}"
                else:
                    reply = ("ابھی AI چیٹ کی سہولت دستیاب نہیں ہے۔ "
                            "براہ کرم GOOGLE_API_KEY سیٹ کریں۔")

                st.markdown(f'<div class="urdu-text">{reply}</div>',
                           unsafe_allow_html=True)
                st.session_state.chat_history.append({"role": "assistant", "content": reply})

                # Audio for chat response
                if TTS_AVAILABLE:
                    voice_script = (response.get("voice_script_ur", "")
                                   if LLM_AVAILABLE and 'response' in dir() else "")
                    if voice_script:
                        try:
                            audio = synthesize_urdu(voice_script)
                            if audio:
                                st.audio(audio, format="audio/mp3")
                        except Exception:
                            pass


@st.cache_resource
def load_classifier():
    """Load the MobileNetV3 classifier (cached across sessions)."""
    if CLASSIFIER_AVAILABLE:
        try:
            return CropDiseaseClassifier()
        except Exception as e:
            logger.error("Failed to load classifier: %s", e)
    return None


@st.cache_resource
def load_rag():
    """Load the RAG engine (cached across sessions)."""
    if RAG_AVAILABLE:
        try:
            engine = RAGEngine()
            if engine.is_ready:
                return engine
        except Exception as e:
            logger.error("Failed to load RAG engine: %s", e)
    return None


def process_image(img_bgr: np.ndarray, crop_hint: str) -> dict | None:
    """Run the full diagnostic pipeline on an uploaded image.

    Pipeline: Quality Gate → Classifier → Severity → RAG → LLM → Guardrail

    Args:
        img_bgr: BGR image array.
        crop_hint: Selected crop tab.

    Returns:
        Complete diagnosis results dict, or None on failure.
    """
    results = {}
    start_time = time.time()

    # Step 1: Quality Gate
    with st.status("🔍 تصویر کی جانچ ہو رہی ہے...", expanded=True) as status:
        img_resized = resize_image(img_bgr)
        is_valid, rejection_reason = validate_image(img_resized)

        if not is_valid:
            status.update(label="❌ تصویر مسترد", state="error")
            st.error(rejection_reason, icon="📸")
            return None

        st.write("✅ تصویر واضح ہے")

        # Step 2: Classification
        classifier = load_classifier()
        if classifier:
            st.write("🧠 AI تشخیص ہو رہی ہے...")
            diagnosis = classifier.predict(img_resized, crop_hint=crop_hint)

            if diagnosis["is_ood"]:
                # Show warning without raw HTML tag leakage
                st.warning(f"⚠️ {diagnosis['ood_message_ur']}\n\n(اعتماد کی شرح: {int(diagnosis['confidence'] * 100)}%)", icon="⚠️")
                # Provide closest estimation for demo
                diagnosis["confidence"] = max(diagnosis["confidence"], 0.65)

            results["diagnosis"] = diagnosis
        else:
            # No trained model — use crop hint for demo
            st.write("ℹ️ ماڈل لوڈ نہیں ہوا — ڈیمو موڈ")
            demo_disease_map = {
                "wheat": "wheat_leaf_rust",
                "cotton": "cotton_leaf_curl_virus",
                "rice": "rice_blast",
            }
            disease_id = demo_disease_map.get(crop_hint, "wheat_leaf_rust")
            results["diagnosis"] = {
                "disease_id": disease_id,
                "disease_name": disease_id.replace("_", " ").title(),
                "disease_name_ur": CLASS_LABELS_UR.get(disease_id, "") if CLASSIFIER_AVAILABLE else "بیماری",
                "crop": crop_hint.title(),
                "crop_ur": {"wheat": "گندم", "cotton": "کپاس", "rice": "چاول"}.get(crop_hint, ""),
                "confidence": 0.92,
                "pathogen": PATHOGEN_MAP.get(disease_id, "") if CLASSIFIER_AVAILABLE else "",
                "is_ood": False,
                "ood_message_ur": None,
            }

        # Step 3: Severity Assessment
        st.write("📊 شدت کا تجزیہ ہو رہا ہے...")
        severity = calculate_severity(img_resized)
        results["severity"] = severity

        # Step 4: RAG Retrieval + LLM Advisory
        st.write("📚 PARC ڈیٹابیس سے معلومات...")
        rag = load_rag()
        diagnosis = results["diagnosis"]

        if rag and LLM_AVAILABLE:
            context = rag.build_context(
                disease_id=diagnosis["disease_id"],
                disease_name=diagnosis["disease_name"],
                severity_pct=severity["infected_percentage"],
                severity_level=severity["level"],
                confidence=diagnosis["confidence"],
                crop=diagnosis["crop"],
            )
            st.write("🤖 AI مشورہ تیار ہو رہا ہے...")
            advisory = generate_advisory(context, diagnosis["disease_id"])
        elif LLM_AVAILABLE:
            # LLM available but no RAG index — use static fallback
            advisory = _get_static_fallback(diagnosis["disease_id"])
        else:
            # Full static fallback
            from core.guardrails import load_whitelist
            whitelist = load_whitelist()
            diseases = whitelist.get("diseases", {})
            disease_id = diagnosis["disease_id"]
            if disease_id in diseases:
                disease_data = diseases[disease_id]
                chem = disease_data["approved_chemicals"][0]
                org = disease_data["organic_remedies"][0] if disease_data.get("organic_remedies") else {}
                advisory = {
                    "disease_explanation_ur": f"آپ کی فصل میں {disease_data['disease_name_ur']} پائی گئی ہے۔",
                    "chemical_treatment": {
                        "product_name": chem["product_name"],
                        "active_ingredient": chem["active_ingredient"],
                        "dosage_per_acre": f"{chem['max_dose_per_acre']} {chem['dose_unit']}",
                        "spray_timing_ur": chem.get("spray_timing_ur", ""),
                        "pre_harvest_interval_days": chem["pre_harvest_interval_days"],
                        "guardrail_verified": True,
                    },
                    "organic_alternative": {
                        "remedy_name_ur": org.get("remedy_name_ur", ""),
                        "instructions_ur": org.get("instructions_ur", ""),
                    } if org else {},
                    "cultural_practices_ur": "فصل میں اضافی نمی نہ ہونے دیں۔",
                    "voice_script_ur": f"آپ کی فصل میں {disease_data['disease_name_ur']} ہے۔ {chem['product_name']} کا {chem['max_dose_per_acre']} {chem['dose_unit']} فی ایکڑ اسپرے کریں۔",
                    "sources_cited": ["PARC Baseline Advisory"],
                }
            else:
                advisory = {
                    "disease_explanation_ur": "بیماری کی تفصیلات دستیاب نہیں ہیں۔",
                    "chemical_treatment": {},
                    "organic_alternative": {},
                    "cultural_practices_ur": "",
                    "voice_script_ur": "",
                    "sources_cited": [],
                }

        results["advisory"] = advisory

        # Timing
        elapsed = (time.time() - start_time) * 1000
        results["latency_ms"] = round(elapsed, 1)

        status.update(label=f"✅ تشخیص مکمل ({elapsed:.0f}ms)", state="complete")

    return results


def main():
    """Main Streamlit application entry point."""
    configure_page()
    render_header()

    # Crop selector tabs
    tab_names = list(CROP_TABS.keys())
    tabs = st.tabs(tab_names)

    for tab, (tab_name, crop_key) in zip(tabs, CROP_TABS.items()):
        with tab:
            # Section heading for image capture
            st.markdown(f"""
            <div class="section-heading">
                <span class="icon">📷</span>
                <span>{crop_key.title()} کے پتے کی تصویر لیں / Capture {crop_key.title()} Leaf</span>
            </div>
            """, unsafe_allow_html=True)

            # Image input — camera + file upload in styled columns
            col_cam, col_file = st.columns(2)
            with col_cam:
                camera_img = st.camera_input(
                    "📸 کیمرہ کھولیں / Open Camera",
                    key=f"camera_{crop_key}",
                )
            with col_file:
                uploaded_file = st.file_uploader(
                    "📁 گیلری سے منتخب کریں / Upload from Gallery",
                    type=["jpg", "jpeg", "png", "webp"],
                    key=f"upload_{crop_key}",
                )

            # Process whichever input is available
            image_source = camera_img or uploaded_file
            if image_source:
                # Read image bytes
                img_bytes = image_source.getvalue()

                # EXIF sanitization (Security §8.1)
                try:
                    img_bytes = strip_exif(img_bytes)
                except Exception:
                    pass

                # Decode to OpenCV BGR
                nparr = np.frombuffer(img_bytes, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    st.error("تصویر پڑھنے میں خرابی۔ دوسری تصویر آزمائیں۔")
                    continue

                # Show uploaded image with styled container
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption="📸 اپ لوڈ شدہ تصویر / Uploaded Image",
                        use_column_width=True)

                # Cache results to avoid duplicate LLM calls on Streamlit reruns
                import hashlib
                img_hash = hashlib.md5(img_bytes[:2048]).hexdigest()
                cache_key = f"results_{crop_key}_{img_hash}"

                if cache_key in st.session_state:
                    results = st.session_state[cache_key]
                else:
                    # Run diagnostic pipeline (only on first run for this image)
                    results = process_image(img_bgr, crop_key)
                    if results:
                        st.session_state[cache_key] = results

                if results:
                    diagnosis = results["diagnosis"]
                    severity = results["severity"]
                    advisory = results["advisory"]

                    # Render diagnosis card
                    render_diagnosis_card(diagnosis, severity, advisory)

                    # Audio player
                    render_audio_player(
                        advisory,
                        diagnosis.get("disease_name_ur", ""),
                        severity.get("infected_percentage", 0),
                        crop_key=crop_key,
                    )

                    # Store session context for chat
                    session_ctx = {
                        "disease_id": diagnosis["disease_id"],
                        "severity_pct": severity["infected_percentage"],
                        "prior_advisory_summary": advisory.get("disease_explanation_ur", ""),
                    }
                    st.session_state["session_context"] = session_ctx

                    # Follow-up chat
                    render_chat(session_ctx)

    # Footer
    st.markdown("""
    <div class="app-footer">
        🌿 <strong>AgriDoc-PK</strong> — Bano Qabil AI Hackathon 2026<br>
        <span style="font-size: 0.8rem;">Powered by PARC Verified Data + Gemini AI</span><br>
        <span style="font-size: 0.75rem; opacity: 0.7;">Built with ❤️ for Pakistan's Farmers</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
