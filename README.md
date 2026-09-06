# 🌿 AgriDoc-PK — AI Crop Pathology & Localized Agronomy Advisory

> **Bano Qabil AI Hackathon 2026** | Alibaba Cloud & Alkhidmat Foundation  
> Built by Mubashir Uddin — AI/ML Engineer & Systems Architect

AgriDoc-PK is an AI-powered mobile web application that provides Pakistani farmers with instant crop disease diagnosis, severity assessment, PARC-verified treatment recommendations in Urdu, and voice-guided agricultural advisory — all from a single leaf photo.

---

## Features

| Module | Capability |
|--------|-----------|
| **Image Quality Gate** | Rejects blurry/dark/overexposed photos with Urdu feedback |
| **Disease Classifier** | MobileNetV3-Small (10 classes × 3 crops) with OOD detection |
| **Severity Engine** | CLAHE + HSV segmentation → Mild/Moderate/Severe triage |
| **RAG Advisory** | ChromaDB vector search over PARC agronomic bulletins |
| **Grounded LLM** | Gemini Flash with anti-hallucination guardrails |
| **Voice Output** | Edge-TTS Urdu speech synthesis (ur-PK-AsadNeural) |
| **Follow-up Chat** | Context-aware Q&A maintaining diagnostic session |

### Crops Supported
- **Wheat**: Leaf Rust, Stripe Rust, Loose Smut
- **Cotton**: Leaf Curl Virus (CLCuV), Bacterial Blight
- **Rice**: Blast, Brown Spot

---

## Quick Start

### Prerequisites
- Python 3.10+
- pip / venv

### Installation

```bash
# Clone and enter project
cd "Smart Agriculture"

# Create virtual environment
python -m venv venv
venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Build the PARC knowledge index (one-time)
python knowledge_base/build_index.py
```

### Set API Key (Optional)

```bash
# For live Gemini AI advisory (optional — static PARC fallback works without it)
set GOOGLE_API_KEY=your_key_here     # Windows
# export GOOGLE_API_KEY=your_key_here  # Linux/Mac
```

### Launch

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Use your phone camera or upload a leaf photo.

---
## Dataset Downloading from Hugging Face

### Strategy 1: Demo Day (Recommended, ~150 MB, 30 min)
```bash
# Install streaming library
pip install datasets huggingface_hub

# Stream only 300 images per class (~3000 total, ~150MB)
python scripts/prepare_dataset.py --max_per_class 300

# Then train (fast — ~5-10 min on CPU)
python models/train.py --data_dir data/processed --epochs 10 --batch_size 16
```

---

### Strategy 2: Full Dataset (~2.5 GB, 3-4 hours)
```bash
# Download everything (no limit)
python scripts/prepare_dataset.py --full

# Train (slower — ~20-30 min on CPU)
python models/train.py --data_dir data/processed --epochs 20 --batch_size 32
```

### Strategy 3: Skip Training Entirely for Demo Day
```bash
#Use a pretrained model with no fine-tuning — the app still works via Gemini API for advisory:
set GOOGLE_API_KEY=your_gemini_key_here
streamlit run app.py
```


---

## Model Training

```bash
# 1. Prepare dataset (expected structure: data/processed/train/<class_name>/*)
python models/train.py --data_dir data/processed --epochs 25 --batch_size 32

# 2. Evaluate on test set
python models/evaluate.py

# 3. Export to ONNX (quantized)
python models/quantize.py
```

---

## Project Structure

```
Smart Agriculture/
├── app.py                          # Streamlit mobile UI
├── requirements.txt                # Frozen dependencies
├── Dockerfile                      # Container deployment
├── README.md                       # This file
│
├── core/                           # Core engine modules
│   ├── preprocessor.py             # Image quality gate
│   ├── classifier.py               # MobileNetV3 inference
│   ├── severity_engine.py          # HSV severity analysis
│   ├── rag_engine.py               # ChromaDB RAG retrieval
│   ├── llm_orchestrator.py         # Gemini Flash advisory
│   ├── guardrails.py               # Chemical safety validator
│   └── tts_engine.py               # Urdu voice synthesis
│
├── knowledge_base/                 # PARC agronomic corpus
│   ├── parc_whitelist.json         # Verified chemical registry
│   ├── build_index.py              # ChromaDB indexer
│   └── parc_documents/             # PARC advisory bulletins
│
├── models/                         # Training pipeline
│   ├── train.py                    # MobileNetV3 fine-tuning
│   ├── evaluate.py                 # Metrics & confusion matrix
│   ├── quantize.py                 # ONNX INT8 export
│   └── weights/                    # Model checkpoints
│
├── data/                           # Dataset
│   ├── raw/                        # Original images
│   └── processed/                  # Augmented train/val/test
│
├── data_store/chroma_db/           # ChromaDB vector index
│
└── tests/                          # Test suite
    ├── test_core.py                # Phase 1-2 unit tests (24)
    ├── test_phase3.py              # Phase 3 module tests (26)
    └── test_integration.py         # Integration & benchmarks
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/test_core.py -v              # Quality gate, severity, guardrails
python -m pytest tests/test_phase3.py -v             # Classifier, RAG, LLM, TTS
python -m pytest tests/test_integration.py -v        # Groundedness, pipeline integration
```

---

## Architecture

```
Camera Photo → Quality Gate → MobileNetV3 Classifier → Severity Engine
                                      ↓                      ↓
                              Disease ID + Conf     Infected % + Triage
                                      ↓                      ↓
                                  ChromaDB RAG ← PARC Bulletins
                                      ↓
                              Gemini Flash LLM → Structured JSON
                                      ↓
                              Guardrail Validator ← PARC Whitelist
                                      ↓
                              Urdu Advisory Card + Voice (Edge-TTS)
                                      ↓
                              Follow-up Chat (Context-Aware)
```

---

## Safety & Guardrails

- **0% hallucination policy**: Every chemical recommendation is validated against the PARC-approved whitelist
- **Dosage clamping**: LLM-suggested dosages exceeding PARC maximums are automatically clamped
- **Unknown chemical override**: Non-whitelisted chemicals are replaced with verified PARC baseline
- **EXIF stripping**: All GPS and personal metadata removed from uploaded photos
- **Static fallback**: Full advisory available even without internet (no LLM/TTS required)

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Vision | PyTorch, TorchVision, OpenCV, Albumentations |
| RAG | ChromaDB (MiniLM-L6-v2 embeddings) |
| LLM | Google Gemini Flash via google-genai SDK |
| Voice | Edge-TTS (ur-PK-AsadNeural) |
| UI | Streamlit (mobile-first PWA) |
| Testing | pytest, scikit-learn |

---

## Dataset

[mubashiruddin01/agridoc-pk-dataset](https://huggingface.co/datasets/mubashiruddin01/agridoc-pk-dataset) on Hugging Face

---

## License

Built for the Bano Qabil AI Hackathon 2026. All PARC advisory content is adapted from publicly available agricultural extension bulletins.

---

**Author**: Mubashir Uddin | AI/ML Engineer & Systems Architect
