# 🌿 AgriDoc-PK — AI Crop Pathology & Localized Agronomy Advisory

> **Bano Qabil AI Hackathon 2026** | Alibaba Cloud & Alkhidmat Foundation  
> Built by Mubashir Uddin — AI/ML Engineer & Systems Architect

AgriDoc-PK is an AI-powered mobile web application that provides Pakistani farmers with **instant crop disease diagnosis**, severity assessment, **PARC-verified treatment recommendations in Urdu**, and voice-guided agricultural advisory — all from a single leaf photo.

---

## 📑 Table of Contents

- [Features](#-features)
- [Crops Supported](#-crops-supported)
- [Prerequisites](#-prerequisites)
- [Getting Started](#-getting-started)
  - [Step 1 — Clone the Repository](#step-1--clone-the-repository)
  - [Step 2 — Create a Virtual Environment](#step-2--create-a-virtual-environment)
  - [Step 3 — Activate the Virtual Environment](#step-3--activate-the-virtual-environment)
  - [Step 4 — Install Dependencies](#step-4--install-dependencies)
  - [Step 5 — Build the Knowledge Base Index](#step-5--build-the-knowledge-base-index-one-time)
  - [Step 6 — Set Your Gemini API Key (Optional)](#step-6--set-your-gemini-api-key-optional)
  - [Step 7 — Launch the App](#step-7--launch-the-app)
- [Dataset Download](#-dataset-download)
- [Model Training Pipeline](#-model-training-pipeline)
- [Testing](#-testing)
- [Docker Deployment](#-docker-deployment)
- [Project Structure](#-project-structure)
- [Architecture](#-architecture)
- [Safety & Guardrails](#-safety--guardrails)
- [Technology Stack](#-technology-stack)
- [Team Members](#-team-members)
- [License](#-license)

---

## ✨ Features

| Module | Capability |
|--------|-----------|
| **Image Quality Gate** | Rejects blurry / dark / overexposed photos with Urdu feedback |
| **Disease Classifier** | MobileNetV3-Small (10 classes × 3 crops) with OOD detection |
| **Severity Engine** | CLAHE + HSV segmentation → Mild / Moderate / Severe triage |
| **RAG Advisory** | ChromaDB vector search over PARC agronomic bulletins |
| **Grounded LLM** | Gemini Flash with anti-hallucination guardrails |
| **Voice Output** | Edge-TTS Urdu speech synthesis (ur-PK-AsadNeural) |
| **Follow-up Chat** | Context-aware Q&A maintaining diagnostic session |

---

## 🌾 Crops Supported

| Crop | Diseases Detected |
|------|------------------|
| **Wheat** | Leaf Rust, Stripe Rust, Loose Smut |
| **Cotton** | Leaf Curl Virus (CLCuV), Bacterial Blight |
| **Rice** | Blast, Brown Spot |

Each crop also includes a **Healthy** class for negative diagnosis.

---

## 📋 Prerequisites

Before you begin, make sure you have the following installed:

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Python** | 3.10 or higher | `python --version` |
| **pip** | Latest | `pip --version` |
| **Git** | Any | `git --version` |

> **Optional:** A [Google Gemini API key](https://aistudio.google.com/apikey) for live AI advisory. The app still works without it using static PARC fallback data.

---

## 🚀 Getting Started

Follow these steps **in order** to set up and run AgriDoc-PK on your local machine.

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Mubashir-Uddin-AI/AgriDoc-PK.git
cd AgriDoc-PK
```

### Step 2 — Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3 — Activate the Virtual Environment

**Windows (PowerShell):**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

> You should see `(venv)` appear at the beginning of your terminal prompt.

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs PyTorch, OpenCV, ChromaDB, Streamlit, and all other required packages (~1–2 GB depending on your platform).

### Step 5 — Build the Knowledge Base Index (One-Time)

```bash
python knowledge_base/build_index.py
```

This creates the ChromaDB vector index from PARC agronomic bulletins in `data_store/chroma_db/`. You only need to run this once.

### Step 6 — Set Your Gemini API Key (Optional)

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

**Windows (CMD):**
```cmd
set GOOGLE_API_KEY=your_api_key_here
```

**Linux / macOS:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
```

> **Note:** Without this key, the app uses static PARC-verified fallback recommendations. All core features (image diagnosis, severity analysis, Urdu advisory) work without it.

### Step 7 — Launch the App

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**. Use your phone camera or upload a leaf photo to get started.

---

## 📦 Dataset Download

The dataset is hosted on Hugging Face: [mubashiruddin01/agridoc-pk-dataset](https://huggingface.co/datasets/mubashiruddin01/agridoc-pk-dataset)

Choose **one** of the three strategies below based on your time and storage constraints.

### Strategy A — Demo-Ready (Recommended)

**~150 MB download · ~30 minutes · 300 images per class**

Best for quick demos, hackathon presentations, or testing.

```bash
# Step 1: Install the Hugging Face library
pip install datasets huggingface_hub

# Step 2: Download 300 images per class (~3,000 total images)
python scripts/prepare_dataset.py --max_per_class 300

# Step 3: Verify the download
#   You should see folders under data/processed/train/ and data/processed/val/
#   with 9 class subfolders each
```

### Strategy B — Full Dataset

**~2.5 GB download · 3–4 hours · All images**

Best for production training and maximum accuracy.

```bash
# Step 1: Install the Hugging Face library (skip if already done)
pip install datasets huggingface_hub

# Step 2: Download the complete dataset
python scripts/prepare_dataset.py --full

# Step 3: Verify the download
#   Check data/processed/train/ and data/processed/val/ for all class folders
```

### Strategy C — Skip Download Entirely

**No dataset needed · Use Gemini API for advisory only**

If you just want to run the app without the disease classifier model:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
streamlit run app.py
```

**Linux / macOS:**
```bash
export GOOGLE_API_KEY="your_api_key_here"
streamlit run app.py
```

---

## 🧠 Model Training Pipeline

After downloading the dataset (Strategy A or B above), follow these steps to train, evaluate, and export the model.

### Step 1 — Train the Model

```bash
python models/train.py --data_dir data/processed --epochs 25 --batch_size 32
```

**Available arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/processed` | Root directory with `train/` and `val/` splits |
| `--epochs` | `25` | Number of training epochs |
| `--batch_size` | `32` | Batch size (reduce to `16` if you run out of memory) |
| `--lr` | `0.001` | Initial learning rate |
| `--num_workers` | `2` | DataLoader worker processes |

> **Tip:** For the demo dataset (Strategy A), use `--epochs 10 --batch_size 16` for faster training (~5–10 min on CPU).

The best model checkpoint is saved to `models/weights/mobilenetv3_best.pth`.

### Step 2 — Evaluate the Model

```bash
python models/evaluate.py --data_dir data/processed --weights models/weights/mobilenetv3_best.pth
```

This prints a classification report with per-class precision, recall, F1-score, confusion matrix, and OOD detection statistics.

### Step 3 — Export to ONNX (Quantized)

```bash
python models/quantize.py --weights models/weights/mobilenetv3_best.pth
```

This exports the model in three formats:
- `mobilenetv3_fp32.onnx` — Full-precision ONNX
- `mobilenetv3_quantized.pth` — Quantized PyTorch
- `mobilenetv3_quantized.onnx` — INT8 quantized ONNX (~4 MB, ~75% size reduction)

All exported files are saved to `models/weights/`.

---

## 🧪 Testing

### Run All Tests

```bash
python -m pytest tests/ -v
```

### Run Individual Test Suites

```bash
# Core modules: quality gate, severity engine, guardrails
python -m pytest tests/test_core.py -v

# Phase 3 modules: classifier, RAG, LLM, TTS
python -m pytest tests/test_phase3.py -v

# Integration tests and benchmarks
python -m pytest tests/test_integration.py -v
```

---

## 🐳 Docker Deployment

### Step 1 — Build the Docker Image

```bash
docker build -t agridoc-pk .
```

### Step 2 — Run the Container

**Without Gemini API key (static fallback mode):**
```bash
docker run -p 8501:8501 agridoc-pk
```

**With Gemini API key (full AI advisory):**
```bash
docker run -p 8501:8501 -e GOOGLE_API_KEY="your_api_key_here" agridoc-pk
```

### Step 3 — Open the App

Navigate to **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
AgriDoc-PK/
├── app.py                          # Streamlit mobile-first UI
├── requirements.txt                # Pinned Python dependencies
├── Dockerfile                      # Container deployment config
├── README.md                       # This file
│
├── core/                           # Core engine modules
│   ├── preprocessor.py             # Image quality gate (blur, exposure)
│   ├── classifier.py               # MobileNetV3-Small inference
│   ├── severity_engine.py          # CLAHE + HSV severity analysis
│   ├── rag_engine.py               # ChromaDB RAG retrieval
│   ├── llm_orchestrator.py         # Gemini Flash advisory generation
│   ├── guardrails.py               # Chemical safety validator
│   └── tts_engine.py               # Urdu voice synthesis (Edge-TTS)
│
├── knowledge_base/                 # PARC agronomic corpus
│   ├── build_index.py              # ChromaDB vector index builder
│   ├── parc_whitelist.json         # Verified chemical registry
│   └── parc_documents/             # PARC advisory bulletins
│
├── models/                         # Training & export pipeline
│   ├── train.py                    # MobileNetV3-Small fine-tuning
│   ├── evaluate.py                 # Metrics, confusion matrix, OOD stats
│   ├── quantize.py                 # ONNX INT8 export & quantization
│   └── weights/                    # Model checkpoints (git-ignored)
│
├── scripts/                        # Utility scripts
│   ├── prepare_dataset.py          # Hugging Face dataset downloader
│   └── fast_download.py            # Bulk download helper
│
├── data/                           # Dataset (git-ignored)
│   ├── raw/                        # Original images
│   └── processed/                  # Augmented train/val splits
│
├── data_store/
│   └── chroma_db/                  # ChromaDB vector index
│
├── .streamlit/
│   └── config.toml                 # Streamlit theme & settings
│
└── tests/                          # Test suite (50+ tests)
    ├── test_core.py                # Phase 1–2 unit tests
    ├── test_phase3.py              # Phase 3 module tests
    └── test_integration.py         # Integration & benchmarks
```

---

## 🏗️ Architecture

```
┌─────────────┐
│ Camera/Photo│
└──────┬──────┘
       ▼
┌─────────────────┐    Reject + Urdu
│  Quality Gate    │───────────────────► User Feedback
│ (blur/exposure)  │
└──────┬──────────┘
       ▼ Pass
┌─────────────────┐
│  MobileNetV3    │──► Disease ID + Confidence
│  Classifier     │       │
└──────┬──────────┘       │
       ▼                  ▼
┌─────────────────┐  ┌──────────────┐
│ Severity Engine │  │ ChromaDB RAG │◄── PARC Bulletins
│ (HSV analysis)  │  │ (retrieval)  │
└──────┬──────────┘  └──────┬───────┘
       │  Infected %        │  Context
       └────────┬───────────┘
                ▼
       ┌────────────────┐
       │  Gemini Flash  │
       │  LLM Advisory  │
       └───────┬────────┘
               ▼
       ┌────────────────┐
       │  Guardrail     │◄── PARC Whitelist
       │  Validator     │    (chemical safety)
       └───────┬────────┘
               ▼
       ┌────────────────┐
       │ Urdu Advisory  │
       │ Card + Voice   │──► Edge-TTS (ur-PK)
       └───────┬────────┘
               ▼
       ┌────────────────┐
       │  Follow-up     │
       │  Chat (Q&A)    │
       └────────────────┘
```

---

## 🛡️ Safety & Guardrails

| Guardrail | Description |
|-----------|-------------|
| **0% Hallucination Policy** | Every chemical recommendation is validated against the PARC-approved whitelist |
| **Dosage Clamping** | LLM-suggested dosages exceeding PARC maximums are automatically clamped |
| **Unknown Chemical Override** | Non-whitelisted chemicals are replaced with verified PARC baseline |
| **EXIF Stripping** | All GPS and personal metadata removed from uploaded photos |
| **Static Fallback** | Full advisory available even without internet (no LLM/TTS required) |

---

## 🛠️ Technology Stack

| Component | Technology |
|-----------|-----------|
| **Deep Learning** | PyTorch 2.3, TorchVision, MobileNetV3-Small |
| **Image Processing** | OpenCV, Albumentations, Pillow |
| **Vector Database** | ChromaDB with MiniLM-L6-v2 embeddings |
| **LLM** | Google Gemini Flash via `google-genai` SDK |
| **Voice** | Edge-TTS (ur-PK-AsadNeural) |
| **Web UI** | Streamlit (mobile-first, PWA-ready) |
| **Model Export** | ONNX Runtime (INT8 quantization) |
| **Testing** | pytest, scikit-learn |
| **Deployment** | Docker |

---

## 👥 Team Members

| Name | Role |
|------|------|
| **Mubashir Uddin** | AI/ML Engineer & Systems Architect |
| **Saad Mustafa** | Workflow Manager |
| **Dheeraj Khatri** | Data Engineer |

---

## 📜 License

Built for the **Bano Qabil AI Hackathon 2026**. All PARC advisory content is adapted from publicly available agricultural extension bulletins.

---

## 📊 Dataset

The complete dataset is available on Hugging Face:  
🔗 [mubashiruddin01/agridoc-pk-dataset](https://huggingface.co/datasets/mubashiruddin01/agridoc-pk-dataset)
