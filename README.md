# Community Helper Guidebook
### An Offline, Multimodal Healthcare Assistant for Low-Resource India

[![MedGemma Impact Challenge](https://img.shields.io/badge/Kaggle-MedGemma%20Impact%20Challenge-20BEFF?logo=kaggle)](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![MedGemma 4B-IT](https://img.shields.io/badge/Model-MedGemma%204B--IT-orange)](https://huggingface.co/google/medgemma-4b-it)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> **Responsible AI Notice:** This system is assistive guidance for community health workers only. It does NOT diagnose, prescribe, or replace clinical care. Every output instructs escalation to a qualified healthcare professional. Emergency: **108**

---

## Table of Contents

- [What This Is](#-what-this-is)
- [The Problem](#-the-problem)
- [Pipeline Overview](#-pipeline-overview)
- [HAI-DEF Models Used](#-hai-def-models-used)
- [RAG Knowledge Base](#-rag-knowledge-base)
- [Demo Scenarios](#-demo-scenarios)
- [Quick Start — Kaggle](#-quick-start--kaggle)
- [Quick Start — Local](#-quick-start--local)
- [Project Structure](#-project-structure)
- [Assumptions & Limitations](#-assumptions--limitations)
- [Production Roadmap](#-production-roadmap)
- [Submission Documents](#-submission-documents)
- [License](#-license)

---

## What This Is

The Community Helper Guidebook converts spoken notes and medical images from community health workers into safe, simplified, multilingual **Helper Guidance Packs** — grounded in real Indian healthcare regulations from MoHFW and NHM.

**For:** ASHA Workers · Community Health Workers · NGO Volunteers · First-contact helpers

**Not for:** Patients directly · Clinical diagnosis · Treatment decisions

**Languages:** Hindi · Tamil · Bengali · Telugu · Marathi · Gujarati · Kannada · English

---

## The Problem

India has 1.04 million ASHA workers as the first point of contact for over 600 million rural citizens. They face three daily gaps:

| Gap | Impact |
|---|---|
| **Knowledge gap** | No safe first aid guidance without clinical training |
| **Urgency gap** | Cannot reliably assess when to escalate to a hospital |
| **Language gap** | Work in local languages with no multilingual AI support |

Existing digital health tools are built for doctors or patients — not for the community helpers who show up first.

---

## Pipeline Overview

```
 Audio (upload or record)      Text input         Image upload
          ↓                            ↓                     ↓
   Whisper STT              [skip confirmation]        SigLIP encoding
   (transcribe mode)
          ↓
   Confirmation Layer
   (helper reviews & corrects transcript)
          ↓
   Google Translate → English
   (IndicTrans2 in production)
          ↓
   ★ RAG Retrieval ★
   FAISS + sentence-transformers
   over real MoHFW / NHM documents
          ↓  ←─────────────── image context
   Tool Pre-execution (all 4 tools run deterministically)
     ├── get_first_aid_steps()     → MedGemma, strict boundary prompt
     ├── check_urgency()           → keyword-based, spoken phrases
     ├── lookup_nearest_facility() → India PHC/CHC database
     └── generate_referral_note()  → structured doctor handoff
          ↓
   MedGemma 4B-IT Synthesis
   (writes structured guidance pack from tool results + RAG context)
          ↓
   Safety Guardrails
   (forbidden phrase scan + immediate escalation triggers)
          ↓
   Google Translate → Local Language
   (IndicTrans2 in production)
          ↓
    Helper Guidance Pack
```

---

## HAI-DEF Models Used

| Model | Role | Notes |
|---|---|---|
| **MedGemma 4B-IT** | Core reasoning — synthesises case + tool results into guidance pack | 4-bit NF4 quantized for Kaggle T4 GPU |
| **SigLIP** | Medical image classification (MedSigLIP proxy) | Zero-shot, 10 medical image categories |

### Why MedGemma

General-purpose LLMs will readily provide diagnoses and treatment protocols when asked. MedGemma's medical training gives it the contextual understanding to reason about health cases while respecting the hard clinical scope-of-practice boundaries we enforce through system prompting and guardrails.

### Why SigLIP as MedSigLIP Proxy

SigLIP's strong zero-shot classification provides meaningful image context to MedGemma without requiring fine-tuning. Production would use a fine-tuned MedSigLIP checkpoint on Indian clinical images.

---

## RAG Knowledge Base

The system retrieves context from real Indian government documents at inference time — addressing the gap between MedGemma's Western training data and Indian healthcare regulations.

| Document | Source | Format |
|---|---|---|
| National Health Policy 2017 | MoHFW | Downloaded PDF at runtime |
| IPHS Guidelines for PHC/CHC | NHM 2022 | Downloaded PDF at runtime |
| NHM Community Health Worker Guidelines | NHM | Downloaded PDF at runtime |
| ASHA Training Module excerpts | MoHFW | Embedded directly |
| Ayushman Bharat PM-JAY Guidelines | NHA | Embedded directly |

> **Production note:** LangChain + ChromaDB replaces FAISS for persistent storage, metadata filtering, and automated document ingestion pipelines. See [Production Roadmap](PRODUCTION_ROADMAP.md).

---

## Demo Scenarios

| Scenario | Input Language | Output Language | Case |
|---|---|---|---|
| 1 | Hindi | Hindi | Child with prolonged fever, 40km from hospital |
| 2 | English | Hindi | Burn injury with image, mustard oil applied |
| 3 | English | English | Elderly man — chest pain + difficulty breathing (immediate escalation) |

---

## Quick Start — Kaggle

**Prerequisites:**
1. Kaggle account with GPU quota
2. HuggingFace account — accept [MedGemma license](https://huggingface.co/google/medgemma-4b-it)
3. HuggingFace classic Read token → add to Kaggle Secrets as `HF_TOKEN`

**Setup:**
1. Create a new Kaggle notebook
2. Settings → Accelerator → **GPU T4 x2**
3. Settings → Internet → **ON**
4. Copy each `# %% [code]` cell from `medgemma_community_helper_FINAL.py` as separate notebook cells
5. Add HF login cell before Cell 4:

```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
login(token=secrets.get_secret("HF_TOKEN"))
```

6. Run all cells in order — Cell 18 launches the Gradio UI with a public share link

**Expected runtime:** ~35–45 minutes total (model downloads + RAG build + 3 demo scenarios)

---

## Quick Start — Local

```bash
# Clone
git clone https://github.com/[YOUR-USERNAME]/community-helper-guidebook.git
cd community-helper-guidebook

# Install dependencies
pip install -r requirements.txt

# Login to HuggingFace
huggingface-cli login

# Run (requires CUDA GPU with 16GB+ VRAM, or CPU with 32GB RAM)
python medgemma_community_helper_FINAL.py
```

> **Note:** For CPU-only or low-RAM environments, set `USE_4BIT = False` in Cell 3 and `WHISPER_SIZE = "small"`.

---

## Project Structure

```
community-helper-guidebook/
│
├── medgemma_community_helper_FINAL.py   # Main Kaggle notebook (19 cells)
├── PRODUCTION_ROADMAP.md                # Production upgrade plan
├── KAGGLE_SETUP.md                      # Detailed Kaggle setup guide
├── requirements.txt                     # Python dependencies
├── .gitignore                           # Excludes models, cache, audio
│
├── docs/
│   ├── writeup.docx                     # Kaggle competition writeup
│   └── video_script.md                  # 3-minute video script
│
└── README.md
```

---

## Assumptions & Limitations

| Assumption | Detail |
|---|---|
| User is a CHW / ASHA worker | Not a patient or doctor |
| First aid boundary | Bare hands only — no medication, no equipment |
| Central regulations only | State-specific ASHA guidelines out of scope |
| Google Translate for demo | IndicTrans2 required for production offline use |
| SigLIP as proxy | Not a fine-tuned medical image model |
| PHC database is representative | Production needs live NHA GPS registry |
| Clinical validation required | MedGemma outputs must be validated by qualified medical professionals before any real CHW programme deployment |

---

## Production Roadmap

See [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) for the full plan. Key items:

- **Offline deployment** — Ollama + ONNX + IndicTrans2 packaged as a single installable app for Android tablets with no internet required
- **Production RAG** — LangChain + ChromaDB replacing FAISS for persistent, automatically updated document store
- **Fine-tuning** — QLoRA MedGemma on synthetic CHW scenarios; Whisper on Indian dialect audio
- **Real tool backends** — NHA facility registry API, state HMIS integration for digital referral tracking
- **Clinical validation** — Partner with NHM or Indian NGO for real-world validation

---

## Submission Documents

| Document | Link |
|---|---|
| Kaggle Notebook | [YOUR KAGGLE NOTEBOOK URL] |
| Demo Video | [YOUR VIDEO URL] |
| Competition Writeup | `docs/writeup.docx` |
| Video Script | `docs/video_script.md` |

---

## Responsible AI

- **No diagnosis** — system never names or identifies a condition
- **No treatment advice** — no medications, dosages, or clinical procedures  
- **Human-in-the-loop at input** — helper reviews and corrects transcript before generation
- **Human-in-the-loop at output** — every pack instructs escalation to a professional
- **Indian regulatory grounding** — RAG over MoHFW / NHM / NHP 2017
- **Privacy-first** — Whisper runs locally; no audio leaves the device in production
- **Language-inclusive** — 8 Indian languages, designed for non-literate helpers
- **Urgency in plain language** — spoken phrases, never clinical labels

---

## License

MIT License — see [LICENSE](LICENSE)

---

*Built for the MedGemma Impact Challenge · Kaggle 2025*  
*India-first · Offline-first · Community-centric · Human-in-the-loop*  
*Grounded in MoHFW / NHM / National Health Policy 2017*
