# 🚀 Kaggle Setup Guide

## Step-by-Step Instructions to Run the Community Helper Guidebook on Kaggle

---

## 1. Kaggle Notebook Settings

Create a new notebook on Kaggle and configure:

| Setting | Value |
|---|---|
| Accelerator | GPU T4 x2 |
| Internet | ON |
| Persistence | Files only |

---

## 2. Accept MedGemma License

Go to: https://huggingface.co/google/medgemma-4b-it

Click **"Agree and access repository"** — you must be logged in to your HuggingFace account.

> Without this step, Cell 4 will fail with a 403 error.

---

## 3. Create HuggingFace Token

1. Go to: https://huggingface.co/settings/tokens
2. Click **"New token"**
3. Select token type: **Classic**
4. Role: **Read**
5. Copy the token (starts with `hf_`)

---

## 4. Add Token to Kaggle Secrets

In your notebook:
1. Click **Add-ons** (top menu) → **Secrets**
2. Click **Add new secret**
3. Name: `HF_TOKEN`
4. Value: paste your HuggingFace token
5. Toggle **"Notebook has access"** to ON

---

## 5. Add HuggingFace Login Cell

Add this as a new cell **before Cell 4** (before MedGemma loads):

```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient

secrets  = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
login(token=hf_token)
print("✅ HuggingFace login successful.")
```

---

## 6. Copy Notebook Cells

Copy each `# %% [code]` block from `medgemma_community_helper_FINAL.py`
as a separate cell in your Kaggle notebook. The file has 19 cells total.

Cell structure:
| Cell | Content |
|---|---|
| 1 | Markdown — pipeline overview |
| 2 | Install dependencies |
| 3 | Imports & global config |
| **HF Login** | **Add this manually (see Step 5)** |
| 4 | Load MedGemma 4B-IT |
| 5 | Load SigLIP |
| 6 | Load Whisper |
| 7 | Load sentence embedder |
| 8 | RAG knowledge base |
| 9 | Whisper STT module |
| 10 | Confirmation layer |
| 11 | Translation module |
| 12 | SigLIP image encoding |
| 13 | Agentic tool definitions |
| 14 | MedGemma synthesis pipeline |
| 15 | Safety guardrails |
| 16 | Full pipeline orchestrator |
| 17 | Demo scenarios (run to test) |
| 18 | Gradio UI (launches public URL) |
| 19 | Markdown — system summary |

---

## 7. Expected Runtime

| Step | Expected Time |
|---|---|
| Cell 2 — Install dependencies | 2–4 minutes |
| Cell 4 — Load MedGemma (first run) | 25–30 minutes (8.6GB download) |
| Cell 5 — Load SigLIP | 2–3 minutes |
| Cell 6 — Load Whisper | 1–2 minutes |
| Cell 8 — Build RAG index | 3–5 minutes |
| Cell 17 — Run 3 demo scenarios | 5–8 minutes |
| Cell 18 — Launch Gradio | < 1 minute |

**Total first run: ~40–55 minutes**
**Subsequent runs (cached): ~10–15 minutes**

---

## 8. Gradio Public URL

When Cell 18 runs successfully you will see:

```
🚀 Launching Gradio...
Running on public URL: https://xxxxxxxxxxxx.gradio.live
```

This URL is public and shareable for 72 hours. Include it in your Kaggle submission.

---

## 9. Common Errors & Fixes

| Error | Cause | Fix |
|---|---|---|
| `GatedRepoError: 401` | License not accepted or wrong token type | Accept license at HF. Use Classic Read token, not fine-grained. |
| `GatedRepoError: 403` | Fine-grained token missing gated repo permission | Create a new Classic token instead |
| `KeyError: added_tokens` | Wrong model path — Keras format loaded | Use `"google/medgemma-4b-it"` string, not local path |
| `AttributeError: shape` | BatchEncoding returned instead of tensor | Check `call_medgemma()` has `if hasattr(input_ids, "input_ids")` fix |
| `SyntaxWarning: invalid escape sequence` | pydub Python 3.12 compatibility | Harmless — ignore and continue |
| Prompt truncated warning | Synthesis prompt too long | Expected — model still generates correctly |
| Output in English despite Hindi selected | Wrong variable printed in verbose mode | In `run_full_pipeline`, print `final_pack_local` not `final_pack_en` |

---

## 10. Testing the Gradio UI

Use these example inputs to test all three paths:

**Voice Note tab:**
- Record yourself saying: *"Baccha teen din se bukhaar mein hai"* (Hindi)
- Language: Hindi → Hindi
- Region: Rajasthan

**Type Notes tab:**
- Text: `The woman's hand got burned while cooking. Redness and blistering on the palm.`
- Language: English → Hindi
- Region: Maharashtra
- Helper notes: `Mustard oil was applied. Family has Ayushman Bharat card.`

**Image tab:**
- Upload any medical image (wound photo, X-ray, prescription)
- The system will classify it using SigLIP and include the context

---

*For production deployment options, see [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)*
