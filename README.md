# Asda SKU Feedback Classifier

A CPU-only [Gradio](https://gradio.app) app that classifies Asda SKU-level
colleague-testing feedback (free-text reviews) along two axes — **Sentiment**
and **Theme** — and benchmarks two approaches against score-derived ground truth.

- **Zero-shot (local)** — `transformers` `pipeline("zero-shot-classification")`.
  Runs entirely on CPU over the **full** cleaned dataset. This is the workhorse.
- **One-shot (remote)** — `huggingface_hub.InferenceClient` with a small instruct
  model, called with a 1-shot prompt on a **capped sample** (default 100 rows),
  with retry/backoff for rate limits and cold starts.

Ground-truth buckets are derived from the 0–10 score column
(`Positive = 8–10`, `Neutral = 6–7`, `Negative = 0–5` — editable in `app.py`)
and used to evaluate both methods.

---

## Features

| Tab | What it does |
|-----|--------------|
| **Data** | Upload `.xlsx`/`.csv`, auto-detect + override the text / score / status columns, and see the cleaning summary (kept vs excluded + availability counts). |
| **Classify** | Pick model + method, edit label sets live, set the one-shot sample size, run with a progress bar, view a results table with predictions, confidence, and ground-truth bucket. |
| **Benchmark** | Compare zero-shot vs one-shot accuracy / macro-F1 against the ground-truth buckets on the same sample — bar chart + confusion matrices. |
| **Word Cloud** | Generate a cloud from all reviews or a sentiment/theme-filtered subset; English + Asda-specific stopwords removed. |
| **Charts** | Counts per sentiment, per theme, and average score by category. |

Loaded local pipelines are **cached**, so switching models back is instant, and
classification results are cached so changing a word-cloud / benchmark filter
never re-runs the models.

---

## Setup

Python 3.10+ recommended.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install the CPU build of PyTorch first (avoids pulling CUDA wheels)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install everything else
pip install -r requirements.txt
```

### Hugging Face token (only needed for the one-shot method)

The one-shot method calls the HF Inference API and needs a token.
The zero-shot (local) method works **without** it.

```bash
cp .env.example .env
# then edit .env and set HF_TOKEN=hf_xxx
```

Create a read token at <https://huggingface.co/settings/tokens>.
The token is read from the `HF_TOKEN` environment variable and is **never**
hardcoded. If it is missing, the one-shot tab shows a clear error.

---

## Run

```bash
python app.py
```

Then open the forwarded port (default **7860**). In a GitHub Codespace the app
binds to `0.0.0.0`, so use the **Ports** tab to open the forwarded URL.

---

## Notes on CPU limits

- Zero-shot NLI models (e.g. `facebook/bart-large-mnli`) are the heaviest step.
  For faster sentiment on CPU, tick **"Use fast local sentiment model"** — the
  sentiment axis then uses `cardiffnlp/twitter-roberta-base-sentiment-latest`
  and only the theme axis uses the NLI model.
- Keep the one-shot sample small — the free HF Inference API is rate-limited and
  models can cold-start (handled with exponential backoff).

## Editing labels & thresholds

- **Labels**: edit the comma-separated boxes in the **Classify** tab — no code
  changes needed.
- **Score thresholds**: edit `POSITIVE_MIN` / `NEUTRAL_MIN` at the top of `app.py`.
- **Models**: edit `ZERO_SHOT_MODELS` / `ONE_SHOT_MODELS` in `app.py`.

## Data expectations

The input `.xlsx`/`.csv` should include columns for: product name, SKU ID,
category/department, a 0–10 likelihood/NPS score, an availability/status field,
and a free-text review. Columns are auto-detected by name and can be overridden
in the Data tab. Rows that are out-of-stock / not-stocked / not-tested or have an
empty comment are excluded from modelling and counted only in the Availability
summary.
