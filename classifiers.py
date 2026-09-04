"""Zero-shot (local transformers) and one-shot (HF Inference API) classifiers."""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import gradio as gr
import pandas as pd

from config import SENTIMENT_FALLBACK_MODEL


# ---------------------------------------------------------------------------
# Local pipeline caching (so switching models back is instant)
# ---------------------------------------------------------------------------

_PIPELINE_CACHE: Dict[str, object] = {}


def get_zero_shot_pipeline(model_name: str):
    """Lazily build + cache a zero-shot-classification pipeline (CPU)."""
    key = f"zsc::{model_name}"
    if key not in _PIPELINE_CACHE:
        from transformers import pipeline  # imported lazily to speed app start

        _PIPELINE_CACHE[key] = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=-1,  # force CPU
        )
    return _PIPELINE_CACHE[key]


def get_sentiment_pipeline():
    """Lazily build + cache the fast 3-class sentiment pipeline (CPU)."""
    key = f"sent::{SENTIMENT_FALLBACK_MODEL}"
    if key not in _PIPELINE_CACHE:
        from transformers import pipeline

        _PIPELINE_CACHE[key] = pipeline(
            "sentiment-analysis",
            model=SENTIMENT_FALLBACK_MODEL,
            device=-1,
        )
    return _PIPELINE_CACHE[key]


# Map the roberta sentiment model's raw labels to our sentiment vocabulary.
_ROBERTA_LABEL_MAP = {
    "negative": "Negative",
    "neutral": "Neutral",
    "positive": "Positive",
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
}


# ---------------------------------------------------------------------------
# Zero-shot classification (local)
# ---------------------------------------------------------------------------

def classify_zero_shot(
    texts: List[str],
    sentiment_labels: List[str],
    theme_labels: List[str],
    model_name: str,
    use_fast_sentiment: bool,
    progress: Optional[gr.Progress] = None,
) -> pd.DataFrame:
    """Classify each text for sentiment + theme.

    If `use_fast_sentiment` is set, sentiment comes from the fast roberta model
    and only the theme axis uses the (slower) NLI zero-shot pipeline.
    """
    theme_pipe = get_zero_shot_pipeline(model_name)
    sent_pipe = get_zero_shot_pipeline(model_name)
    fast_pipe = get_sentiment_pipeline() if use_fast_sentiment else None

    rows = []
    n = len(texts)
    for i, text in enumerate(texts):
        if progress is not None:
            progress((i + 1) / max(n, 1), desc=f"Zero-shot {i + 1}/{n}")

        text = (text or "").strip()

        # --- sentiment ---
        if fast_pipe is not None:
            out = fast_pipe(text[:512])[0]
            sent = _ROBERTA_LABEL_MAP.get(out["label"].lower(), out["label"])
            sent_conf = float(out["score"])
        else:
            res = sent_pipe(text, sentiment_labels, multi_label=False)
            sent = res["labels"][0]
            sent_conf = float(res["scores"][0])

        # --- theme ---
        tres = theme_pipe(text, theme_labels, multi_label=False)
        theme = tres["labels"][0]
        theme_conf = float(tres["scores"][0])

        rows.append(
            {
                "predicted_sentiment": sent,
                "sentiment_confidence": round(sent_conf, 4),
                "predicted_theme": theme,
                "theme_confidence": round(theme_conf, 4),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One-shot classification (HF Inference API)
# ---------------------------------------------------------------------------

def _one_shot_prompt(review: str, labels: List[str], axis: str) -> List[dict]:
    """Build a chat-style 1-shot prompt: instruction + ONE example + the review."""
    label_str = ", ".join(labels)
    # A single hand-written labelled example (the "1-shot" demonstration).
    if axis == "sentiment":
        example_review = "The pizza base was soggy and overpriced for what you get."
        example_label = "Negative"
    else:  # theme
        example_review = "Loved the flavour but the box was crushed on the shelf."
        example_label = "Packaging"

    system = (
        f"You are a strict classifier. Classify the customer review into exactly "
        f"one of these {axis} labels: {label_str}. "
        f"Reply with ONLY the label text, nothing else."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f'Review: "{example_review}"\nLabel:'},
        {"role": "assistant", "content": example_label},
        {"role": "user", "content": f'Review: "{review}"\nLabel:'},
    ]


def _parse_label(raw: str, labels: List[str]) -> str:
    """Best-effort match of the model's free-text answer to a known label."""
    if not raw:
        return labels[0]
    cleaned = raw.strip().strip('"').strip(".").lower()
    for lab in labels:
        if cleaned == lab.lower():
            return lab
    for lab in labels:
        if lab.lower() in cleaned or cleaned in lab.lower():
            return lab
    return labels[0]  # fall back to first label if unparseable


def _hf_chat_with_retry(client, model, messages, max_retries=5):
    """Call the Inference API with exponential backoff for 429s / cold starts."""
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat_completion(
                messages=messages, model=model, max_tokens=12, temperature=0.0
            )
            return resp.choices[0].message.content
        except Exception as err:  # noqa: BLE001 — surface after retries
            last_err = err
            msg = str(err).lower()
            if any(k in msg for k in ("429", "rate", "loading", "503", "timeout")):
                time.sleep(delay)
                delay = min(delay * 2, 30)
                continue
            raise
    raise RuntimeError(f"HF Inference API failed after {max_retries} retries: {last_err}")


def classify_one_shot(
    texts: List[str],
    sentiment_labels: List[str],
    theme_labels: List[str],
    model_name: str,
    progress: Optional[gr.Progress] = None,
) -> pd.DataFrame:
    """Classify a (capped) list of texts via the HF Inference API, 1-shot."""
    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error(
            "HF_TOKEN is not set. Add it to your environment or .env file to use "
            "the one-shot method (see .env.example)."
        )

    from huggingface_hub import InferenceClient

    client = InferenceClient(token=token)

    rows = []
    n = len(texts)
    for i, text in enumerate(texts):
        if progress is not None:
            progress((i + 1) / max(n, 1), desc=f"One-shot (API) {i + 1}/{n}")
        text = (text or "").strip()

        sent_raw = _hf_chat_with_retry(
            client, model_name, _one_shot_prompt(text, sentiment_labels, "sentiment")
        )
        theme_raw = _hf_chat_with_retry(
            client, model_name, _one_shot_prompt(text, theme_labels, "theme")
        )
        rows.append(
            {
                "predicted_sentiment": _parse_label(sent_raw, sentiment_labels),
                "sentiment_confidence": 1.0,  # API returns no probability
                "predicted_theme": _parse_label(theme_raw, theme_labels),
                "theme_confidence": 1.0,
            }
        )
    return pd.DataFrame(rows)
