"""Benchmark comparison and visualizations (word cloud + charts)."""

from __future__ import annotations

from typing import List

import gradio as gr
import matplotlib

matplotlib.use("Agg")  # headless / CPU-only backend
import matplotlib.pyplot as plt
import pandas as pd

from classifiers import classify_one_shot, classify_zero_shot
from config import ASDA_STOPWORDS, ONE_SHOT_MODELS, ZERO_SHOT_MODELS
from data import guess_column


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def build_benchmark(state, progress=gr.Progress()):
    """Compare zero-shot vs one-shot on the SAME sample against ground-truth buckets.

    Uses cached zero-shot predictions where possible and always runs one-shot on
    the sample (API). Requires a score column for ground truth.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    if not state or state.get("results") is None:
        raise gr.Error("Run a classification in the Classify tab first.")

    meta = state["results_meta"]
    result = state["results"]
    text_col = state["text_col"]
    sentiment_labels = meta["sentiment_labels"]
    theme_labels = meta["theme_labels"]

    if "ground_truth" not in result or result["ground_truth"].isna().all():
        raise gr.Error("No ground-truth buckets — select a valid 0-10 score column.")

    sample = result.dropna(subset=["ground_truth"]).reset_index(drop=True)
    texts = sample[text_col].astype(str).tolist()
    gt = sample["ground_truth"].tolist()

    if meta["method"].startswith("zero-shot"):
        zs_pred = sample["predicted_sentiment"].tolist()
    else:
        progress(0.1, desc="Zero-shot on benchmark sample")
        zs_df = classify_zero_shot(
            texts, sentiment_labels, theme_labels,
            ZERO_SHOT_MODELS[0], use_fast_sentiment=False, progress=progress,
        )
        zs_pred = zs_df["predicted_sentiment"].tolist()

    if meta["method"].startswith("one-shot"):
        os_pred = sample["predicted_sentiment"].tolist()
    else:
        progress(0.5, desc="One-shot (API) on benchmark sample")
        os_df = classify_one_shot(
            texts, sentiment_labels, theme_labels, ONE_SHOT_MODELS[0], progress
        )
        os_pred = os_df["predicted_sentiment"].tolist()

    gt_classes = ["Positive", "Neutral", "Negative"]

    def _score(pred):
        acc = accuracy_score(gt, pred)
        f1 = f1_score(gt, pred, labels=gt_classes, average="macro", zero_division=0)
        return acc, f1

    zs_acc, zs_f1 = _score(zs_pred)
    os_acc, os_f1 = _score(os_pred)

    metrics_df = pd.DataFrame(
        {
            "Method": ["Zero-shot (local)", "One-shot (API)"],
            "Accuracy": [round(zs_acc, 3), round(os_acc, 3)],
            "Macro F1": [round(zs_f1, 3), round(os_f1, 3)],
            "Sample size": [len(gt), len(gt)],
        }
    )

    # --- bar chart ---
    fig_bar, ax = plt.subplots(figsize=(6, 4))
    x = range(len(metrics_df))
    ax.bar([i - 0.2 for i in x], metrics_df["Accuracy"], width=0.4, label="Accuracy")
    ax.bar([i + 0.2 for i in x], metrics_df["Macro F1"], width=0.4, label="Macro F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics_df["Method"])
    ax.set_ylim(0, 1)
    ax.set_title("Zero-shot vs One-shot (sentiment vs ground truth)")
    ax.legend()
    fig_bar.tight_layout()

    # --- confusion matrices (side by side) ---
    fig_cm, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax_cm, pred, title in [
        (axes[0], zs_pred, "Zero-shot"),
        (axes[1], os_pred, "One-shot"),
    ]:
        cm = confusion_matrix(gt, pred, labels=gt_classes)
        ax_cm.imshow(cm, cmap="Blues")
        ax_cm.set_xticks(range(len(gt_classes)))
        ax_cm.set_yticks(range(len(gt_classes)))
        ax_cm.set_xticklabels(gt_classes, rotation=45, ha="right")
        ax_cm.set_yticklabels(gt_classes)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        ax_cm.set_title(title)
        for r in range(len(gt_classes)):
            for c in range(len(gt_classes)):
                ax_cm.text(c, r, cm[r, c], ha="center", va="center",
                           color="black", fontsize=9)
    fig_cm.tight_layout()

    return metrics_df, fig_bar, fig_cm


# ---------------------------------------------------------------------------
# Word cloud
# ---------------------------------------------------------------------------

def _wordcloud_basis_choices(state) -> List[str]:
    """Available filters for the word cloud, derived from cached results."""
    choices = ["Overall"]
    if not state or state.get("results") is None:
        return choices
    result = state["results"]
    for col, prefix in [("predicted_sentiment", "Sentiment"),
                        ("predicted_theme", "Theme")]:
        if col in result.columns:
            for val in sorted(result[col].dropna().unique()):
                choices.append(f"{prefix}: {val}")
    return choices


def refresh_wordcloud_choices(state):
    return gr.update(choices=_wordcloud_basis_choices(state), value="Overall")


def build_wordcloud(state, basis):
    """Generate a word cloud from rows matching the chosen basis."""
    from wordcloud import STOPWORDS, WordCloud

    if not state or state.get("results") is None:
        raise gr.Error("Run a classification first (word clouds use the reviews).")

    result = state["results"]
    text_col = state["text_col"]

    if basis and basis != "Overall" and ":" in basis:
        prefix, value = [p.strip() for p in basis.split(":", 1)]
        col = "predicted_sentiment" if prefix.lower() == "sentiment" else "predicted_theme"
        subset = result[result[col] == value]
    else:
        subset = result

    text = " ".join(subset[text_col].astype(str).tolist()).strip()
    if not text:
        raise gr.Error("No text matches that filter.")

    stopwords = set(STOPWORDS) | ASDA_STOPWORDS
    wc = WordCloud(
        width=900, height=450, background_color="white",
        stopwords=stopwords, collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word cloud — {basis} ({len(subset)} reviews)")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def build_charts(state):
    """Counts per sentiment, per theme, and avg score by category."""
    if not state or state.get("results") is None:
        raise gr.Error("Run a classification first.")
    result = state["results"]
    score_col = state.get("score_col")

    fig_sent, ax1 = plt.subplots(figsize=(6, 4))
    result["predicted_sentiment"].value_counts().plot(kind="bar", ax=ax1, color="#4C72B0")
    ax1.set_title("Reviews per predicted sentiment")
    ax1.set_ylabel("Count")
    fig_sent.tight_layout()

    fig_theme, ax2 = plt.subplots(figsize=(7, 4))
    result["predicted_theme"].value_counts().plot(kind="bar", ax=ax2, color="#55A868")
    ax2.set_title("Reviews per predicted theme")
    ax2.set_ylabel("Count")
    fig_theme.tight_layout()

    fig_cat, ax3 = plt.subplots(figsize=(7, 4))
    cat_col = guess_column(list(result.columns), ["category", "department", "dept", "aisle"])
    if cat_col and score_col and score_col in result.columns:
        tmp = result.copy()
        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
        avg = tmp.groupby(cat_col)[score_col].mean().sort_values(ascending=False).head(15)
        avg.plot(kind="barh", ax=ax3, color="#C44E52")
        ax3.set_title(f"Avg score by {cat_col}")
        ax3.set_xlabel("Avg 0-10 score")
    else:
        ax3.text(0.5, 0.5, "No category + score columns detected",
                 ha="center", va="center")
        ax3.axis("off")
    fig_cat.tight_layout()

    return fig_sent, fig_theme, fig_cat
