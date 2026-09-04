"""Gradio UI: Blocks layout and event callbacks."""

from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from analysis import (
    build_benchmark,
    build_charts,
    build_wordcloud,
    refresh_wordcloud_choices,
)
from classifiers import classify_one_shot, classify_zero_shot
from config import (
    DEFAULT_SENTIMENT_LABELS,
    DEFAULT_THEME_LABELS,
    ONE_SHOT_MODELS,
    ONE_SHOT_SAMPLE_DEFAULT,
    ONE_SHOT_SAMPLE_MAX,
    SCORE_COL_HINTS,
    STATUS_COL_HINTS,
    TEXT_COL_HINTS,
    ZERO_SHOT_MODELS,
)
from data import (
    clean_dataframe,
    format_cleaning_summary,
    guess_column,
    load_dataframe,
    parse_sentiment_labels,
    parse_theme_labels,
    score_to_bucket,
)


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

def on_load(file_obj):
    """Load a file, auto-detect columns, and produce the cleaning preview."""
    if file_obj is None:
        raise gr.Error("Please upload an .xlsx or .csv file first.")
    path = file_obj if isinstance(file_obj, str) else file_obj.name
    df = load_dataframe(path)
    cols = list(df.columns)

    text_col = guess_column(cols, TEXT_COL_HINTS) or (cols[0] if cols else None)
    score_col = guess_column(cols, SCORE_COL_HINTS)
    status_col = guess_column(cols, STATUS_COL_HINTS)

    kept, summary = clean_dataframe(df, text_col, status_col)
    summary_md = format_cleaning_summary(summary)

    state = {
        "raw": df,
        "text_col": text_col,
        "score_col": score_col,
        "status_col": status_col,
        "results": None,
        "results_meta": None,
    }
    return (
        state,
        gr.update(choices=cols, value=text_col),
        gr.update(choices=cols, value=score_col),
        gr.update(choices=cols, value=status_col),
        summary_md,
        kept.head(50),
    )


def on_recompute_cleaning(state, text_col, score_col, status_col):
    """Re-run cleaning when the user overrides column selections."""
    if not state or state.get("raw") is None:
        raise gr.Error("Load a file first.")
    df = state["raw"]
    kept, summary = clean_dataframe(df, text_col, status_col)
    state = dict(state)
    state.update(text_col=text_col, score_col=score_col, status_col=status_col,
                 results=None, results_meta=None)
    return state, format_cleaning_summary(summary), kept.head(50)


def run_classification(
    state,
    method,
    zero_shot_model,
    one_shot_model,
    use_fast_sentiment,
    sample_size,
    sentiment_raw,
    theme_raw,
    progress=gr.Progress(),
):
    """Main entry point for the Classify tab."""
    if not state or state.get("raw") is None:
        raise gr.Error("Load a file in the Data tab first.")

    text_col = state["text_col"]
    score_col = state.get("score_col")
    status_col = state.get("status_col")

    sentiment_labels = parse_sentiment_labels(sentiment_raw)
    theme_labels = parse_theme_labels(theme_raw)

    kept, _ = clean_dataframe(state["raw"], text_col, status_col)
    if kept.empty:
        raise gr.Error("No rows left after cleaning — check your column choices.")

    if score_col and score_col in kept.columns:
        kept = kept.assign(ground_truth=kept[score_col].apply(score_to_bucket))
    else:
        kept = kept.assign(ground_truth=None)

    if method == "one-shot (API sample)":
        n = int(min(sample_size, len(kept)))
        sample = kept.head(n).reset_index(drop=True)
        texts = sample[text_col].astype(str).tolist()
        preds = classify_one_shot(
            texts, sentiment_labels, theme_labels, one_shot_model, progress
        )
        used = sample
    else:  # zero-shot local, full dataset
        sample = kept.reset_index(drop=True)
        texts = sample[text_col].astype(str).tolist()
        preds = classify_zero_shot(
            texts, sentiment_labels, theme_labels, zero_shot_model,
            use_fast_sentiment, progress,
        )
        used = sample

    result = pd.concat(
        [used.reset_index(drop=True), preds.reset_index(drop=True)], axis=1
    )

    # Cache results so Benchmark / Word Cloud / Charts never re-run models.
    state = dict(state)
    state["results"] = result
    state["results_meta"] = {
        "method": method,
        "model": one_shot_model if method.startswith("one-shot") else zero_shot_model,
        "sentiment_labels": sentiment_labels,
        "theme_labels": theme_labels,
    }

    display_cols = [text_col]
    if "ground_truth" in result:
        display_cols.append("ground_truth")
    display_cols += [
        "predicted_sentiment", "sentiment_confidence",
        "predicted_theme", "theme_confidence",
    ]
    display = result[[c for c in display_cols if c in result.columns]]

    note = (
        f"Classified **{len(result)}** rows using **{method}** "
        f"({state['results_meta']['model']})."
    )
    if method.startswith("one-shot"):
        note += "  \n⚠️ One-shot uses the free HF Inference API — rate-limited; keep samples small."

    return state, display, note


def download_results(state):
    """Write the cached results to a CSV and return the path for download."""
    if not state or state.get("results") is None:
        raise gr.Error("Run a classification first.")
    path = os.path.join(os.getcwd(), "results_export.csv")
    state["results"].to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Asda SKU Feedback Classifier", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# Asda SKU Feedback Classifier\n"
            "Zero-shot (local, full dataset) & one-shot (HF Inference API, sampled) "
            "classification of colleague-testing reviews. CPU-only."
        )
        state = gr.State(value=None)

        # ---------------- Data tab ----------------
        with gr.Tab("Data"):
            file_in = gr.File(
                label="Upload SKU scores (.xlsx or .csv)",
                file_types=[".xlsx", ".xls", ".csv"],
                type="filepath",
            )
            load_btn = gr.Button("Load file", variant="primary")
            with gr.Row():
                text_dd = gr.Dropdown(label="Text (review) column", interactive=True)
                score_dd = gr.Dropdown(label="Score (0-10) column", interactive=True)
                status_dd = gr.Dropdown(label="Status/availability column", interactive=True)
            recompute_btn = gr.Button("Apply column choices")
            clean_md = gr.Markdown()
            preview_df = gr.Dataframe(label="Cleaned preview (first 50 kept rows)",
                                      interactive=False, wrap=True)

        # ---------------- Classify tab ----------------
        with gr.Tab("Classify"):
            with gr.Row():
                method_dd = gr.Radio(
                    ["zero-shot (local, full set)", "one-shot (API sample)"],
                    value="zero-shot (local, full set)",
                    label="Method",
                )
            with gr.Row():
                zs_model_dd = gr.Dropdown(
                    ZERO_SHOT_MODELS, value=ZERO_SHOT_MODELS[0],
                    label="Zero-shot model (local)",
                )
                os_model_dd = gr.Dropdown(
                    ONE_SHOT_MODELS, value=ONE_SHOT_MODELS[0],
                    label="One-shot model (API)",
                )
            fast_sent_cb = gr.Checkbox(
                value=False,
                label="Use fast local sentiment model for the sentiment axis "
                      "(cardiffnlp roberta) — speeds up CPU zero-shot",
            )
            with gr.Row():
                sentiment_tb = gr.Textbox(
                    DEFAULT_SENTIMENT_LABELS, label="Sentiment labels (comma-separated)"
                )
                theme_tb = gr.Textbox(
                    DEFAULT_THEME_LABELS, label="Theme labels (comma-separated)"
                )
            sample_slider = gr.Slider(
                10, ONE_SHOT_SAMPLE_MAX, value=ONE_SHOT_SAMPLE_DEFAULT, step=10,
                label="One-shot sample size (API rows — rate-limited free tier)",
            )
            run_btn = gr.Button("Run classification", variant="primary")
            run_note = gr.Markdown()
            results_df = gr.Dataframe(label="Results", interactive=False, wrap=True)
            download_btn = gr.Button("Download results CSV")
            download_file = gr.File(label="results_export.csv")

        # ---------------- Benchmark tab ----------------
        with gr.Tab("Benchmark"):
            gr.Markdown(
                "Compares zero-shot vs one-shot **sentiment** predictions against "
                "the score-derived ground-truth buckets on the same sample."
            )
            bench_btn = gr.Button("Run benchmark", variant="primary")
            bench_metrics = gr.Dataframe(label="Metrics", interactive=False)
            bench_bar = gr.Plot(label="Accuracy / F1")
            bench_cm = gr.Plot(label="Confusion matrices")

        # ---------------- Word Cloud tab ----------------
        with gr.Tab("Word Cloud"):
            wc_basis = gr.Dropdown(["Overall"], value="Overall", label="Basis / filter")
            with gr.Row():
                wc_refresh = gr.Button("Refresh filter options")
                wc_btn = gr.Button("Generate word cloud", variant="primary")
            wc_plot = gr.Plot(label="Word cloud")

        # ---------------- Charts tab ----------------
        with gr.Tab("Charts"):
            charts_btn = gr.Button("Build charts", variant="primary")
            chart_sent = gr.Plot(label="Sentiment counts")
            chart_theme = gr.Plot(label="Theme counts")
            chart_cat = gr.Plot(label="Avg score by category")

        # ---------------- wiring ----------------
        load_btn.click(
            on_load, inputs=[file_in],
            outputs=[state, text_dd, score_dd, status_dd, clean_md, preview_df],
        )
        recompute_btn.click(
            on_recompute_cleaning,
            inputs=[state, text_dd, score_dd, status_dd],
            outputs=[state, clean_md, preview_df],
        )
        run_btn.click(
            run_classification,
            inputs=[state, method_dd, zs_model_dd, os_model_dd, fast_sent_cb,
                    sample_slider, sentiment_tb, theme_tb],
            outputs=[state, results_df, run_note],
        )
        download_btn.click(download_results, inputs=[state], outputs=[download_file])

        bench_btn.click(
            build_benchmark, inputs=[state],
            outputs=[bench_metrics, bench_bar, bench_cm],
        )

        wc_refresh.click(refresh_wordcloud_choices, inputs=[state], outputs=[wc_basis])
        wc_btn.click(build_wordcloud, inputs=[state, wc_basis], outputs=[wc_plot])

        charts_btn.click(
            build_charts, inputs=[state],
            outputs=[chart_sent, chart_theme, chart_cat],
        )

    return demo
