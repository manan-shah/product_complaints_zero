""" app Slim entrypoint for the Asda SKU-level colleague-testing feedback classifier.

A CPU-only Gradio app that classifies free-text colleague reviews of Asda SKUs
along two axes (Sentiment + Theme) using two methods:

  * ZERO-SHOT (local): transformers `pipeline("zero-shot-classification")`.
    The workhorse; runs over the whole cleaned dataset on CPU.

  * ONE-SHOT  (remote): huggingface_hub `InferenceClient` with a small instruct
    model, called with a 1-shot prompt on a *capped* sample.

Modules:
  config.py       - constants and defaults
  data.py         - loading, cleaning, ground-truth bucketing, small helpers
  classifiers.py  - zero-shot and one-shot classification
  analysis.py     - benchmark, word cloud, and charts
  ui.py           - Gradio Blocks layout and event callbacks

Run:  python app.py
"""

from __future__ import annotations

import os

from ui import build_ui


if __name__ == "__main__":
    if not os.getenv("HF_TOKEN"):
        print("[warn] HF_TOKEN is not set — the one-shot (API) method will error "
              "until you set it. Zero-shot (local) works without it.")
    app = build_ui()
    # server_name=0.0.0.0 so Codespaces can forward the port.
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")))
