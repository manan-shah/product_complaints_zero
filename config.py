"""Configuration constants for the Asda SKU feedback classifier.

Edit these freely — no UI needed.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()  # read .env so HF_TOKEN is available at import time

# Ground-truth score->bucket thresholds. Positive = 8-10, Neutral = 6-7,
# Negative = 0-5. Change the boundaries here to re-map the ground truth.
POSITIVE_MIN = 8  # score >= POSITIVE_MIN            -> Positive
NEUTRAL_MIN = 6   # NEUTRAL_MIN <= score < POSITIVE_MIN -> Neutral
#                    score < NEUTRAL_MIN              -> Negative

# Default label sets (also editable live in the UI).
DEFAULT_SENTIMENT_LABELS = "Positive, Negative, Neutral, Mixed"
DEFAULT_THEME_LABELS = (
    "Taste/Flavour, Texture, Value/Price, Packaging, Availability, "
    "Quality, Freshness, Portion Size, Findability, Competitor Comparison"
)

# Statuses that mean "no genuine review to classify" -> excluded from modelling,
# kept only in the Availability summary. Matched case-insensitively as substrings.
EXCLUDE_STATUS_PATTERNS = [
    "out of stock",
    "out-of-stock",
    "oos",
    "doesn't stock",
    "does not stock",
    "not stock",
    "not tested",
    "untested",
    "n/a",
    "not applicable",
]

# Model registries (swappable via dropdown).
ZERO_SHOT_MODELS = [
    "facebook/bart-large-mnli",                          # default
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
]
# Fast local sentiment-only fallback (3-class). Handy when the big NLI model is
# too slow on CPU and you only need sentiment.
SENTIMENT_FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

ONE_SHOT_MODELS = [
    "Qwen/Qwen2.5-1.5B-Instruct",     # default: small + cheap
    "mistralai/Mistral-7B-Instruct-v0.3",
]

ONE_SHOT_SAMPLE_DEFAULT = 100
ONE_SHOT_SAMPLE_MAX = 300

# Asda-specific stopwords stacked on top of the English list for word clouds.
ASDA_STOPWORDS = {"asda", "store", "product", "bought", "item", "buy", "shop"}

# Column auto-detection hints (substring match on lower-cased header).
TEXT_COL_HINTS = ["review", "comment", "feedback", "verbatim", "text", "note"]
SCORE_COL_HINTS = ["nps", "likelihood", "score", "rating", "recommend", "0-10", "0 10"]
STATUS_COL_HINTS = ["status", "availability", "available", "stock", "tested"]
