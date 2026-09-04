"""Data loading, cleaning, ground-truth bucketing, and small text helpers."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from config import (
    DEFAULT_SENTIMENT_LABELS,
    DEFAULT_THEME_LABELS,
    EXCLUDE_STATUS_PATTERNS,
    NEUTRAL_MIN,
    POSITIVE_MIN,
)


# ---------------------------------------------------------------------------
# Ground-truth bucketing
# ---------------------------------------------------------------------------

def score_to_bucket(score) -> Optional[str]:
    """Map a 0-10 score to a ground-truth sentiment bucket. None if unparseable."""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s >= POSITIVE_MIN:
        return "Positive"
    if s >= NEUTRAL_MIN:
        return "Neutral"
    return "Negative"


# ---------------------------------------------------------------------------
# Column detection + IO
# ---------------------------------------------------------------------------

def guess_column(columns: List[str], hints: List[str]) -> Optional[str]:
    """Return the first column whose lower-cased name contains any hint."""
    lowered = {c: str(c).lower() for c in columns}
    for hint in hints:
        for col, low in lowered.items():
            if hint in low:
                return col
    return None


def load_dataframe(file_path: str) -> pd.DataFrame:
    """Read an uploaded .xlsx or .csv into a DataFrame."""
    if file_path is None:
        raise ValueError("No file uploaded.")
    lower = file_path.lower()
    if lower.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(file_path, engine="openpyxl")
    if lower.endswith(".csv"):
        return pd.read_csv(file_path)
    raise ValueError("Unsupported file type — please upload .xlsx or .csv.")


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def _is_excluded_status(value) -> bool:
    """True if a status value indicates OOS / not-stocked / not-tested."""
    if value is None:
        return False
    text = str(value).strip().lower()
    if not text:
        return False
    return any(pat in text for pat in EXCLUDE_STATUS_PATTERNS)


def clean_dataframe(
    df: pd.DataFrame,
    text_col: str,
    status_col: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Split the raw frame into (kept, summary).

    Rows are excluded when the status is OOS / not-stocked / not-tested, or when
    the comment text is empty. Excluded rows are only tallied for the
    Availability summary; they are not modelled.
    """
    total = len(df)
    work = df.copy()

    if status_col and status_col in work.columns:
        status_excluded_mask = work[status_col].apply(_is_excluded_status)
    else:
        status_excluded_mask = pd.Series(False, index=work.index)

    text_series = work[text_col].astype("string").fillna("").str.strip()
    empty_mask = text_series.eq("") | text_series.str.lower().eq("nan")

    keep_mask = (~status_excluded_mask) & (~empty_mask)
    kept = work.loc[keep_mask].copy()

    availability_counts: Dict[str, int] = {}
    if status_col and status_col in work.columns:
        excluded_statuses = work.loc[status_excluded_mask, status_col].astype(str)
        availability_counts = (
            excluded_statuses.str.strip().value_counts().to_dict()
        )

    summary = {
        "total_rows": total,
        "kept_rows": int(keep_mask.sum()),
        "excluded_status": int(status_excluded_mask.sum()),
        "excluded_empty": int((empty_mask & ~status_excluded_mask).sum()),
        "availability_counts": availability_counts,
    }
    return kept, summary


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def parse_labels(raw: str, fallback: str) -> List[str]:
    """Split a comma-separated label textbox into a clean list."""
    labels = [x.strip() for x in (raw or "").split(",") if x.strip()]
    if not labels:
        labels = [x.strip() for x in fallback.split(",") if x.strip()]
    return labels


def parse_sentiment_labels(raw: str) -> List[str]:
    return parse_labels(raw, DEFAULT_SENTIMENT_LABELS)


def parse_theme_labels(raw: str) -> List[str]:
    return parse_labels(raw, DEFAULT_THEME_LABELS)


def format_cleaning_summary(summary: Dict) -> str:
    """Markdown summary of the cleaning step."""
    lines = [
        f"**Total rows:** {summary['total_rows']}",
        f"**Kept for modelling:** {summary['kept_rows']}",
        f"**Excluded — availability/not-tested:** {summary['excluded_status']}",
        f"**Excluded — empty comment:** {summary['excluded_empty']}",
        "",
        "**Availability breakdown (excluded rows):**",
    ]
    counts = summary.get("availability_counts") or {}
    if counts:
        for status, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {status}: {count}")
    else:
        lines.append("- (no status column detected / none excluded)")
    return "\n".join(lines)
