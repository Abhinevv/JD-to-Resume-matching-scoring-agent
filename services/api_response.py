"""Utilities for preparing API responses."""

from __future__ import annotations

import math
from typing import Any, Dict


def dataframe_to_records(df) -> list[Dict[str, Any]]:
    """Convert pandas dataframes to JSON-safe record lists."""
    if df is None:
        return []

    try:
        normalized = df.copy()
        if "itemsets" in normalized.columns:
            normalized["itemsets"] = normalized["itemsets"].apply(lambda value: sorted(list(value)))
        if "antecedents" in normalized.columns:
            normalized["antecedents"] = normalized["antecedents"].apply(lambda value: sorted(list(value)))
            normalized["consequents"] = normalized["consequents"].apply(lambda value: sorted(list(value)))
        return clean_json_value(normalized.to_dict(orient="records"))
    except Exception:
        return []


def clean_json_value(value: Any) -> Any:
    """Recursively replace non-JSON-safe values."""
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
