"""
src/workshop_utils.py — Shared Notebook Utilities
==================================================
Lightweight helpers shared across instructor notebooks.
Import these instead of repeating boilerplate in every module.
"""

import pandas as pd

from config import MICRO_SUBSET_N


def get_micro_subset(panel: pd.DataFrame, n: int = MICRO_SUBSET_N):
    """
    Return the top-N series by total sales volume and the filtered panel.

    Selection rule: rank all unique_ids by sum(y) descending, take the top N.
    This is deterministic given a fixed panel and n.

    Parameters
    ----------
    panel : Validated panel DataFrame with columns [unique_id, ds, y].
    n     : Number of series to select (default: MICRO_SUBSET_N from config).

    Returns
    -------
    micro      : Filtered panel containing only the top-N series.
    top_series : Index of unique_ids in volume-descending order.

    Usage
    -----
        from src.workshop_utils import get_micro_subset
        micro, top_series = get_micro_subset(panel, n=MICRO_SUBSET_N)
    """
    top_series = (
        panel.groupby("unique_id")["y"]
        .sum()
        .sort_values(ascending=False)
        .head(n)
        .index
    )
    micro = panel[panel["unique_id"].isin(top_series)].copy()
    return micro, top_series
