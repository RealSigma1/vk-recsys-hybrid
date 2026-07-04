from __future__ import annotations

import pandas as pd


def filter_interactions(
    interactions: pd.DataFrame,
    user_column: str,
    item_column: str,
    min_user_interactions: int,
    min_item_interactions: int,
) -> pd.DataFrame:
    filtered = interactions.copy()

    while True:
        before = len(filtered)

        user_counts = filtered[user_column].value_counts()
        kept_users = user_counts[user_counts >= min_user_interactions].index
        filtered = filtered[filtered[user_column].isin(kept_users)]

        item_counts = filtered[item_column].value_counts()
        kept_items = item_counts[item_counts >= min_item_interactions].index
        filtered = filtered[filtered[item_column].isin(kept_items)]

        if len(filtered) == before:
            break

    return filtered.reset_index(drop=True)
