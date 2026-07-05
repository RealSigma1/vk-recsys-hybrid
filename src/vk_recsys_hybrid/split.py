import numpy as np
import pandas as pd

def random_user_holdout_split(
    interactions: pd.DataFrame,
    user_column: str,
    item_column: str,
    test_fraction: float,
    min_test_items: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")

    random_generator = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for user_id, user_interactions in interactions.groupby(user_column, sort=False):
        if len(user_interactions) < 2:
            continue

        indices = user_interactions.index.to_numpy(copy=True)
        random_generator.shuffle(indices)
        n_test = max(min_test_items, int(round(len(indices) * test_fraction)))
        n_test = min(n_test, len(indices) - 1)

        test_indices = indices[:n_test]
        train_indices = indices[n_test:]

        train_parts.append(interactions.loc[train_indices, [user_column, item_column]])
        test_parts.append(interactions.loc[test_indices, [user_column, item_column]])

    if not train_parts or not test_parts:
        raise ValueError("Split produced empty train or test data.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df
