from pathlib import Path

import numpy as np
import pandas as pd

def read_table(path: str) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    if suffix == ".parquet":
        try:
            return pd.read_parquet(file_path)
        except ImportError as exc:
            raise ImportError(
                "Reading parquet requires an installed backend such as pyarrow."
            ) from exc

    raise ValueError(f"Unsupported table format: {file_path.suffix}")

def load_interactions(path: str, user_column: str, item_column: str) -> pd.DataFrame:
    interactions_table = read_table(path)
    missing = [
        column for column in [user_column, item_column] if column not in interactions_table.columns
    ]
    if missing:
        raise ValueError(f"Missing columns in interactions table: {missing}")

    interactions = interactions_table[[user_column, item_column]].dropna().copy()
    interactions[user_column] = interactions[user_column].astype(str)
    interactions[item_column] = interactions[item_column].astype(str)
    interactions = interactions.drop_duplicates()
    return interactions

def load_embeddings(path: str, item_column: str) -> pd.DataFrame:
    file_path = Path(path)
    suffix = file_path.suffix.lower()

    if suffix in {".csv", ".tsv", ".parquet"}:
        embeddings_table = read_table(path)
        if item_column not in embeddings_table.columns:
            raise ValueError(
                f"Embedding file {path} must contain the item column '{item_column}'."
            )
        feature_columns = [
            column for column in embeddings_table.columns if column != item_column
        ]
        if not feature_columns:
            raise ValueError(f"Embedding file {path} has no numeric feature columns.")
        prepared_embeddings = embeddings_table[[item_column] + feature_columns].copy()
        prepared_embeddings[item_column] = prepared_embeddings[item_column].astype(str)
        return prepared_embeddings

    if suffix == ".npy":
        raise ValueError(
            "Plain .npy is ambiguous. Use a table file or an .npz archive with "
            "'item_ids' and 'embeddings'."
        )

    if suffix == ".npz":
        archive = np.load(file_path)
        if "item_ids" not in archive or "embeddings" not in archive:
            raise ValueError(
                "NPZ embeddings must contain arrays named 'item_ids' and 'embeddings'."
            )

        item_ids = archive["item_ids"].astype(str)
        embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings array must be 2-dimensional.")
        if embeddings.shape[0] != item_ids.shape[0]:
            raise ValueError("item_ids and embeddings must have matching first dimension.")

        feature_columns = [f"dim_{index}" for index in range(embeddings.shape[1])]
        embeddings_table = pd.DataFrame(embeddings, columns=feature_columns)
        embeddings_table.insert(0, item_column, item_ids)
        return embeddings_table

    raise ValueError(f"Unsupported embedding format: {file_path.suffix}")
