from pathlib import Path

import numpy as np
import pandas as pd

POSITIVE_SIGNAL_COLUMNS = [
    "like",
    "share",
    "bookmark",
    "click_on_author",
    "open_comments",
]

def load_many_parquets(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)

def prepare_positive_interactions(
    interactions: pd.DataFrame,
    items_metadata: pd.DataFrame,
    user_column: str,
    item_column: str,
    min_watch_ratio: float = 0.8,
) -> pd.DataFrame:
    metadata = items_metadata[[item_column, "duration"]].copy()
    metadata[item_column] = metadata[item_column].astype(np.uint32)

    merged = interactions.merge(metadata, on=item_column, how="left")
    duration = merged["duration"].fillna(1).clip(lower=1)
    watch_ratio = merged["timespent"].astype(np.float32) / duration.astype(np.float32)

    explicit_positive = np.zeros(len(merged), dtype=bool)
    for column in POSITIVE_SIGNAL_COLUMNS:
        explicit_positive |= merged[column].to_numpy(dtype=bool)

    positive_mask = ((watch_ratio >= min_watch_ratio) | explicit_positive) & (
        ~merged["dislike"].to_numpy(dtype=bool)
    )

    positives = merged.loc[positive_mask, [user_column, item_column]].copy()
    positives[user_column] = positives[user_column].astype(str)
    positives[item_column] = positives[item_column].astype(str)
    positives = positives.drop_duplicates().reset_index(drop=True)
    return positives

def build_metadata_embeddings(
    items_metadata: pd.DataFrame,
    item_column: str,
    n_author_buckets: int = 32,
) -> pd.DataFrame:
    metadata_table = items_metadata[[item_column, "author_id", "duration"]].copy()
    metadata_table[item_column] = metadata_table[item_column].astype(str)

    author_bucket = (metadata_table["author_id"].astype(np.uint64) % n_author_buckets).to_numpy()
    features = np.zeros((len(metadata_table), n_author_buckets + 1), dtype=np.float32)
    features[np.arange(len(metadata_table)), author_bucket] = 1.0
    features[:, -1] = metadata_table["duration"].astype(np.float32).to_numpy() / 255.0

    feature_columns = [f"dim_{index}" for index in range(features.shape[1])]
    embeddings_table = pd.DataFrame(features, columns=feature_columns)
    embeddings_table.insert(0, item_column, metadata_table[item_column].to_numpy())
    return embeddings_table

def build_content_embedding_views(
    embeddings_path: str,
    items_to_keep: set[str],
    item_column: str,
    dimensions: list[int],
) -> dict[str, pd.DataFrame]:
    archive = np.load(Path(embeddings_path))
    item_ids = archive["item_id"].astype(str)
    embeddings = np.asarray(archive["embedding"], dtype=np.float32)

    keep_mask = np.isin(item_ids, np.array(sorted(items_to_keep), dtype=str))
    filtered_item_ids = item_ids[keep_mask]
    filtered_embeddings = embeddings[keep_mask]

    results = {}
    for embedding_dim in dimensions:
        sliced_embeddings = filtered_embeddings[:, :embedding_dim]
        feature_columns = [f"dim_{index}" for index in range(embedding_dim)]
        embedding_view = pd.DataFrame(sliced_embeddings, columns=feature_columns)
        embedding_view.insert(0, item_column, filtered_item_ids)
        results[f"multimodal_{embedding_dim}d"] = embedding_view

    return results

def cache_filtered_embeddings(
    source_embeddings_path: str,
    cached_embeddings_path: str,
    items_to_keep: set[str],
) -> str:
    cache_path = Path(cached_embeddings_path)
    if cache_path.exists():
        return str(cache_path)

    archive = np.load(Path(source_embeddings_path))
    item_ids = archive["item_id"].astype(str)
    embeddings = np.asarray(archive["embedding"], dtype=np.float16)
    keep_mask = np.isin(item_ids, np.array(sorted(items_to_keep), dtype=str))

    filtered_item_ids = item_ids[keep_mask]
    filtered_embeddings = embeddings[keep_mask]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        item_id=filtered_item_ids,
        embedding=filtered_embeddings,
    )
    return str(cache_path)
