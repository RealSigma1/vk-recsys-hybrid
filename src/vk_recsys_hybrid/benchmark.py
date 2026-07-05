from dataclasses import dataclass

import pandas as pd

from vk_recsys_hybrid.metrics import evaluate_ranking, group_items_by_user
from vk_recsys_hybrid.recommenders import (
    ContentKNNRecommender,
    HybridKNNRecommender,
    ItemJaccardCFRecommender,
    ItemKNNCFRecommender,
    PopularityRecommender,
)

@dataclass
class BenchmarkResult:
    seed: int
    split_name: str
    model: str
    embedding_name: str
    precision_at_k: float
    recall_at_k: float
    map_at_k: float
    ndcg_at_k: float
    hit_rate_at_k: float
    coverage_at_k: float
    evaluated_users: float

def _build_recommendations(model, users: list[str], top_k: int) -> dict[str, list[str]]:
    return {user_id: model.recommend(user_id, top_k) for user_id in users}

def _sample_eval_users(
    ground_truth: dict[str, set[str]],
    seed: int,
    bootstrap_user_fraction: float,
    max_users: int | None = None,
) -> dict[str, set[str]]:
    all_users = sorted(ground_truth.keys())
    sample_size = len(all_users)
    if bootstrap_user_fraction < 1.0:
        sample_size = max(1, int(round(len(all_users) * bootstrap_user_fraction)))
    if max_users is not None:
        sample_size = min(sample_size, max_users)
    if sample_size >= len(all_users):
        sampled_users = set(all_users)
    else:
        sampled_users_series = pd.Series(all_users).sample(
            n=sample_size, random_state=seed, replace=False
        )
        sampled_users = set(sampled_users_series.tolist())
    return {user_id: items for user_id, items in ground_truth.items() if user_id in sampled_users}

def run_fixed_split_benchmark(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    embeddings: dict[str, pd.DataFrame],
    user_column: str,
    item_column: str,
    top_k: int,
    hybrid_alpha: float,
    split_name: str,
    seeds: list[int],
    bootstrap_user_fraction: float = 1.0,
    max_eval_users: int | None = None,
) -> list[BenchmarkResult]:
    ground_truth = group_items_by_user(
        list(eval_df[[user_column, item_column]].itertuples(index=False, name=None))
    )
    catalog_size = int(train_df[item_column].astype(str).nunique())

    trained_models = {}
    baseline_models = [
        ("popularity", PopularityRecommender(user_column=user_column, item_column=item_column)),
        ("item_knn_cf", ItemKNNCFRecommender(user_column=user_column, item_column=item_column)),
        (
            "item_jaccard_cf",
            ItemJaccardCFRecommender(user_column=user_column, item_column=item_column),
        ),
    ]
    for model_name, model in baseline_models:
        trained_models[(model_name, "-")] = model.fit(train_df)

    for embedding_name, embedding_df in embeddings.items():
        content_model = ContentKNNRecommender(
            user_column=user_column,
            item_column=item_column,
            embeddings=embedding_df,
            embedding_column=item_column,
        ).fit(train_df)
        trained_models[("content_knn", embedding_name)] = content_model

        hybrid_model = HybridKNNRecommender(
            user_column=user_column,
            item_column=item_column,
            cf_model=ItemKNNCFRecommender(user_column=user_column, item_column=item_column),
            content_model=ContentKNNRecommender(
                user_column=user_column,
                item_column=item_column,
                embeddings=embedding_df,
                embedding_column=item_column,
            ),
            alpha=hybrid_alpha,
        ).fit(train_df)
        trained_models[("hybrid_knn", embedding_name)] = hybrid_model

    results = []
    for seed in seeds:
        sampled_ground_truth = _sample_eval_users(
            ground_truth=ground_truth,
            seed=seed,
            bootstrap_user_fraction=bootstrap_user_fraction,
            max_users=max_eval_users,
        )
        eval_users = sorted(sampled_ground_truth.keys())

        for (model_name, embedding_name), model in trained_models.items():
            recommendations = _build_recommendations(model, eval_users, top_k)
            metrics = evaluate_ranking(
                recommendations, sampled_ground_truth, top_k, catalog_size
            )
            results.append(
                BenchmarkResult(
                    seed=seed,
                    split_name=split_name,
                    model=model_name,
                    embedding_name=embedding_name,
                    **metrics,
                )
            )

    return results

def run_single_seed_benchmark(
    interactions: pd.DataFrame,
    embeddings: dict[str, pd.DataFrame],
    user_column: str,
    item_column: str,
    top_k: int,
    test_fraction: float,
    min_test_items: int,
    hybrid_alpha: float,
    seed: int,
) -> list[BenchmarkResult]:
    from vk_recsys_hybrid.split import random_user_holdout_split

    train_df, test_df = random_user_holdout_split(
        interactions=interactions,
        user_column=user_column,
        item_column=item_column,
        test_fraction=test_fraction,
        min_test_items=min_test_items,
        seed=seed,
    )
    return run_fixed_split_benchmark(
        train_df=train_df,
        eval_df=test_df,
        embeddings=embeddings,
        user_column=user_column,
        item_column=item_column,
        top_k=top_k,
        hybrid_alpha=hybrid_alpha,
        split_name="holdout",
        seeds=[seed],
        bootstrap_user_fraction=1.0,
        max_eval_users=None,
    )

def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "precision_at_k",
        "recall_at_k",
        "map_at_k",
        "ndcg_at_k",
        "hit_rate_at_k",
        "coverage_at_k",
        "evaluated_users",
    ]

    aggregated = (
        results_df.groupby(["split_name", "model", "embedding_name"], dropna=False)[
            metric_columns
        ]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregated.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else column
        for column in aggregated.columns.to_flat_index()
    ]
    return (
        aggregated.sort_values(["split_name", "model", "embedding_name"])
        .reset_index(drop=True)
    )
