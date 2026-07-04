from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
VENDOR_DIR = PROJECT_ROOT / ".vendor"
for candidate in [str(VENDOR_DIR), str(SRC_DIR)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import pandas as pd

from vk_recsys_hybrid.benchmark import aggregate_results, run_fixed_split_benchmark
from vk_recsys_hybrid.config import BenchmarkConfig
from vk_recsys_hybrid.preprocessing import filter_interactions
from vk_recsys_hybrid.vklsvd import (
    build_content_embedding_views,
    build_metadata_embeddings,
    cache_filtered_embeddings,
    load_many_parquets,
    prepare_positive_interactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the VK-LSVD hybrid recommender benchmark on a real subset."
    )
    parser.add_argument("--data-root", required=True, help="Path to local VK-LSVD root.")
    parser.add_argument("--subset", default="ur0.01_ir0.01")
    parser.add_argument("--train-start-week", type=int, default=20)
    parser.add_argument("--train-end-week", type=int, default=24)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--min-item-interactions", type=int, default=3)
    parser.add_argument("--min-watch-ratio", type=float, default=0.8)
    parser.add_argument("--hybrid-alpha", type=float, default=0.7)
    parser.add_argument("--bootstrap-user-fraction", type=float, default=0.8)
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--embedding-dims", nargs="+", type=int, default=[16, 64])
    parser.add_argument("--reports-dir", default="reports/vklsvd")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_subset_paths(
    data_root: Path,
    subset: str,
    train_start_week: int,
    train_end_week: int,
) -> tuple[list[str], str, str]:
    train_paths = [
        str(data_root / "subsamples" / subset / "train" / f"week_{week:02}.parquet")
        for week in range(train_start_week, train_end_week + 1)
    ]
    val_path = str(data_root / "subsamples" / subset / "validation" / "week_25.parquet")
    test_path = str(data_root / "subsamples" / subset / "test" / "week_26.parquet")
    return train_paths, val_path, test_path


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    item_column = "item_id"
    user_column = "user_id"

    train_paths, val_path, test_path = build_subset_paths(
        data_root=data_root,
        subset=args.subset,
        train_start_week=args.train_start_week,
        train_end_week=args.train_end_week,
    )
    items_metadata_path = data_root / "metadata" / "items_metadata.parquet"
    users_metadata_path = data_root / "metadata" / "users_metadata.parquet"
    item_embeddings_path = data_root / "metadata" / "item_embeddings.npz"

    print("Loading metadata...")
    items_metadata = pd.read_parquet(items_metadata_path)
    users_metadata = pd.read_parquet(users_metadata_path)

    print("Loading train/validation/test interactions...")
    raw_train = load_many_parquets(train_paths)
    raw_val = pd.read_parquet(val_path)
    raw_test = pd.read_parquet(test_path)

    print("Preparing implicit positive interactions...")
    train_positive = prepare_positive_interactions(
        interactions=raw_train,
        items_metadata=items_metadata,
        user_column=user_column,
        item_column=item_column,
        min_watch_ratio=args.min_watch_ratio,
    )
    val_positive = prepare_positive_interactions(
        interactions=raw_val,
        items_metadata=items_metadata,
        user_column=user_column,
        item_column=item_column,
        min_watch_ratio=args.min_watch_ratio,
    )
    test_positive = prepare_positive_interactions(
        interactions=raw_test,
        items_metadata=items_metadata,
        user_column=user_column,
        item_column=item_column,
        min_watch_ratio=args.min_watch_ratio,
    )

    train_filtered = filter_interactions(
        interactions=train_positive,
        user_column=user_column,
        item_column=item_column,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
    )
    train_users = set(train_filtered[user_column].astype(str))
    train_items = set(train_filtered[item_column].astype(str))

    val_filtered = val_positive[
        val_positive[user_column].isin(train_users) & val_positive[item_column].isin(train_items)
    ].drop_duplicates().reset_index(drop=True)
    test_filtered = test_positive[
        test_positive[user_column].isin(train_users) & test_positive[item_column].isin(train_items)
    ].drop_duplicates().reset_index(drop=True)

    filtered_items_metadata = items_metadata[
        items_metadata[item_column].astype(str).isin(train_items)
    ].copy()

    print("Building structured metadata embeddings...")
    embeddings = {
        "metadata_structured": build_metadata_embeddings(
            items_metadata=filtered_items_metadata,
            item_column=item_column,
            n_author_buckets=32,
        )
    }
    cached_embeddings_path = (
        data_root
        / "metadata"
        / f"item_embeddings_{args.subset}_w{args.train_start_week:02}_{args.train_end_week:02}_filtered.npz"
    )
    print("Caching filtered multimodal embeddings...")
    filtered_embeddings_path = cache_filtered_embeddings(
        source_embeddings_path=str(item_embeddings_path),
        cached_embeddings_path=str(cached_embeddings_path),
        items_to_keep=train_items,
    )
    print("Building multimodal embedding views...")
    embeddings.update(
        build_content_embedding_views(
            embeddings_path=filtered_embeddings_path,
            items_to_keep=train_items,
            item_column=item_column,
            dimensions=args.embedding_dims,
        )
    )

    config = BenchmarkConfig(
        interactions_path=str(data_root),
        user_column=user_column,
        item_column=item_column,
        embeddings={name: f"in_memory:{name}" for name in embeddings},
        top_k=args.top_k,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        test_fraction=0.0,
        min_test_items=1,
        hybrid_alpha=args.hybrid_alpha,
        seeds=args.seeds,
        reports_dir=args.reports_dir,
    )

    all_results = []
    for split_name, eval_df in {
        "validation": val_filtered,
        "test": test_filtered,
    }.items():
        print(f"Running benchmark for {split_name}...")
        split_results = run_fixed_split_benchmark(
            train_df=train_filtered,
            eval_df=eval_df,
            embeddings=embeddings,
            user_column=user_column,
            item_column=item_column,
            top_k=args.top_k,
            hybrid_alpha=args.hybrid_alpha,
            split_name=split_name,
            seeds=args.seeds,
            bootstrap_user_fraction=args.bootstrap_user_fraction,
            max_eval_users=args.max_eval_users,
        )
        all_results.extend(split_results)

    results_df = pd.DataFrame([asdict(result) for result in all_results])
    summary_df = aggregate_results(results_df)

    dataset_stats = {
        "subset": args.subset,
        "train_start_week": args.train_start_week,
        "train_end_week": args.train_end_week,
        "raw_train_interactions": int(len(raw_train)),
        "raw_validation_interactions": int(len(raw_val)),
        "raw_test_interactions": int(len(raw_test)),
        "train_positive_interactions": int(len(train_positive)),
        "validation_positive_interactions": int(len(val_positive)),
        "test_positive_interactions": int(len(test_positive)),
        "filtered_train_interactions": int(len(train_filtered)),
        "filtered_validation_interactions": int(len(val_filtered)),
        "filtered_test_interactions": int(len(test_filtered)),
        "filtered_train_users": int(train_filtered[user_column].nunique()),
        "filtered_train_items": int(train_filtered[item_column].nunique()),
        "max_eval_users": args.max_eval_users,
        "users_metadata_rows": int(len(users_metadata)),
        "items_metadata_rows": int(len(items_metadata)),
    }

    reports_dir = PROJECT_ROOT / args.reports_dir
    ensure_dir(reports_dir)

    runs_path = reports_dir / "benchmark_runs.csv"
    summary_csv_path = reports_dir / "benchmark_summary.csv"
    summary_json_path = reports_dir / "benchmark_summary.json"
    config_path = reports_dir / "run_config.json"
    stats_path = reports_dir / "dataset_stats.json"

    results_df.to_csv(runs_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)
    summary_json_path.write_text(
        summary_df.to_json(orient="records", indent=2), encoding="utf-8"
    )
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    stats_path.write_text(json.dumps(dataset_stats, indent=2), encoding="utf-8")

    print(summary_df.to_string(index=False))
    print(json.dumps(dataset_stats, indent=2))
    print(f"Per-run results saved to: {runs_path}")
    print(f"Summary saved to: {summary_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
