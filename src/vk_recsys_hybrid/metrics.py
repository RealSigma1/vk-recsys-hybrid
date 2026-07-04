from __future__ import annotations

import math
from collections import defaultdict


def group_items_by_user(rows: list[tuple[str, str]]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for user_id, item_id in rows:
        grouped[user_id].add(item_id)
    return dict(grouped)


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for item_id in recommended[:k] if item_id in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for item_id in recommended[:k] if item_id in relevant)
    return hits / len(relevant)


def average_precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0

    score = 0.0
    hits = 0
    for index, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            hits += 1
            score += hits / index

    denominator = min(len(relevant), k)
    return score / denominator if denominator else 0.0


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for index, item_id in enumerate(recommended[:k], start=1):
        if item_id in relevant:
            dcg += 1.0 / math.log2(index + 1.0)

    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(index + 1.0) for index in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def hit_rate_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    return 1.0 if any(item_id in relevant for item_id in recommended[:k]) else 0.0


def evaluate_ranking(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k: int,
    catalog_size: int,
) -> dict[str, float]:
    users = [user_id for user_id in ground_truth if user_id in recommendations]
    if not users:
        raise ValueError("No overlapping users between recommendations and ground truth.")

    precision_scores: list[float] = []
    recall_scores: list[float] = []
    map_scores: list[float] = []
    ndcg_scores: list[float] = []
    hit_scores: list[float] = []
    recommended_items: set[str] = set()

    for user_id in users:
        recommended = recommendations[user_id]
        relevant = ground_truth[user_id]

        precision_scores.append(precision_at_k(recommended, relevant, k))
        recall_scores.append(recall_at_k(recommended, relevant, k))
        map_scores.append(average_precision_at_k(recommended, relevant, k))
        ndcg_scores.append(ndcg_at_k(recommended, relevant, k))
        hit_scores.append(hit_rate_at_k(recommended, relevant, k))
        recommended_items.update(recommended[:k])

    return {
        "precision_at_k": sum(precision_scores) / len(precision_scores),
        "recall_at_k": sum(recall_scores) / len(recall_scores),
        "map_at_k": sum(map_scores) / len(map_scores),
        "ndcg_at_k": sum(ndcg_scores) / len(ndcg_scores),
        "hit_rate_at_k": sum(hit_scores) / len(hit_scores),
        "coverage_at_k": len(recommended_items) / catalog_size if catalog_size else 0.0,
        "evaluated_users": float(len(users)),
    }
