from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = np.array(list(scores.values()), dtype=np.float64)
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < 1e-12:
        return {key: 1.0 for key in scores}
    return {
        key: (value - min_value) / (max_value - min_value)
        for key, value in scores.items()
    }

@dataclass
class BaseRecommender:
    user_column: str
    item_column: str

    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        raise NotImplementedError

    def recommend(self, user_id: str, k: int) -> list[str]:
        raise NotImplementedError

@dataclass
class PopularityRecommender(BaseRecommender):
    popularity: list[str] | None = None
    user_history: dict[str, set[str]] | None = None

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        counts = interactions[self.item_column].value_counts()
        self.popularity = counts.index.astype(str).tolist()
        self.user_history = (
            interactions.groupby(self.user_column)[self.item_column]
            .agg(lambda items: set(items.astype(str)))
            .to_dict()
        )
        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        return [item_id for item_id in (self.popularity or []) if item_id not in seen][:k]

@dataclass
class ItemKNNCFRecommender(BaseRecommender):
    item_neighbors: dict[str, dict[str, float]] | None = None
    item_popularity: dict[str, int] | None = None
    user_history: dict[str, set[str]] | None = None
    fallback_items: list[str] | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNCFRecommender":
        user_history_by_user = (
            interactions.groupby(self.user_column)[self.item_column]
            .agg(lambda items: set(items.astype(str)))
        )
        self.user_history = user_history_by_user.to_dict()

        item_popularity = Counter(interactions[self.item_column].astype(str).tolist())
        co_counts = defaultdict(Counter)

        for user_items in self.user_history.values():
            item_list = list(user_items)
            for item_id in item_list:
                for related_item_id in item_list:
                    if item_id == related_item_id:
                        continue
                    co_counts[item_id][related_item_id] += 1

        neighbors = {}
        for item_id, related in co_counts.items():
            scores = {}
            for other_item, overlap in related.items():
                denom = np.sqrt(item_popularity[item_id] * item_popularity[other_item])
                if denom > 0:
                    scores[other_item] = float(overlap / denom)
            neighbors[item_id] = scores

        self.item_neighbors = neighbors
        self.item_popularity = dict(item_popularity)
        self.fallback_items = [item_id for item_id, item_count in item_popularity.most_common()]
        return self

    def score_items(self, user_id: str) -> dict[str, float]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        scores = defaultdict(float)

        for item_id in seen:
            for neighbor_id, similarity in self.item_neighbors.get(item_id, {}).items():
                if neighbor_id not in seen:
                    scores[neighbor_id] += similarity

        return dict(scores)

    def recommend(self, user_id: str, k: int) -> list[str]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        scores = self.score_items(user_id)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        recommendations = [item_id for item_id, _ in ranked if item_id not in seen]

        for item_id in self.fallback_items or []:
            if item_id not in seen and item_id not in recommendations:
                recommendations.append(item_id)
            if len(recommendations) >= k:
                break

        return recommendations[:k]

@dataclass
class ItemJaccardCFRecommender(BaseRecommender):
    item_neighbors: dict[str, dict[str, float]] | None = None
    item_popularity: dict[str, int] | None = None
    user_history: dict[str, set[str]] | None = None
    fallback_items: list[str] | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemJaccardCFRecommender":
        user_history_by_user = (
            interactions.groupby(self.user_column)[self.item_column]
            .agg(lambda items: set(items.astype(str)))
        )
        self.user_history = user_history_by_user.to_dict()

        item_popularity = Counter(interactions[self.item_column].astype(str).tolist())
        co_counts = defaultdict(Counter)

        for user_items in self.user_history.values():
            item_list = list(user_items)
            for item_id in item_list:
                for related_item_id in item_list:
                    if item_id == related_item_id:
                        continue
                    co_counts[item_id][related_item_id] += 1

        neighbors = {}
        for item_id, related in co_counts.items():
            scores = {}
            for other_item, overlap in related.items():
                union_size = item_popularity[item_id] + item_popularity[other_item] - overlap
                if union_size > 0:
                    scores[other_item] = float(overlap / union_size)
            neighbors[item_id] = scores

        self.item_neighbors = neighbors
        self.item_popularity = dict(item_popularity)
        self.fallback_items = [item_id for item_id, item_count in item_popularity.most_common()]
        return self

    def score_items(self, user_id: str) -> dict[str, float]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        scores = defaultdict(float)

        for item_id in seen:
            for neighbor_id, similarity in self.item_neighbors.get(item_id, {}).items():
                if neighbor_id not in seen:
                    scores[neighbor_id] += similarity

        return dict(scores)

    def recommend(self, user_id: str, k: int) -> list[str]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        scores = self.score_items(user_id)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        recommendations = [item_id for item_id, _ in ranked if item_id not in seen]

        for item_id in self.fallback_items or []:
            if item_id not in seen and item_id not in recommendations:
                recommendations.append(item_id)
            if len(recommendations) >= k:
                break

        return recommendations[:k]

@dataclass
class ContentKNNRecommender(BaseRecommender):
    embeddings: pd.DataFrame
    embedding_column: str
    item_vectors: dict[str, np.ndarray] | None = None
    candidate_items: list[str] | None = None
    user_history: dict[str, set[str]] | None = None
    item_matrix: np.ndarray | None = None
    item_index: dict[str, int] | None = None

    def fit(self, interactions: pd.DataFrame) -> "ContentKNNRecommender":
        user_history_by_user = (
            interactions.groupby(self.user_column)[self.item_column]
            .agg(lambda items: set(items.astype(str)))
        )
        self.user_history = user_history_by_user.to_dict()

        available_items = set(interactions[self.item_column].astype(str))
        feature_columns = [
            column for column in self.embeddings.columns if column != self.embedding_column
        ]
        available_embeddings = self.embeddings[
            self.embeddings[self.embedding_column].astype(str).isin(available_items)
        ].copy()
        available_embeddings[self.embedding_column] = available_embeddings[
            self.embedding_column
        ].astype(str)

        vectors = {}
        for item_row in available_embeddings.to_dict("records"):
            item_id = item_row[self.embedding_column]
            vector = np.array(
                [item_row[column_name] for column_name in feature_columns],
                dtype=np.float64,
            )
            norm = np.linalg.norm(vector)
            if norm > 0:
                vectors[item_id] = vector / norm

        self.item_vectors = vectors
        self.candidate_items = list(vectors.keys())
        self.item_index = {item_id: index for index, item_id in enumerate(self.candidate_items)}
        self.item_matrix = np.vstack([vectors[item_id] for item_id in self.candidate_items])
        return self

    def score_items(self, user_id: str) -> dict[str, float]:
        seen = self.user_history.get(user_id, set()) if self.user_history else set()
        seen_indices = [
            self.item_index[item_id] for item_id in seen if item_id in self.item_index
        ]
        if not seen_indices:
            return {}

        user_profile = self.item_matrix[seen_indices].mean(axis=0)
        all_scores = self.item_matrix @ user_profile
        scores = {
            item_id: float(all_scores[index])
            for index, item_id in enumerate(self.candidate_items or [])
            if item_id not in seen
        }
        return scores

    def recommend(self, user_id: str, k: int) -> list[str]:
        scores = self.score_items(user_id)
        ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
        return [item_id for item_id, _ in ranked[:k]]

@dataclass
class HybridKNNRecommender(BaseRecommender):
    cf_model: ItemKNNCFRecommender
    content_model: ContentKNNRecommender
    alpha: float = 0.7

    def fit(self, interactions: pd.DataFrame) -> "HybridKNNRecommender":
        self.cf_model.fit(interactions)
        self.content_model.fit(interactions)
        return self

    def recommend(self, user_id: str, k: int) -> list[str]:
        cf_scores = _normalize_scores(self.cf_model.score_items(user_id))
        content_scores = _normalize_scores(self.content_model.score_items(user_id))

        candidates = set(cf_scores) | set(content_scores)
        combined_scores = {
            item_id: self.alpha * cf_scores.get(item_id, 0.0)
            + (1.0 - self.alpha) * content_scores.get(item_id, 0.0)
            for item_id in candidates
        }

        seen = self.cf_model.user_history.get(user_id, set()) if self.cf_model.user_history else set()
        ranked = sorted(combined_scores.items(), key=lambda pair: pair[1], reverse=True)
        recommendations = [item_id for item_id, _ in ranked if item_id not in seen]

        for item_id in self.cf_model.fallback_items or []:
            if item_id not in seen and item_id not in recommendations:
                recommendations.append(item_id)
            if len(recommendations) >= k:
                break

        return recommendations[:k]
