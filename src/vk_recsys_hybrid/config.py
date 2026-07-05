from dataclasses import asdict, dataclass

@dataclass
class BenchmarkConfig:
    interactions_path: str
    user_column: str
    item_column: str
    embeddings: dict[str, str]
    top_k: int = 10
    min_user_interactions: int = 5
    min_item_interactions: int = 5
    test_fraction: float = 0.2
    min_test_items: int = 1
    hybrid_alpha: float = 0.7
    seeds: list[int] | None = None
    reports_dir: str = "reports"

    def to_dict(self) -> dict:
        return asdict(self)
