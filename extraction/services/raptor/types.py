from dataclasses import dataclass


@dataclass(frozen=True)
class RaptorBuildItem:
    node_id: int
    text: str
    vector: list[float]
    level: int


@dataclass(frozen=True)
class RaptorCluster:
    cluster_id: int
    member_indices: tuple[int, ...]


@dataclass(frozen=True)
class RaptorLevelBuild:
    level: int
    clusters: tuple[RaptorCluster, ...]


@dataclass(frozen=True)
class RaptorRetrievedNode:
    node_id: int
    level: int
    node_index: int
    node_type: str
    text: str
    cosine_distance: float
    similarity_score: float
    chunk_id: int | None = None
    chunk_index: int | None = None
    expanded_from_parent_id: int | None = None
    retrieval_score: float | None = None
    selection_reason: str = "vector"
    matched_subintents: tuple[int, ...] = ()


@dataclass(frozen=True)
class RaptorContext:
    text: str
    sources: list
    metadata: dict
