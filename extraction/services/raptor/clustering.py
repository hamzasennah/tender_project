import logging

import numpy as np
import umap
from sklearn.mixture import GaussianMixture

from extraction.services.raptor.types import RaptorCluster

logger = logging.getLogger(__name__)


def _validate_embedding_matrix(vectors):
    if not vectors:
        raise ValueError("empty_embedding_matrix")

    dimension = len(vectors[0])
    if dimension <= 0:
        raise ValueError("invalid_embedding_dimension")

    matrix = []
    for vector in vectors:
        if len(vector) != dimension:
            raise ValueError("inconsistent_embedding_dimensions")
        matrix.append([float(value) for value in vector])

    return np.asarray(matrix, dtype=np.float64)


def _candidate_cluster_limit(item_count, max_clusters):
    if item_count <= 2:
        return 1
    return max(1, min(max_clusters, item_count // 2))


def reduce_embeddings(vectors, *, random_state):
    matrix = _validate_embedding_matrix(vectors)
    item_count, dimension = matrix.shape
    if item_count <= 3 or dimension <= 3:
        return matrix

    n_neighbors = max(2, min(15, item_count - 1))
    n_components = max(2, min(5, item_count - 2, dimension))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        random_state=random_state,
        transform_seed=random_state,
        low_memory=True,
    )
    return reducer.fit_transform(matrix)


def _best_gmm(reduced_matrix, *, max_clusters, random_state):
    item_count = len(reduced_matrix)
    candidate_limit = _candidate_cluster_limit(item_count, max_clusters)
    best_model = None
    best_bic = None

    for cluster_count in range(1, candidate_limit + 1):
        model = GaussianMixture(
            n_components=cluster_count,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6,
        )
        model.fit(reduced_matrix)
        bic = model.bic(reduced_matrix)
        if best_bic is None or bic < best_bic:
            best_bic = bic
            best_model = model

    return best_model


def cluster_embeddings(
    vectors,
    *,
    max_clusters,
    soft_cluster_threshold,
    random_state,
):
    item_count = len(vectors)
    if item_count <= 0:
        return ()
    if item_count <= 2:
        return (RaptorCluster(cluster_id=0, member_indices=tuple(range(item_count))),)

    reduced_matrix = reduce_embeddings(vectors, random_state=random_state)
    model = _best_gmm(
        reduced_matrix,
        max_clusters=max_clusters,
        random_state=random_state,
    )
    probabilities = model.predict_proba(reduced_matrix)
    assignments = {cluster_id: set() for cluster_id in range(model.n_components)}

    for item_index, probability_row in enumerate(probabilities):
        assigned = [
            cluster_id
            for cluster_id, probability in enumerate(probability_row)
            if probability >= soft_cluster_threshold
        ]
        if not assigned:
            assigned = [int(np.argmax(probability_row))]
        for cluster_id in assigned:
            assignments[cluster_id].add(item_index)

    clusters = [
        RaptorCluster(cluster_id=cluster_id, member_indices=tuple(sorted(indices)))
        for cluster_id, indices in sorted(assignments.items())
        if indices
    ]
    if not clusters:
        return (RaptorCluster(cluster_id=0, member_indices=tuple(range(item_count))),)
    return tuple(clusters)
