"""Analysis tools for steering vectors."""

from steerex.analysis.similarity import (
    cosine_similarity,
    pairwise_cosine_similarity,
    similarity_matrix,
    cluster_vectors,
    find_outliers,
    project_vectors,
)
from steerex.analysis.visualization import (
    plot_similarity_matrix,
    plot_umap,
    plot_loss_history,
    plot_vector_norms,
)

__all__ = [
    "cosine_similarity",
    "pairwise_cosine_similarity",
    "similarity_matrix",
    "cluster_vectors",
    "find_outliers",
    "project_vectors",
    "plot_similarity_matrix",
    "plot_umap",
    "plot_loss_history",
    "plot_vector_norms",
]
