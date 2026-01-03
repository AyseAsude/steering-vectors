"""Analysis tools for steering vectors."""

from steering_vectors.analysis.similarity import (
    cosine_similarity,
    pairwise_cosine_similarity,
    similarity_matrix,
    cluster_vectors,
)
from steering_vectors.analysis.visualization import (
    plot_similarity_matrix,
    plot_umap,
    plot_loss_history,
)

__all__ = [
    "cosine_similarity",
    "pairwise_cosine_similarity",
    "similarity_matrix",
    "cluster_vectors",
    "plot_similarity_matrix",
    "plot_umap",
    "plot_loss_history",
]
