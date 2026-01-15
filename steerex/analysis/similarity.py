"""Similarity analysis for steering vectors."""

from typing import List, Optional, Tuple, Dict, Any

import torch
import numpy as np


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a_norm = a / a.norm()
    b_norm = b / b.norm()
    return torch.dot(a_norm, b_norm).item()


def pairwise_cosine_similarity(
    vectors_a: List[torch.Tensor],
    vectors_b: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarities.

    Args:
        vectors_a: First set of vectors.
        vectors_b: Second set of vectors. If None, compute self-similarity.

    Returns:
        Similarity matrix (len(a), len(b)).
    """
    # Stack and normalize
    A = torch.stack(vectors_a)
    A_norm = A / A.norm(dim=1, keepdim=True)

    if vectors_b is None:
        return A_norm @ A_norm.T
    else:
        B = torch.stack(vectors_b)
        B_norm = B / B.norm(dim=1, keepdim=True)
        return A_norm @ B_norm.T


def similarity_matrix(
    vectors: List[torch.Tensor],
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute similarity matrix with labels.

    Args:
        vectors: List of steering vectors.
        labels: Optional labels for each vector.

    Returns:
        Dictionary with 'matrix' and 'labels'.
    """
    matrix = pairwise_cosine_similarity(vectors)

    if labels is None:
        labels = [f"vec_{i}" for i in range(len(vectors))]

    return {
        "matrix": matrix,
        "labels": labels,
    }


def cluster_vectors(
    vectors: List[torch.Tensor],
    n_clusters: int = 5,
    method: str = "kmeans",
) -> Tuple[List[int], Any]:
    """
    Cluster vectors into groups.

    Args:
        vectors: List of steering vectors.
        n_clusters: Number of clusters.
        method: Clustering method ("kmeans", "hierarchical", "spectral").

    Returns:
        Tuple of (cluster_labels, cluster_model).
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

    # Stack vectors to numpy
    X = torch.stack(vectors).numpy()

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "hierarchical":
        model = AgglomerativeClustering(n_clusters=n_clusters)
    elif method == "spectral":
        model = SpectralClustering(n_clusters=n_clusters, random_state=42)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    labels = model.fit_predict(X)

    return labels.tolist(), model


def find_outliers(
    vectors: List[torch.Tensor],
    threshold: float = 0.5,
) -> List[int]:
    """
    Find vectors that are dissimilar to most others.

    Args:
        vectors: List of steering vectors.
        threshold: Maximum average similarity to be considered outlier.

    Returns:
        Indices of outlier vectors.
    """
    sim_matrix = pairwise_cosine_similarity(vectors)
    n = len(vectors)

    outliers = []
    for i in range(n):
        # Average similarity to others (excluding self)
        avg_sim = (sim_matrix[i].sum() - 1) / (n - 1)
        if avg_sim < threshold:
            outliers.append(i)

    return outliers


def project_vectors(
    vectors: List[torch.Tensor],
    method: str = "umap",
    n_components: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Project vectors to lower dimensions for visualization.

    Args:
        vectors: List of steering vectors.
        method: Projection method ("umap", "tsne", "pca").
        n_components: Number of output dimensions.
        **kwargs: Additional arguments for the projection method.

    Returns:
        Projected coordinates (n_vectors, n_components).
    """
    X = torch.stack(vectors).numpy()

    if method == "umap":
        from umap import UMAP
        reducer = UMAP(n_components=n_components, **kwargs)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, **kwargs)
    elif method == "pca":
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, **kwargs)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    return reducer.fit_transform(X)
