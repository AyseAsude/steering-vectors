"""Visualization utilities for steering vectors."""

from typing import List, Optional, Dict, Any

import torch
import numpy as np


def plot_similarity_matrix(
    matrix: torch.Tensor,
    labels: Optional[List[str]] = None,
    title: str = "Vector Similarity Matrix",
    figsize: tuple = (10, 8),
    cmap: str = "RdBu_r",
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot a similarity matrix as a heatmap.

    Args:
        matrix: Similarity matrix (n, n).
        labels: Labels for rows/columns.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.
        show: Whether to display the plot.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix.cpu().float().numpy() if isinstance(matrix, torch.Tensor) else matrix,
                   cmap=cmap, vmin=-1, vmax=1)

    if labels:
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)

    plt.colorbar(im, ax=ax, label="Cosine Similarity")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_umap(
    embeddings: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[List[Any]] = None,
    title: str = "Vector UMAP Projection",
    figsize: tuple = (10, 8),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot 2D UMAP projection of vectors.

    Args:
        embeddings: 2D coordinates (n, 2).
        labels: Text labels for each point.
        colors: Color for each point (can be cluster IDs).
        title: Plot title.
        figsize: Figure size.
        show: Whether to display.
        save_path: Optional path to save.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=colors,
        cmap="tab10" if colors is not None else None,
        s=100,
        alpha=0.7,
    )

    if labels:
        for i, label in enumerate(labels):
            ax.annotate(
                label,
                (embeddings[i, 0], embeddings[i, 1]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    if colors is not None:
        plt.colorbar(scatter, ax=ax, label="Cluster")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_loss_history(
    losses: List[float],
    target_loss: Optional[float] = None,
    title: str = "Optimization Loss",
    figsize: tuple = (10, 6),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot loss over optimization steps.

    Args:
        losses: Loss values per step.
        target_loss: Optional target loss line.
        title: Plot title.
        figsize: Figure size.
        show: Whether to display.
        save_path: Optional path to save.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(losses, label="Loss", linewidth=2)

    if target_loss is not None:
        ax.axhline(y=target_loss, color="r", linestyle="--",
                   label=f"Target: {target_loss:.4f}")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_vector_norms(
    vectors: List[torch.Tensor],
    labels: Optional[List[str]] = None,
    title: str = "Vector Norms",
    figsize: tuple = (12, 6),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot bar chart of vector norms.

    Args:
        vectors: List of vectors.
        labels: Labels for each vector.
        title: Plot title.
        figsize: Figure size.
        show: Whether to display.
        save_path: Optional path to save.

    Returns:
        Matplotlib figure.
    """
    import matplotlib.pyplot as plt

    norms = [v.norm().item() for v in vectors]

    if labels is None:
        labels = [f"vec_{i}" for i in range(len(vectors))]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(range(len(norms)), norms)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("L2 Norm")
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig
