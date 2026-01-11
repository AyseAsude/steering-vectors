"""Optimization configuration."""

from typing import Optional
from pydantic import BaseModel, Field


class OptimizationConfig(BaseModel):
    """
    Configuration for steering vector optimization.

    All hyperparameters in one place for reproducibility.

    Attributes:
        lr: Learning rate for Adam optimizer.
        max_iters: Maximum number of optimization steps.
        coldness: Inverse temperature for softmax (higher = sharper).
        starting_norm: Initial norm of the steering vector.
        max_norm: Clip vector norm to this value after each step.
        satisfice: If True, optimize squared diff from per-datapoint target losses.
        normalize_by_length: Divide loss by completion length.
        use_one_minus: For suppression, use log(1-p) vs -log(p).
        use_batched: Use batched forward passes for faster optimization.
        batch_size: Batch size for batched optimization.
    """

    # Basic optimization
    lr: float = Field(default=0.1, gt=0)
    max_iters: int = Field(default=50, gt=0)

    # Temperature
    coldness: float = Field(default=0.7, gt=0)

    # Norm constraints
    starting_norm: float = Field(default=1.0, gt=0)
    max_norm: Optional[float] = Field(default=None, gt=0)

    # Loss behavior
    satisfice: bool = False
    normalize_by_length: bool = False
    use_one_minus: bool = True

    # Batched optimization (performance)
    use_batched: bool = Field(
        default=True,
        description="Use batched forward passes for 10-50x faster optimization.",
    )
    batch_size: int = Field(
        default=16,
        gt=0,
        le=128,
        description="Batch size for batched optimization. Larger = faster but more memory.",
    )

    # Numerical stability
    grad_clip_value: Optional[float] = Field(
        default=1.0,
        gt=0,
        description="Clip gradient values to this magnitude to prevent NaN from gradient explosion.",
    )
    loss_eps: float = Field(
        default=1e-6,
        gt=0,
        description="Epsilon for numerical stability in log computations. Larger = more stable but less precise.",
    )

    class Config:
        frozen = False  # Allow modification after creation


class AffineConfig(BaseModel):
    """Additional config for affine steering mode."""

    rank: int = Field(default=1, gt=0)
    max_affine_norm: float = Field(default=2.0, gt=0)
    starting_affine_norm: float = Field(default=1.0, gt=0)


class NoiseConfig(BaseModel):
    """Configuration for noisy steering regularization."""

    noise_scale: Optional[float] = None
    tangent_space_noise: bool = True
    noise_abl_relu: bool = False
    noise_iters: int = Field(default=1, ge=1)
    anti_pgd: bool = False


class ManifoldConfig(BaseModel):
    """
    Configuration for manifold regularization.

    Used with ManifoldLoss to keep steering vectors on the manifold
    of natural model activations.
    """

    # Data collection
    n_samples: int = Field(
        default=100,
        gt=0,
        description="Number of Alpaca samples to use for PCA.",
    )
    num_activations_per_sample: int = Field(
        default=10,
        gt=0,
        description="Number of activation vectors to sample per example.",
    )
    layer: int = Field(
        default=16,
        ge=0,
        description="Layer to extract activations from.",
    )

    # PCA configuration
    explained_variance_threshold: float = Field(
        default=0.95,
        gt=0,
        le=1.0,
        description="Fraction of variance to explain with PCA components.",
    )

    # Loss weighting
    weight: float = Field(
        default=1.0,
        gt=0,
        description="Weight for manifold loss (lambda).",
    )

    # Reproducibility
    seed: int = Field(default=42, description="Random seed for sampling.")
