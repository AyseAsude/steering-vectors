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
        target_loss: Stop when loss falls below this threshold.
        coldness: Inverse temperature for softmax (higher = sharper).
        starting_norm: Initial norm of the steering vector.
        max_norm: Clip vector norm to this value after each step.
        target_loss_iters: Stop after this many consecutive steps below target.
        eps: Small constant for numerical stability.
        sum_losses: If True, use sum of losses; else check each completion.
        satisfice: If True, optimize squared diff from target_loss.
        normalize_by_length: Divide loss by completion length.
        use_one_minus: For suppression, use log(1-p) vs -log(p).
    """

    # Basic optimization
    lr: float = Field(default=0.1, gt=0)
    max_iters: int = Field(default=50, gt=0)

    # Early stopping
    target_loss: Optional[float] = None
    target_loss_iters: int = Field(default=1, ge=1)
    eps: float = Field(default=1e-6, gt=0)

    # Temperature
    coldness: float = Field(default=0.7, gt=0)

    # Norm constraints
    starting_norm: float = Field(default=1.0, gt=0)
    max_norm: Optional[float] = Field(default=None, gt=0)

    # Loss behavior
    sum_losses: bool = True
    satisfice: bool = False
    normalize_by_length: bool = False
    use_one_minus: bool = True

    # Debugging
    debug: bool = False

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
