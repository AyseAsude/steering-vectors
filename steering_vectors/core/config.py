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
        noise: Optional noise configuration for regularization.
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

    # Noise regularization (for generalization)
    noise: Optional["NoiseConfig"] = None

    class Config:
        frozen = False  # Allow modification after creation


class AffineConfig(BaseModel):
    """Additional config for affine steering mode."""

    rank: int = Field(default=1, gt=0)
    max_affine_norm: float = Field(default=2.0, gt=0)
    starting_affine_norm: float = Field(default=1.0, gt=0)


class NoiseConfig(BaseModel):
    """
    Configuration for noisy steering regularization.

    Adding noise during optimization helps steering vectors generalize
    beyond the exact training examples. This implements the algorithm from:
    https://github.com/jacobdunefsky/llm-steering-opt

    Attributes:
        noise_scale: Standard deviation of Gaussian noise. None disables noise.
        tangent_space_noise: If True, project noise perpendicular to gradient.
            This prevents noise from fighting the optimization direction.
        noise_abl_relu: If True, only remove noise pointing toward decreasing loss.
            More conservative - keeps noise that makes optimization harder.
        noise_iters: Number of noise samples per completion. Higher = more robust
            but slower (more forward passes).
        anti_pgd: If True, use anti-correlated noise (noise_t - noise_{t-1}).
            Encourages robustness to opposite-direction perturbations.
    """

    noise_scale: Optional[float] = Field(default=None, gt=0)
    tangent_space_noise: bool = True
    noise_abl_relu: bool = False
    noise_iters: int = Field(default=1, ge=1)
    anti_pgd: bool = False
