"""
Noise injection for steering vector optimization.

Uses strategy pattern with closures - no ABC classes needed.
Strategies are plugged in at initialization time.

This module implements the noisy steering algorithm from:
https://github.com/jacobdunefsky/llm-steering-opt

The noise helps steering vectors generalize beyond training examples by
forcing the optimization to find solutions robust to small perturbations.
"""

from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from steering_vectors.core.config import NoiseConfig


# Type aliases for strategy signatures
NoiseGeneratorFn = Callable[
    [Union[torch.Size, Tuple[int, ...]], Union[str, torch.device], torch.dtype],
    torch.Tensor,
]
NoiseProjectorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class NoiseApplicator:
    """
    Applies pluggable noise strategies during optimization.

    Strategies are injected at initialization - no if/else branching
    in the apply() method. This follows the strategy pattern.

    Attributes:
        generator: Function that generates raw noise.
        projector: Function that transforms noise (e.g., tangent space projection).
        noise_iters: Number of noise samples per completion during optimization.

    Example:
        >>> applicator = NoiseApplicator(
        ...     generator=gaussian_generator(scale=0.1),
        ...     projector=tangent_space_projector(use_relu=False),
        ...     noise_iters=3,
        ... )
        >>> noisy_vec = applicator.apply(vector, gradient)
    """

    def __init__(
        self,
        generator: NoiseGeneratorFn,
        projector: NoiseProjectorFn,
        noise_iters: int = 1,
    ):
        """
        Initialize the noise applicator.

        Args:
            generator: A function that generates noise given (shape, device, dtype).
            projector: A function that projects noise given (noise, gradient).
            noise_iters: Number of noise samples per completion.
        """
        self.generator = generator
        self.projector = projector
        self.noise_iters = noise_iters

    def apply(
        self,
        vector: torch.Tensor,
        gradient: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply noise to vector using configured strategies.

        Args:
            vector: The steering vector to perturb.
            gradient: Optional gradient for tangent space projection.
                     If None, projection is skipped.

        Returns:
            The vector with noise added.
        """
        noise = self.generator(vector.shape, vector.device, vector.dtype)
        if gradient is not None:
            noise = self.projector(noise, gradient)
        return vector + noise

    def reset(self) -> None:
        """Reset stateful generators (e.g., AntiPGD)."""
        if hasattr(self.generator, "reset"):
            self.generator.reset()


def gaussian_generator(scale: float) -> NoiseGeneratorFn:
    """
    Create a generator that produces scaled Gaussian noise.

    This is the simplest noise strategy - just random perturbations.

    Args:
        scale: Standard deviation of the noise.

    Returns:
        A function that generates noise: (shape, device, dtype) -> Tensor
    """

    def generate(
        shape: Union[torch.Size, Tuple[int, ...]],
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=dtype) * scale

    return generate


def antipgd_generator(scale: float) -> NoiseGeneratorFn:
    """
    Create a stateful generator with anti-correlated noise.

    Each call produces: noise_t - noise_{t-1}

    This implements the anti-PGD technique from:
    https://proceedings.mlr.press/v162/orvieto22a/orvieto22a.pdf

    The anti-correlation encourages the optimizer to find solutions
    robust to opposite-direction perturbations.

    Args:
        scale: Standard deviation of the base noise.

    Returns:
        A function that generates anti-correlated noise.
        The function has a .reset() method to clear state.
    """
    state: dict = {"prev": 0}

    def generate(
        shape: Union[torch.Size, Tuple[int, ...]],
        device: Union[str, torch.device],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        noise = torch.randn(shape, device=device, dtype=dtype) * scale
        prev = state["prev"]
        if isinstance(prev, torch.Tensor):
            result = noise - prev.to(device)
        else:
            result = noise
        state["prev"] = noise.detach().clone()
        return result

    def reset() -> None:
        state["prev"] = 0

    # Attach reset method to the function
    generate.reset = reset  # type: ignore[attr-defined]
    return generate


def identity_projector() -> NoiseProjectorFn:
    """
    Create a no-op projector that returns noise unchanged.

    Use this when tangent space projection is disabled.

    Returns:
        A function that returns noise unchanged.
    """
    return lambda noise, gradient: noise


def tangent_space_projector(use_relu: bool = False) -> NoiseProjectorFn:
    """
    Create a projector that removes gradient-aligned component from noise.

    Projects noise onto the tangent space of the loss surface,
    preventing noise from fighting the optimization direction.

    The projection formula:
        projected = noise - (noise · grad / ||grad||²) * grad

    This ensures the noise is perpendicular to the gradient,
    so it doesn't interfere with the optimization direction.

    Args:
        use_relu: If True, only remove noise pointing toward decreasing loss.
                  This is a more conservative approach that preserves noise
                  that would increase the loss (making optimization harder).

    Returns:
        A function that projects noise perpendicular to the gradient.
    """

    def project(noise: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
        # Compute projection coefficient: (noise · grad) / ||grad||²
        grad_norm_sq = gradient.norm() ** 2 + 1e-8
        abl_component = (
            torch.dot(noise.flatten(), gradient.flatten()) / grad_norm_sq
        )

        if use_relu:
            # Only ablate if noise points toward decreasing loss
            # (negative component means noise reduces loss)
            abl_component = -F.relu(-abl_component)

        # Remove the gradient-aligned component
        return noise - abl_component * gradient

    return project


def create_noise_applicator(
    config: Optional[NoiseConfig],
) -> Optional[NoiseApplicator]:
    """
    Create a NoiseApplicator from configuration.

    This factory function selects the appropriate generator and projector
    strategies based on the NoiseConfig settings.

    Args:
        config: Noise configuration. If None or noise_scale is None,
               returns None (noise disabled).

    Returns:
        A configured NoiseApplicator, or None if noise is disabled.

    Example:
        >>> config = NoiseConfig(
        ...     noise_scale=0.1,
        ...     tangent_space_noise=True,
        ...     noise_abl_relu=False,
        ...     noise_iters=3,
        ...     anti_pgd=False,
        ... )
        >>> applicator = create_noise_applicator(config)
    """
    if config is None or config.noise_scale is None:
        return None

    # Select generator strategy
    generator: NoiseGeneratorFn
    if config.anti_pgd:
        generator = antipgd_generator(config.noise_scale)
    else:
        generator = gaussian_generator(config.noise_scale)

    # Select projector strategy
    projector: NoiseProjectorFn
    if config.tangent_space_noise:
        projector = tangent_space_projector(config.noise_abl_relu)
    else:
        projector = identity_projector()

    return NoiseApplicator(generator, projector, config.noise_iters)
