"""Clamp steering mode."""

from typing import Callable, List, Optional

import torch

from steerex.steering.base import SteeringMode
from steerex.core.types import TokenSpec


class ClampSteering(SteeringMode):
    """
    Clamp steering: ablate direction, then add scaled vector.

    x' = x - (x·v̂)v̂ + c*v

    This removes the component of activations in the vector
    direction (ablation) and replaces it with a fixed value.

    Useful for "clamping" a concept to a specific activation level.

    Attributes:
        vector: The steering vector (hidden_dim,).
        clamp_value: The value to clamp to.

    Reference:
        Section 2 of https://arxiv.org/pdf/2411.09003
    """

    def __init__(
        self,
        vector: Optional[torch.Tensor] = None,
        clamp_value: float = 1.0,
    ):
        """
        Initialize clamp steering.

        Args:
            vector: Pre-trained steering vector.
            clamp_value: The coefficient for the clamped direction.
        """
        self.vector = vector
        self.clamp_value = clamp_value
        self._ablation_matrix: Optional[torch.Tensor] = None

    def _compute_ablation_matrix(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute matrix that projects out the v direction.

        M = -v⊗v / ||v||²

        When applied: x + Mx = x - (x·v̂)v̂
        """
        return -torch.outer(v, v) / (v.norm() ** 2)

    def init_parameters(
        self,
        hidden_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        starting_norm: float = 1.0,
    ) -> None:
        """Initialize vector with random direction."""
        vector = torch.randn(hidden_dim, device=device, dtype=dtype)
        vector = vector / vector.norm() * starting_norm
        vector.requires_grad_(True)
        self.vector = vector

    def create_hook(
        self,
        token_slice: TokenSpec = None,
        strength: float = 1.0,
    ) -> Callable:
        """Create hook that clamps activations in the vector direction."""
        vector = self.vector
        clamp_value = self.clamp_value * strength
        idx = token_slice if token_slice is not None else slice(None)

        def hook_fn(module, args):
            hidden_states = args[0]
            modified = hidden_states.clone()

            x = modified[:, idx].detach().clone()

            # Compute ablation matrix
            v = vector.to(x.device, x.dtype)
            abl_matrix = -torch.outer(v, v) / (v.norm() ** 2)

            # Apply: x' = x + Mx + c*v = x - (x·v̂)v̂ + c*v
            ablation_term = torch.einsum("...n, mn -> ...m", x, abl_matrix)
            modified[:, idx] = x + ablation_term + clamp_value * v

            return (modified,) + args[1:]

        return hook_fn

    def parameters(self) -> List[torch.Tensor]:
        """Return the vector as the parameter to optimize."""
        if self.vector is None:
            raise ValueError("Parameters not initialized.")
        return [self.vector]

    def get_vector(self) -> torch.Tensor:
        """Return detached copy of the steering vector."""
        if self.vector is None:
            raise ValueError("Vector not initialized.")
        return self.vector.detach().clone()

    def set_vector(self, vector: torch.Tensor) -> None:
        """Set the steering vector."""
        self.vector = vector.clone()
        self.vector.requires_grad_(True)
