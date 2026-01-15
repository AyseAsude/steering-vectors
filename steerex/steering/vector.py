"""Standard additive vector steering."""

from typing import Callable, List, Optional

import torch

from steerex.steering.base import SteeringMode
from steerex.core.types import TokenSpec


class VectorSteering(SteeringMode):
    """
    Standard additive steering: x' = x + v

    The simplest steering mode. Adds a learned vector to
    the activations at specified positions.

    Attributes:
        vector: The steering vector (hidden_dim,).

    Example:
        >>> steering = VectorSteering()
        >>> steering.init_parameters(hidden_dim=4096, device="cuda")
        >>> hook = steering.create_hook(token_slice=slice(-5, None))
    """

    def __init__(self, vector: Optional[torch.Tensor] = None):
        """
        Initialize with optional pre-trained vector.

        Args:
            vector: Pre-trained steering vector. If None,
                call init_parameters() before use.
        """
        self.vector = vector

    def init_parameters(
        self,
        hidden_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        starting_norm: float = 1.0,
    ) -> None:
        """Initialize vector with random direction and specified norm."""
        vector = torch.randn(hidden_dim, device=device, dtype=dtype)
        vector = vector / vector.norm() * starting_norm
        vector.requires_grad_(True)
        self.vector = vector

    def create_hook(
        self,
        token_slice: TokenSpec = None,
        strength: float = 1.0,
    ) -> Callable:
        """Create hook that adds vector to activations."""
        vector = self.vector
        idx = token_slice if token_slice is not None else slice(None)

        def hook_fn(module, args):
            hidden_states = args[0]  # Shape: [batch, seq_len, hidden_dim]

            # Clone to avoid in-place modification issues
            modified = hidden_states.clone()
            modified[:, idx] = modified[:, idx] + strength * vector.to(
                modified.device, modified.dtype
            )

            return (modified,) + args[1:]

        return hook_fn

    def parameters(self) -> List[torch.Tensor]:
        """Return the vector as the only parameter."""
        if self.vector is None:
            raise ValueError("Parameters not initialized. Call init_parameters() first.")
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
