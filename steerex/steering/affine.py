"""Affine steering mode with low-rank matrix."""

from typing import Callable, List, Optional

import torch

from steerex.steering.base import SteeringMode
from steerex.core.types import TokenSpec
from steerex.core.config import AffineConfig


class AffineSteering(SteeringMode):
    """
    Low-rank affine steering: x' = x + Mx + v

    Where M = L^T @ R is a low-rank matrix.

    This allows learning input-dependent modifications
    while keeping the parameter count manageable.

    Attributes:
        vector: The additive steering vector (hidden_dim,).
        matrix_left: Left factor of M (rank, hidden_dim).
        matrix_right: Right factor of M (rank, hidden_dim).
        config: Affine-specific configuration.

    Reference:
        Inspired by MELBO: https://www.lesswrong.com/posts/ioPnHKFyy4Cw2Gr2x
    """

    def __init__(
        self,
        vector: Optional[torch.Tensor] = None,
        matrix_left: Optional[torch.Tensor] = None,
        matrix_right: Optional[torch.Tensor] = None,
        config: Optional[AffineConfig] = None,
    ):
        """
        Initialize affine steering.

        Args:
            vector: Pre-trained steering vector.
            matrix_left: Left factor of the low-rank matrix.
            matrix_right: Right factor of the low-rank matrix.
            config: Affine configuration.
        """
        self.vector = vector
        self.matrix_left = matrix_left
        self.matrix_right = matrix_right
        self.config = config or AffineConfig()

    def init_parameters(
        self,
        hidden_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        starting_norm: float = 1.0,
    ) -> None:
        """Initialize vector and matrix factors."""
        rank = self.config.rank
        starting_affine_norm = self.config.starting_affine_norm

        # Initialize vector
        vector = torch.randn(hidden_dim, device=device, dtype=dtype)
        vector = vector / vector.norm() * starting_norm
        vector.requires_grad_(True)
        self.vector = vector

        # Initialize matrix factors
        matrix_left = torch.randn(rank, hidden_dim, device=device, dtype=dtype)
        matrix_right = torch.randn(rank, hidden_dim, device=device, dtype=dtype)

        # Normalize rows to starting_affine_norm
        matrix_left = torch.einsum(
            "rm, r -> rm", matrix_left, starting_affine_norm / matrix_left.norm(dim=1)
        )
        matrix_right = torch.einsum(
            "rm, r -> rm", matrix_right, starting_affine_norm / matrix_right.norm(dim=1)
        )

        matrix_left.requires_grad_(True)
        matrix_right.requires_grad_(True)

        self.matrix_left = matrix_left
        self.matrix_right = matrix_right

    def get_matrix(self) -> torch.Tensor:
        """Compute the full matrix M = L^T @ R."""
        return self.matrix_left.T @ self.matrix_right

    def create_hook(
        self,
        token_slice: TokenSpec = None,
        strength: float = 1.0,
    ) -> Callable:
        """Create hook that applies affine transformation."""
        vector = self.vector
        matrix_left = self.matrix_left
        matrix_right = self.matrix_right
        idx = token_slice if token_slice is not None else slice(None)

        def hook_fn(module, args):
            hidden_states = args[0]
            modified = hidden_states.clone()

            x = modified[:, idx]
            v = vector.to(x.device, x.dtype)
            ml = matrix_left.to(x.device, x.dtype)
            mr = matrix_right.to(x.device, x.dtype)

            # Compute M @ x for each position
            # M = L^T @ R, so Mx = L^T @ (R @ x)
            matrix = ml.T @ mr
            affine_term = torch.einsum("...n, mn -> ...m", x, matrix)

            # Apply: x' = x + Mx + v
            modified[:, idx] = x + strength * (affine_term + v)

            return (modified,) + args[1:]

        return hook_fn

    def parameters(self) -> List[torch.Tensor]:
        """Return vector and both matrix factors."""
        if self.vector is None:
            raise ValueError("Parameters not initialized.")
        return [self.vector, self.matrix_left, self.matrix_right]

    def get_vector(self) -> torch.Tensor:
        """Return detached copy of the steering vector."""
        if self.vector is None:
            raise ValueError("Vector not initialized.")
        return self.vector.detach().clone()

    def set_vector(self, vector: torch.Tensor) -> None:
        """Set the steering vector."""
        self.vector = vector.clone()
        self.vector.requires_grad_(True)

    def apply_constraints(self, max_norm: Optional[float] = None) -> None:
        """Apply norm constraints to vector and matrix factors."""
        super().apply_constraints(max_norm)

        # Also constrain matrix factor row norms
        max_affine_norm = self.config.max_affine_norm
        if max_affine_norm is not None:
            with torch.no_grad():
                for matrix in [self.matrix_left, self.matrix_right]:
                    if matrix is None:
                        continue
                    row_norms = matrix.norm(dim=1)
                    scale = torch.where(
                        row_norms > max_affine_norm,
                        max_affine_norm / row_norms,
                        torch.ones_like(row_norms),
                    )
                    matrix[:] = torch.einsum("rm, r -> rm", matrix, scale)
