"""Base class for steering modes."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional
from contextlib import contextmanager

import torch

from steerex.core.types import TokenSpec


class SteeringMode(ABC):
    """
    Abstract base class for all steering strategies.

    A steering mode defines how to modify model activations during
    the forward pass. Subclasses implement different strategies:
    - VectorSteering: x' = x + v
    - ClampSteering: x' = x - (x·v̂)v̂ + c*v
    - AffineSteering: x' = x + Mx + v

    The steering mode owns its parameters (vector, matrix, etc.)
    and is responsible for:
    1. Initializing parameters
    2. Creating hooks for activation modification
    3. Providing parameters to the optimizer
    """

    @abstractmethod
    def init_parameters(
        self,
        hidden_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        starting_norm: float = 1.0,
    ) -> None:
        """
        Initialize steering parameters with random values.

        Args:
            hidden_dim: Size of the hidden dimension.
            device: Device to create tensors on.
            dtype: Data type for tensors.
            starting_norm: Initial L2 norm of the vector.
        """
        ...

    @abstractmethod
    def create_hook(
        self,
        token_slice: TokenSpec = None,
        strength: float = 1.0,
    ) -> Callable:
        """
        Create a forward pre-hook for activation modification.

        Args:
            token_slice: Which token positions to modify.
                None means all positions.
            strength: Multiplier for the steering effect.

        Returns:
            A hook function compatible with register_forward_pre_hook.
        """
        ...

    @abstractmethod
    def parameters(self) -> List[torch.Tensor]:
        """
        Return list of tensors to optimize.

        Returns:
            List of parameter tensors with requires_grad=True.
        """
        ...

    @abstractmethod
    def get_vector(self) -> torch.Tensor:
        """
        Return the primary steering vector.

        Returns:
            The steering vector (detached copy).
        """
        ...

    def set_vector(self, vector: torch.Tensor) -> None:
        """
        Set the steering vector (for loading saved vectors).

        Args:
            vector: The vector to set.
        """
        raise NotImplementedError("Subclass must implement set_vector")

    def apply_constraints(self, max_norm: Optional[float] = None) -> None:
        """
        Apply constraints to parameters after optimization step.

        Args:
            max_norm: If set, clip vector norm to this value.
        """
        if max_norm is not None:
            with torch.no_grad():
                for param in self.parameters():
                    norm = param.norm()
                    if norm > max_norm:
                        param.mul_(max_norm / norm)

    @contextmanager
    def apply_to_model(
        self,
        model,
        layers: List[int],
        token_slice: TokenSpec = None,
        strength: float = 1.0,
    ):
        """
        Context manager to apply steering to a model.

        Args:
            model: The model to steer.
            layers: Which layers to apply steering to.
            token_slice: Which tokens to steer.
            strength: Steering strength multiplier.

        Yields:
            None. Steering is active within the context.
        """
        # This is a convenience method; actual implementation
        # depends on the backend being used
        raise NotImplementedError(
            "Use backend.apply_steering() instead for model-specific logic"
        )
