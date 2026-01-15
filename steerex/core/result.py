"""Optimization result container."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch


@dataclass
class OptimizationResult:
    """
    Result from steering vector optimization.

    Attributes:
        vector: The optimized steering vector.
        matrix: For affine steering, the optimized matrix (L^T @ R).
        iterations: Number of optimization steps taken.
        final_loss: Loss value at termination.
        loss_history: Loss at each step (if recorded).
        vector_history: Vector at each step (if recorded).
        per_completion_losses: Final loss for each completion.
        metadata: Additional info (config, timing, etc.).
    """

    vector: torch.Tensor
    matrix: Optional[torch.Tensor] = None

    # Diagnostics
    iterations: int = 0
    final_loss: float = 0.0
    loss_history: Optional[List[float]] = None
    vector_history: Optional[List[torch.Tensor]] = None
    per_completion_losses: Optional[List[List[float]]] = None

    # For reproducibility
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def norm(self) -> float:
        """Return the L2 norm of the steering vector."""
        return self.vector.norm().item()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for saving)."""
        return {
            "vector": self.vector.detach().cpu(),
            "matrix": self.matrix.detach().cpu() if self.matrix is not None else None,
            "iterations": self.iterations,
            "final_loss": self.final_loss,
            "loss_history": self.loss_history,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Deserialize from dictionary."""
        return cls(
            vector=data["vector"],
            matrix=data.get("matrix"),
            iterations=data.get("iterations", 0),
            final_loss=data.get("final_loss", 0.0),
            loss_history=data.get("loss_history"),
            metadata=data.get("metadata", {}),
        )
