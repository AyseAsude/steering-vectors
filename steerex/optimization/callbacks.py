"""Callbacks for optimization loop."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any

import torch


class OptimizationCallback(ABC):
    """
    Abstract base class for optimization callbacks.

    Callbacks hook into the optimization loop to provide
    functionality like logging, early stopping, and checkpointing.
    """

    def on_optimization_start(
        self,
        parameters: List[torch.Tensor],
        config: Any,
    ) -> None:
        """Called before optimization begins."""
        pass

    @abstractmethod
    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Called after each optimization step.

        Args:
            step: Current step number (0-indexed).
            loss: Total loss value.
            parameters: List of parameter tensors.
            extra: Additional info (per-completion losses, etc.).

        Returns:
            True to continue, False to stop early.
        """
        ...

    def on_optimization_end(
        self,
        parameters: List[torch.Tensor],
        final_loss: float,
        total_steps: int,
    ) -> None:
        """Called after optimization completes."""
        pass


class EarlyStoppingCallback(OptimizationCallback):
    """
    Stop optimization when loss falls below target.

    Matches original target_loss + target_loss_target_iters behavior.
    """

    def __init__(
        self,
        target_loss: float,
        patience: int = 1,
        check_per_completion: bool = False,
    ):
        """
        Args:
            target_loss: Stop when loss <= this value.
            patience: Require this many consecutive hits.
            check_per_completion: If True, check each completion's loss.
        """
        self.target_loss = target_loss
        self.patience = patience
        self.check_per_completion = check_per_completion
        self._consecutive_hits = 0

    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if we should stop."""
        if self.check_per_completion and extra and "per_completion_losses" in extra:
            # Check if all completions are below target
            all_below = all(
                l <= self.target_loss
                for losses in extra["per_completion_losses"]
                for l in losses
            )
            hit = all_below
        else:
            hit = loss <= self.target_loss

        if hit:
            self._consecutive_hits += 1
            if self._consecutive_hits >= self.patience:
                return False  # Stop
        else:
            self._consecutive_hits = 0

        return True  # Continue


class ConvergenceCallback(OptimizationCallback):
    """Stop when loss stops changing."""

    def __init__(self, eps: float = 1e-6, patience: int = 1):
        """
        Args:
            eps: Minimum change threshold.
            patience: Consecutive steps below threshold to stop.
        """
        self.eps = eps
        self.patience = patience
        self._prev_loss: Optional[float] = None
        self._consecutive_hits = 0

    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check for convergence."""
        if self._prev_loss is not None:
            if abs(self._prev_loss - loss) < self.eps:
                self._consecutive_hits += 1
                if self._consecutive_hits >= self.patience:
                    return False
            else:
                self._consecutive_hits = 0

        self._prev_loss = loss
        return True


class LoggingCallback(OptimizationCallback):
    """Log progress during optimization."""

    def __init__(self, every_n: int = 10, verbose: bool = True):
        """
        Args:
            every_n: Log every N steps.
            verbose: If True, print to stdout.
        """
        self.every_n = every_n
        self.verbose = verbose

    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Log if on schedule."""
        if step % self.every_n == 0 and self.verbose:
            norm = parameters[0].norm().item() if parameters else 0.0
            print(f"Step {step:4d}: loss={loss:.6f}, norm={norm:.4f}")
        return True


class HistoryCallback(OptimizationCallback):
    """Record loss and vector history."""

    def __init__(self, record_vectors: bool = False):
        """
        Args:
            record_vectors: If True, save vector at each step (memory intensive).
        """
        self.record_vectors = record_vectors
        self.losses: List[float] = []
        self.vectors: List[torch.Tensor] = []
        self.per_completion_losses: List[Any] = []

    def on_optimization_start(
        self,
        parameters: List[torch.Tensor],
        config: Any,
    ) -> None:
        """Reset history."""
        self.losses = []
        self.vectors = []
        self.per_completion_losses = []

    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Record this step."""
        self.losses.append(loss)

        if self.record_vectors and parameters:
            self.vectors.append(parameters[0].detach().clone().cpu())

        if extra and "per_completion_losses" in extra:
            self.per_completion_losses.append(extra["per_completion_losses"])

        return True


class NormConstraintCallback(OptimizationCallback):
    """Apply norm constraint after each step."""

    def __init__(self, max_norm: float):
        """
        Args:
            max_norm: Maximum allowed norm.
        """
        self.max_norm = max_norm

    def on_step_end(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Clip norms if needed."""
        with torch.no_grad():
            for param in parameters:
                norm = param.norm()
                if norm > self.max_norm:
                    param.mul_(self.max_norm / norm)
        return True
