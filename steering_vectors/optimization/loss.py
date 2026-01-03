"""Loss components for steering optimization."""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch


class LossComponent(ABC):
    """
    Abstract base class for loss components.

    Loss components compute a scalar loss from model logits
    and target tokens. They can be composed to create complex
    loss functions.
    """

    @abstractmethod
    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """
        Compute loss contribution.

        Args:
            logits: Model output logits (seq_len, vocab_size).
            target_ids: Full sequence token IDs (seq_len,).
            prompt_len: Number of prompt tokens (loss only on rest).
            coldness: Inverse temperature for softmax.
            eps: Small constant for numerical stability.

        Returns:
            Scalar loss tensor.
        """
        ...


class PromotionLoss(LossComponent):
    """
    Increase probability of target tokens.

    Loss = -sum(log P(target_token))

    Used for dst_completions (things we want the model to say).
    """

    def __init__(self, normalize_by_length: bool = False):
        """
        Args:
            normalize_by_length: If True, divide by completion length.
        """
        self.normalize_by_length = normalize_by_length

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute negative log probability of target tokens."""
        probs = torch.softmax(logits * coldness, dim=-1)
        total_len = len(target_ids)

        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for i in range(prompt_len, total_len):
            target_token = target_ids[i]
            prob = probs[i - 1, target_token]
            loss = loss - torch.log(prob + eps)

        if self.normalize_by_length and (total_len - prompt_len) > 0:
            loss = loss / (total_len - prompt_len)

        return loss


class SuppressionLoss(LossComponent):
    """
    Decrease probability of target tokens.

    If use_one_minus=True:
        Loss = -sum(log(1 - P(target_token)))
    Else:
        Loss = sum(log P(target_token))

    Used for src_completions (things we want the model NOT to say).
    """

    def __init__(
        self,
        use_one_minus: bool = True,
        normalize_by_length: bool = False,
    ):
        """
        Args:
            use_one_minus: Use log(1-p) vs -log(p).
            normalize_by_length: If True, divide by completion length.
        """
        self.use_one_minus = use_one_minus
        self.normalize_by_length = normalize_by_length

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute suppression loss."""
        probs = torch.softmax(logits * coldness, dim=-1)
        total_len = len(target_ids)

        loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        for i in range(prompt_len, total_len):
            target_token = target_ids[i]
            prob = probs[i - 1, target_token]

            if self.use_one_minus:
                loss = loss - torch.log(1 - prob + eps)
            else:
                loss = loss + torch.log(prob + eps)

        if self.normalize_by_length and (total_len - prompt_len) > 0:
            loss = loss / (total_len - prompt_len)

        return loss


class SatisficingLoss(LossComponent):
    """
    Penalize squared difference from target loss.

    Loss = (actual_loss - target_loss)^2

    Used when you want to achieve a specific loss value,
    not just minimize/maximize.
    """

    def __init__(
        self,
        base_loss: LossComponent,
        target_loss: float,
    ):
        """
        Args:
            base_loss: The underlying loss component.
            target_loss: The target value to achieve.
        """
        self.base_loss = base_loss
        self.target_loss = target_loss

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute squared difference from target."""
        actual_loss = self.base_loss.compute(
            logits, target_ids, prompt_len, coldness, eps
        )
        return (actual_loss - self.target_loss) ** 2


class CompositeLoss(LossComponent):
    """
    Combine multiple loss components.

    Loss = sum(component.compute(...) for component in components)
    """

    def __init__(self, *components: LossComponent):
        """
        Args:
            *components: Loss components to combine.
        """
        self.components = list(components)

    def add(self, component: LossComponent) -> "CompositeLoss":
        """Add a component and return self for chaining."""
        self.components.append(component)
        return self

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute sum of all component losses."""
        total = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        for component in self.components:
            total = total + component.compute(
                logits, target_ids, prompt_len, coldness, eps
            )
        return total


class WeightedLoss(LossComponent):
    """
    Apply a weight to a loss component.

    Loss = weight * base_loss.compute(...)
    """

    def __init__(self, base_loss: LossComponent, weight: float):
        """
        Args:
            base_loss: The underlying loss component.
            weight: Multiplier for the loss.
        """
        self.base_loss = base_loss
        self.weight = weight

    def compute(
        self,
        logits: torch.Tensor,
        target_ids: torch.Tensor,
        prompt_len: int,
        coldness: float = 1.0,
        eps: float = 1e-10,
    ) -> torch.Tensor:
        """Compute weighted loss."""
        return self.weight * self.base_loss.compute(
            logits, target_ids, prompt_len, coldness, eps
        )
