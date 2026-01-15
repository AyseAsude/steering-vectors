"""Training datapoint definition."""

from dataclasses import dataclass, field
from typing import List, Optional

from steerex.core.types import TokenSpec


@dataclass
class TrainingDatapoint:
    """
    A single training example for steering vector optimization.

    Matches the original steering_opt.TrainingDatapoint structure.

    Attributes:
        prompt: The input prompt text.
        src_completions: Completions to suppress (decrease probability).
        dst_completions: Completions to promote (increase probability).
        src_target_losses: Per-completion target loss thresholds for src.
        dst_target_losses: Per-completion target loss thresholds for dst.
        token_slice: Which token positions to apply steering to.
            None means all tokens.
        negate: If True, negate the steering vector for this datapoint.
            Useful for contrastive training.

    Example:
        >>> dp = TrainingDatapoint(
        ...     prompt="Is this code good?",
        ...     dst_completions=["Yes, it's perfect!"],
        ...     src_completions=["Let me review critically..."],
        ... )
    """

    prompt: str
    src_completions: List[str] = field(default_factory=list)
    dst_completions: List[str] = field(default_factory=list)
    src_target_losses: Optional[List[float]] = None
    dst_target_losses: Optional[List[float]] = None
    token_slice: TokenSpec = None
    negate: bool = False

    def __post_init__(self):
        """Validate the datapoint."""
        if not self.src_completions and not self.dst_completions:
            raise ValueError(
                "At least one of src_completions or dst_completions must be provided"
            )
