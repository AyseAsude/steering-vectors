"""
Unified data format for contrast pairs.

This module provides ContrastPair, a unified data structure that can be
used with any extraction method (CAA, gradient, hybrid).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union


@dataclass
class ContrastPair:
    """
    A pair of positive and negative examples for contrastive extraction.

    This unified format supports both:
    - Chat messages (for CAA extraction from chat models)
    - Prompt/completion pairs (for gradient optimization)

    Attributes:
        positive: The positive example (exhibits target behavior).
        negative: The negative example (does not exhibit target behavior).

    Example with chat messages:
        >>> pair = ContrastPair(
        ...     positive={"messages": [
        ...         {"role": "user", "content": "Is my code good?"},
        ...         {"role": "assistant", "content": "It's perfect!"},
        ...     ]},
        ...     negative={"messages": [
        ...         {"role": "user", "content": "Is my code good?"},
        ...         {"role": "assistant", "content": "It has some issues."},
        ...     ]},
        ... )

    Example with prompt/completion:
        >>> pair = ContrastPair(
        ...     positive={"prompt": "My favorite color is", "completion": " blue!"},
        ...     negative={"prompt": "My favorite color is", "completion": " unknown."},
        ... )
    """

    positive: Dict[str, Any]
    negative: Dict[str, Any]

    def validate(self, index: Optional[int] = None) -> None:
        """
        Validate that this pair is properly formatted.

        Args:
            index: Optional index for error messages.

        Raises:
            ValueError: If the pair is invalid.
        """
        prefix = f"ContrastPair[{index}]" if index is not None else "ContrastPair"

        # Check that positive and negative have same format
        pos_keys = set(self.positive.keys())
        neg_keys = set(self.negative.keys())

        if pos_keys != neg_keys:
            raise ValueError(
                f"{prefix}: positive and negative must have same format. "
                f"positive has {pos_keys}, negative has {neg_keys}"
            )

        # Validate based on format
        if "messages" in self.positive:
            self._validate_messages(self.positive["messages"], f"{prefix}.positive")
            self._validate_messages(self.negative["messages"], f"{prefix}.negative")
        elif "prompt" in self.positive:
            self._validate_prompt_completion(self.positive, f"{prefix}.positive")
            self._validate_prompt_completion(self.negative, f"{prefix}.negative")
        else:
            raise ValueError(
                f"{prefix}: must contain either 'messages' or 'prompt'/'completion'"
            )

    def _validate_messages(self, messages: List[Dict], prefix: str) -> None:
        """Validate chat messages format."""
        if not messages:
            raise ValueError(f"{prefix}.messages is empty")

        if messages[-1].get("role") != "assistant":
            raise ValueError(
                f"{prefix}.messages must end with assistant role, "
                f"got '{messages[-1].get('role')}'"
            )

    def _validate_prompt_completion(self, data: Dict, prefix: str) -> None:
        """Validate prompt/completion format."""
        if "prompt" not in data:
            raise ValueError(f"{prefix} missing 'prompt'")
        if "completion" not in data:
            raise ValueError(f"{prefix} missing 'completion'")

    @property
    def format(self) -> str:
        """Return the format of this pair ('messages' or 'prompt_completion')."""
        if "messages" in self.positive:
            return "messages"
        return "prompt_completion"

    def get_positive_messages(self) -> List[Dict[str, str]]:
        """Get positive example as messages (raises if wrong format)."""
        if self.format != "messages":
            raise ValueError("This pair uses prompt/completion format, not messages")
        return self.positive["messages"]

    def get_negative_messages(self) -> List[Dict[str, str]]:
        """Get negative example as messages (raises if wrong format)."""
        if self.format != "messages":
            raise ValueError("This pair uses prompt/completion format, not messages")
        return self.negative["messages"]

    def get_positive_prompt_completion(self) -> tuple:
        """Get positive example as (prompt, completion) tuple."""
        if self.format == "messages":
            raise ValueError("This pair uses messages format, not prompt/completion")
        return self.positive["prompt"], self.positive["completion"]

    def get_negative_prompt_completion(self) -> tuple:
        """Get negative example as (prompt, completion) tuple."""
        if self.format == "messages":
            raise ValueError("This pair uses messages format, not prompt/completion")
        return self.negative["prompt"], self.negative["completion"]

    @classmethod
    def from_messages(
        cls,
        positive_messages: List[Dict[str, str]],
        negative_messages: List[Dict[str, str]],
    ) -> "ContrastPair":
        """
        Create a ContrastPair from chat messages.

        Args:
            positive_messages: Messages for positive example.
            negative_messages: Messages for negative example.

        Returns:
            ContrastPair instance.
        """
        return cls(
            positive={"messages": positive_messages},
            negative={"messages": negative_messages},
        )

    @classmethod
    def from_prompt_completion(
        cls,
        prompt: str,
        positive_completion: str,
        negative_completion: str,
    ) -> "ContrastPair":
        """
        Create a ContrastPair from prompt and completions.

        Args:
            prompt: Shared prompt for both examples.
            positive_completion: Completion for positive example.
            negative_completion: Completion for negative example.

        Returns:
            ContrastPair instance.
        """
        return cls(
            positive={"prompt": prompt, "completion": positive_completion},
            negative={"prompt": prompt, "completion": negative_completion},
        )
