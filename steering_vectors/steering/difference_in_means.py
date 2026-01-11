"""Difference-in-Means steering vector extraction.

This module provides methods for extracting steering vectors by computing
the difference between mean activations of contrastive examples.

Formula: v = mean(positive_activations) - mean(negative_activations)
"""

from typing import Callable, List, Literal, Optional, TYPE_CHECKING

import torch

from steering_vectors.steering.base import SteeringMode
from steering_vectors.core.types import TokenSpec

if TYPE_CHECKING:
    from steering_vectors.backends.base import ModelBackend


class DifferenceInMeansSteering(SteeringMode):
    """
    Steering vector computed as the difference between mean activations.

    Unlike other steering modes that learn vectors through optimization,
    this mode extracts the vector directly from contrastive examples
    in a single forward pass.

    Once computed, the vector can be used for steering like VectorSteering.

    Example with chat messages (recommended for chat models):
        >>> steering = DifferenceInMeansSteering.from_contrastive_messages(
        ...     backend,
        ...     tokenizer,
        ...     positive_messages=[
        ...         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        ...     ],
        ...     negative_messages=[
        ...         [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Go away."}],
        ...     ],
        ...     layer=16,
        ... )
    """

    def __init__(self, vector: Optional[torch.Tensor] = None):
        """
        Initialize with optional pre-computed vector.

        Args:
            vector: Pre-computed steering vector. If None,
                use one of the from_* class methods to compute it.
        """
        self.vector = vector

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_contrastive_messages(
        cls,
        backend: "ModelBackend",
        tokenizer,
        positive_messages: List[List[dict]],
        negative_messages: List[List[dict]],
        layer: int,
        token_position: Literal["last_prompt_token", "mean"] = "mean",
    ) -> "DifferenceInMeansSteering":
        """
        Compute steering vector from contrastive chat message pairs.

        This method handles chat template application and response boundary
        detection internally, ensuring reliable extraction.

        Args:
            backend: Model backend for extracting activations.
            tokenizer: Tokenizer with chat template support.
            positive_messages: List of message lists representing positive examples.
                Each message list is a conversation: [{"role": "...", "content": "..."}].
                Must end with an assistant message.
            negative_messages: List of message lists representing negative examples.
                Same format as positive_messages.
            layer: Layer index to extract activations from.
            token_position: Which token position to extract activations from.
                - "last_prompt_token": Activation at the last token of the prompt
                  (right before assistant response begins).
                - "mean": Mean of all assistant response token activations (default).

        Returns:
            DifferenceInMeansSteering instance with computed vector.

        Raises:
            ValueError: If messages lists are empty or don't end with assistant.
        """
        _validate_message_lists(positive_messages, "positive_messages")
        _validate_message_lists(negative_messages, "negative_messages")

        positive_activations = _extract_response_activations_batch(
            backend, tokenizer, positive_messages, layer, token_position
        )
        negative_activations = _extract_response_activations_batch(
            backend, tokenizer, negative_messages, layer, token_position
        )

        positive_mean = torch.stack(positive_activations).mean(dim=0)
        negative_mean = torch.stack(negative_activations).mean(dim=0)

        return cls(vector=positive_mean - negative_mean)


    @classmethod
    def from_activations(
        cls,
        positive_activations: torch.Tensor,
        negative_activations: torch.Tensor,
    ) -> "DifferenceInMeansSteering":
        """
        Compute steering vector from pre-extracted activation tensors.

        Args:
            positive_activations: Tensor of shape [n_positive, hidden_dim].
            negative_activations: Tensor of shape [n_negative, hidden_dim].

        Returns:
            DifferenceInMeansSteering instance with computed vector.
        """
        positive_mean = positive_activations.mean(dim=0)
        negative_mean = negative_activations.mean(dim=0)
        return cls(vector=positive_mean - negative_mean)

    # =========================================================================
    # SteeringMode Interface
    # =========================================================================

    def init_parameters(
        self,
        hidden_dim: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        starting_norm: float = 1.0,
    ) -> None:
        """
        Initialize with random vector.

        Note: For DifferenceInMeansSteering, you typically use
        one of the from_* class methods instead.
        """
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
            hidden_states = args[0]
            modified = hidden_states.clone()
            modified[:, idx] = modified[:, idx] + strength * vector.to(
                modified.device, modified.dtype
            )
            return (modified,) + args[1:]

        return hook_fn

    def parameters(self) -> List[torch.Tensor]:
        """Return parameters for optimization (allows fine-tuning if desired)."""
        if self.vector is None:
            raise ValueError(
                "Vector not initialized. Use one of the from_* class methods."
            )
        return [self.vector]

    def get_vector(self) -> torch.Tensor:
        """Return detached copy of the steering vector."""
        if self.vector is None:
            raise ValueError("Vector not computed.")
        return self.vector.detach().clone()

    def set_vector(self, vector: torch.Tensor) -> None:
        """Set the steering vector."""
        self.vector = vector.clone()
        self.vector.requires_grad_(True)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def normalize(self, norm: float = 1.0) -> "DifferenceInMeansSteering":
        """
        Normalize the vector to a specified norm.

        Args:
            norm: Target L2 norm for the vector.

        Returns:
            Self for method chaining.
        """
        if self.vector is None:
            raise ValueError("Vector not computed.")
        with torch.no_grad():
            current_norm = self.vector.norm()
            if current_norm > 0:
                self.vector.mul_(norm / current_norm)
        return self


# =============================================================================
# Activation Extraction Utilities
# =============================================================================


def extract_response_activations(
    backend: "ModelBackend",
    tokenizer,
    messages: List[dict],
    layer: int,
    token_position: Literal["last_prompt_token", "mean"] = "mean",
) -> torch.Tensor:
    """
    Extract activation from a conversation based on token position strategy.

    Applies chat template, detects response boundary using offset mapping
    for reliable tokenization, and extracts activations based on the
    specified token position.

    Args:
        backend: Model backend.
        tokenizer: Tokenizer with chat template.
        messages: Conversation messages ending with assistant response.
        layer: Layer index to extract from.
        token_position: Which token position to extract activations from.
            - "last_prompt_token": Activation at the last token of the prompt.
            - "mean": Mean of all assistant response token activations.

    Returns:
        Activation tensor of shape [hidden_dim].

    Raises:
        ValueError: If messages don't end with assistant role.
    """
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Messages must end with an assistant message.")

    # Get full conversation text
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Get prompt-only text (everything before assistant response)
    # Use messages without the assistant turn, then add_generation_prompt
    # adds the assistant header (e.g., "<|im_start|>assistant\n")
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )

    # Verify prompt_text is a prefix of full_text
    if not full_text.startswith(prompt_text):
        raise ValueError(
            "Chat template produced inconsistent results: "
            "prompt_text is not a prefix of full_text"
        )

    # Find boundary using offset mapping (tokenize once, avoid boundary issues)
    prompt_char_end = len(prompt_text)
    encoding = tokenizer(
        full_text,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    full_ids = encoding["input_ids"].to(backend.get_device())
    offsets = encoding["offset_mapping"][0]  # [(start, end), ...]

    # Find first token that starts at or after prompt_char_end
    # This is the first assistant response token
    prompt_len = len(offsets)  # Default to full length if no boundary found
    for i, (start, end) in enumerate(offsets):
        if start >= prompt_char_end:
            prompt_len = i
            break

    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        raise ValueError(
            f"No response tokens found. prompt_len={prompt_len}, full_len={full_len}"
        )

    # Extract activations (uses output hook to get layer output)
    activations = _extract_layer_activations(backend, full_ids, layer)

    if token_position == "last_prompt_token":
        # Activation at the last token of the prompt (index prompt_len - 1)
        return activations[prompt_len - 1]
    elif token_position == "mean":
        # Mean of assistant response tokens only
        response_activations = activations[prompt_len:full_len]
        return response_activations.mean(dim=0)
    else:
        raise ValueError(
            f"Invalid token_position: {token_position}. "
            "Expected 'last_prompt_token' or 'mean'."
        )


# =============================================================================
# Private Helper Functions
# =============================================================================


def _validate_message_lists(messages_list: List[List[dict]], name: str) -> None:
    """Validate that message lists are non-empty and properly formatted."""
    if not messages_list:
        raise ValueError(f"{name} cannot be empty")

    for i, messages in enumerate(messages_list):
        if not messages:
            raise ValueError(f"{name}[{i}] is empty")
        if messages[-1].get("role") != "assistant":
            raise ValueError(
                f"{name}[{i}] must end with an assistant message, "
                f"got role='{messages[-1].get('role')}'"
            )


def _extract_response_activations_batch(
    backend: "ModelBackend",
    tokenizer,
    messages_list: List[List[dict]],
    layer: int,
    token_position: Literal["last_prompt_token", "mean"] = "mean",
) -> List[torch.Tensor]:
    """Extract response activations for multiple message lists."""
    return [
        extract_response_activations(backend, tokenizer, messages, layer, token_position)
        for messages in messages_list
    ]


def _extract_layer_activations(
    backend: "ModelBackend",
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """
    Extract layer output activations.

    Uses output hooks (forward hooks) to capture the layer's output,
    not input. This is important for difference-in-means extraction.
    """
    captured = []

    def capture_hook(module, args, output):
        # Output is typically a tuple (hidden_states, ...) or just hidden_states
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        captured.append(hidden_states.detach().clone())
        return output

    with backend.output_hooks_context([(layer, capture_hook)]):
        _ = backend.get_logits(input_ids)

    return captured[0].squeeze(0)  # [seq_len, hidden_dim]
