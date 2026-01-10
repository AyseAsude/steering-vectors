"""Difference-in-Means steering vector extraction.

This module provides methods for extracting steering vectors by computing
the difference between mean activations of contrastive examples.

Formula: v = mean(positive_activations) - mean(negative_activations)
"""

from typing import Callable, List, Optional, Union, TYPE_CHECKING

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

    Example with plain texts:
        >>> steering = DifferenceInMeansSteering.from_contrastive_texts(
        ...     backend,
        ...     positive_texts=["I love this!", "This is great!"],
        ...     negative_texts=["I hate this!", "This is terrible!"],
        ...     layer=16,
        ... )

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
    ) -> "DifferenceInMeansSteering":
        """
        Compute steering vector from contrastive chat message pairs.

        Extracts activations only from assistant response tokens, computing
        the mean across all response positions for each example.

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

        Returns:
            DifferenceInMeansSteering instance with computed vector.

        Raises:
            ValueError: If messages lists are empty or don't end with assistant.
        """
        _validate_message_lists(positive_messages, "positive_messages")
        _validate_message_lists(negative_messages, "negative_messages")

        positive_activations = _extract_response_activations_batch(
            backend, tokenizer, positive_messages, layer
        )
        negative_activations = _extract_response_activations_batch(
            backend, tokenizer, negative_messages, layer
        )

        positive_mean = torch.stack(positive_activations).mean(dim=0)
        negative_mean = torch.stack(negative_activations).mean(dim=0)

        return cls(vector=positive_mean - negative_mean)

    @classmethod
    def from_contrastive_texts(
        cls,
        backend: "ModelBackend",
        positive_texts: List[str],
        negative_texts: List[str],
        layer: int,
        token_position: Union[int, str] = "last",
    ) -> "DifferenceInMeansSteering":
        """
        Compute steering vector from contrastive plain text pairs.

        This is a simpler method for non-chat models or when you want to
        extract from the entire text without response boundary detection.

        Args:
            backend: Model backend for extracting activations.
            positive_texts: List of texts representing the positive concept.
            negative_texts: List of texts representing the negative concept.
            layer: Layer index to extract activations from.
            token_position: Which token position(s) to use for computing means.
                - "last": Use the last token (default)
                - "mean": Average over all tokens
                - int: Use a specific token index

        Returns:
            DifferenceInMeansSteering instance with computed vector.
        """
        if not positive_texts:
            raise ValueError("positive_texts cannot be empty")
        if not negative_texts:
            raise ValueError("negative_texts cannot be empty")

        positive_activations = [
            _extract_text_activation(backend, text, layer, token_position)
            for text in positive_texts
        ]
        negative_activations = [
            _extract_text_activation(backend, text, layer, token_position)
            for text in negative_texts
        ]

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
) -> torch.Tensor:
    """
    Extract mean activation from assistant response tokens only.

    Applies chat template, detects response boundary, and extracts
    activations from only the assistant's response portion.

    Args:
        backend: Model backend.
        tokenizer: Tokenizer with chat template.
        messages: Conversation messages ending with assistant response.
        layer: Layer index to extract from.

    Returns:
        Mean activation tensor of shape [hidden_dim].

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

    # Tokenize to find boundary
    full_ids = backend.tokenize(full_text)
    prompt_ids = backend.tokenize(prompt_text)

    prompt_len = prompt_ids.shape[1]
    full_len = full_ids.shape[1]

    if full_len <= prompt_len:
        raise ValueError(
            f"No response tokens found. prompt_len={prompt_len}, full_len={full_len}"
        )

    # Extract activations
    activations = _extract_layer_activations(backend, full_ids, layer)

    # Return mean of response tokens only (includes response + end token)
    response_activations = activations[prompt_len:full_len]
    return response_activations.mean(dim=0)


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
) -> List[torch.Tensor]:
    """Extract response activations for multiple message lists."""
    return [
        extract_response_activations(backend, tokenizer, messages, layer)
        for messages in messages_list
    ]


def _extract_text_activation(
    backend: "ModelBackend",
    text: str,
    layer: int,
    token_position: Union[int, str],
) -> torch.Tensor:
    """Extract activation at specified layer and token position from plain text."""
    input_ids = backend.tokenize(text)
    activations = _extract_layer_activations(backend, input_ids, layer)

    if token_position == "last":
        return activations[-1]
    elif token_position == "mean":
        return activations.mean(dim=0)
    elif isinstance(token_position, int):
        return activations[token_position]
    else:
        raise ValueError(
            f"Invalid token_position: {token_position}. "
            "Expected 'last', 'mean', or int."
        )


def _extract_layer_activations(
    backend: "ModelBackend",
    input_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Extract all activations at a specific layer."""
    captured = []

    def capture_hook(module, args):
        hidden_states = args[0]
        captured.append(hidden_states.detach().clone())
        return args

    with backend.hooks_context([(layer, capture_hook)]):
        _ = backend.get_logits(input_ids)

    return captured[0].squeeze(0)  # [seq_len, hidden_dim]
