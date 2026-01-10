"""Difference-in-Means steering vector extraction."""

from typing import Callable, List, Optional, Union, TYPE_CHECKING

import torch

from steering_vectors.steering.base import SteeringMode
from steering_vectors.core.types import TokenSpec

if TYPE_CHECKING:
    from steering_vectors.backends.base import ModelBackend


class DifferenceInMeansSteering(SteeringMode):
    """
    Steering vector computed as the difference between mean activations.

    Formula: v = mean(positive_activations) - mean(negative_activations)

    Unlike other steering modes that learn vectors through optimization,
    this mode extracts the vector directly from contrastive examples
    in a single forward pass.

    Once computed, the vector can be used for steering like VectorSteering.

    Example:
        >>> steering = DifferenceInMeansSteering.from_contrastive_pairs(
        ...     backend,
        ...     positive_texts=["I love this!", "This is great!"],
        ...     negative_texts=["I hate this!", "This is terrible!"],
        ...     layer=16,
        ... )
        >>> output = backend.generate_with_steering(
        ...     "The movie was", steering, layers=16
        ... )
    """

    def __init__(self, vector: Optional[torch.Tensor] = None):
        """
        Initialize with optional pre-computed vector.

        Args:
            vector: Pre-computed steering vector. If None,
                use from_contrastive_pairs() to compute it.
        """
        self.vector = vector

    @classmethod
    def from_contrastive_pairs(
        cls,
        backend: "ModelBackend",
        positive_texts: List[str],
        negative_texts: List[str],
        layer: int,
        token_position: Union[int, str] = "last",
    ) -> "DifferenceInMeansSteering":
        """
        Compute steering vector from contrastive text pairs.

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

        # Extract activations for positive texts
        positive_activations = []
        for text in positive_texts:
            activation = _extract_activation(
                backend, text, layer, token_position
            )
            positive_activations.append(activation)

        # Extract activations for negative texts
        negative_activations = []
        for text in negative_texts:
            activation = _extract_activation(
                backend, text, layer, token_position
            )
            negative_activations.append(activation)

        # Stack and compute means
        positive_mean = torch.stack(positive_activations).mean(dim=0)
        negative_mean = torch.stack(negative_activations).mean(dim=0)

        # Compute difference
        vector = positive_mean - negative_mean

        return cls(vector=vector)

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
        vector = positive_mean - negative_mean
        return cls(vector=vector)

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
        from_contrastive_pairs() instead. This method exists for
        interface compatibility.
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
        """
        Return parameters for optimization.

        Note: The vector from difference-in-means is not typically
        optimized further, but this method allows for fine-tuning
        if desired.
        """
        if self.vector is None:
            raise ValueError(
                "Vector not initialized. Use from_contrastive_pairs() "
                "or init_parameters() first."
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


def _extract_activation(
    backend: "ModelBackend",
    text: str,
    layer: int,
    token_position: Union[int, str],
) -> torch.Tensor:
    """
    Extract activation at specified layer and token position.

    Args:
        backend: Model backend.
        text: Input text.
        layer: Layer index.
        token_position: "last", "mean", or int index.

    Returns:
        Activation tensor of shape [hidden_dim].
    """
    input_ids = backend.tokenize(text)
    activations = _extract_layer_activations(backend, input_ids, layer)

    # activations shape: [seq_len, hidden_dim]
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
    """Extract activations at a specific layer."""
    captured = []

    def capture_hook(module, args):
        hidden_states = args[0]
        captured.append(hidden_states.detach().clone())
        return args

    with backend.hooks_context([(layer, capture_hook)]):
        _ = backend.get_logits(input_ids)

    return captured[0].squeeze(0)  # [seq_len, hidden_dim]
