"""
Abstract base class for vector extractors.

This module defines the VectorExtractor interface that all extraction
methods must implement, following the Strategy pattern.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from steerex.backends.base import ModelBackend
    from steerex.extraction.datapoint import ContrastPair


@dataclass
class ExtractionResult:
    """
    Result of vector extraction.

    Attributes:
        vector: The extracted steering vector (hidden_dim,).
        layer: The layer the vector was extracted for.
        method: The extraction method used ("caa", "gradient", etc.).
        metadata: Additional method-specific information.
    """

    vector: torch.Tensor
    layer: int
    method: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def normalize(self, norm: float = 1.0) -> "ExtractionResult":
        """
        Return a new result with normalized vector.

        Args:
            norm: Target L2 norm.

        Returns:
            New ExtractionResult with normalized vector.
        """
        current_norm = self.vector.norm()
        if current_norm > 0:
            normalized = self.vector * (norm / current_norm)
        else:
            normalized = self.vector.clone()

        return ExtractionResult(
            vector=normalized,
            layer=self.layer,
            method=self.method,
            metadata={**self.metadata, "normalized_to": norm},
        )

    def to_steering(self) -> "VectorSteering":
        """
        Convert to a VectorSteering instance for use with backends.

        Returns:
            VectorSteering instance with this vector.
        """
        from steerex.steering.vector import VectorSteering

        return VectorSteering(vector=self.vector.detach().clone())


class VectorExtractor(ABC):
    """
    Abstract base class for steering vector extraction strategies.

    All extraction methods (CAA, gradient, hybrid, etc.) implement this
    interface, allowing them to be used interchangeably.

    The Strategy pattern allows:
    - Easy switching between extraction methods
    - Adding new methods without modifying existing code
    - Consistent API across all methods
    """

    @property
    @abstractmethod
    def method_name(self) -> str:
        """Return the name of this extraction method."""
        ...

    @abstractmethod
    def extract(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        pairs: List["ContrastPair"],
        layer: int,
    ) -> ExtractionResult:
        """
        Extract a steering vector from contrast pairs.

        Args:
            backend: Model backend for forward passes.
            tokenizer: Tokenizer with chat template support.
            pairs: List of positive/negative contrast pairs.
            layer: Layer to extract the vector for.

        Returns:
            ExtractionResult containing the vector and metadata.
        """
        ...

    def validate_pairs(self, pairs: List["ContrastPair"]) -> None:
        """
        Validate that pairs are properly formatted.

        Args:
            pairs: List of contrast pairs to validate.

        Raises:
            ValueError: If pairs are invalid.
        """
        if not pairs:
            raise ValueError("At least one contrast pair is required")

        for i, pair in enumerate(pairs):
            pair.validate(index=i)
