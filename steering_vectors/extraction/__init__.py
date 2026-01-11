"""
Steering vector extraction module.

This module provides a unified interface for extracting steering vectors
using different methods (CAA, gradient optimization, hybrid).

Example usage:
    >>> from steering_vectors.extraction import extract, ContrastPair
    >>>
    >>> # Define contrast pairs
    >>> pairs = [
    ...     ContrastPair(
    ...         positive={"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
    ...         negative={"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Go away."}]},
    ...     ),
    ... ]
    >>>
    >>> # Extract using CAA (default, recommended)
    >>> vector = extract(backend, tokenizer, pairs, layer=16, method="caa")
    >>>
    >>> # Or use gradient optimization
    >>> vector = extract(backend, tokenizer, pairs, layer=16, method="gradient")
"""

from steering_vectors.extraction.base import (
    VectorExtractor,
    ExtractionResult,
)
from steering_vectors.extraction.datapoint import ContrastPair
from steering_vectors.extraction.caa import CAAExtractor
from steering_vectors.extraction.gradient import GradientExtractor
from steering_vectors.extraction.factory import extract, create_extractor

__all__ = [
    # Base
    "VectorExtractor",
    "ExtractionResult",
    # Data
    "ContrastPair",
    # Extractors
    "CAAExtractor",
    "GradientExtractor",
    # Factory
    "extract",
    "create_extractor",
]
