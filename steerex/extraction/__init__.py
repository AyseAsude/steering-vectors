"""
Steering vector extraction module.

This module provides a unified interface for extracting steering vectors
using different methods (CAA, gradient optimization, hybrid).

Example usage:
    >>> from steerex.extraction import extract, ContrastPair
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

from steerex.extraction.base import (
    VectorExtractor,
    ExtractionResult,
)
from steerex.extraction.datapoint import ContrastPair
from steerex.extraction.caa import CAAExtractor
from steerex.extraction.gradient import GradientExtractor
from steerex.extraction.factory import extract, create_extractor

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
