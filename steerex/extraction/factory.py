"""
Factory functions for steering vector extraction.

This module provides convenient factory functions for creating extractors
and extracting vectors with a simple API.
"""

from typing import Any, Dict, List, Literal, Optional, TYPE_CHECKING, Union

from steerex.extraction.base import VectorExtractor, ExtractionResult
from steerex.extraction.datapoint import ContrastPair
from steerex.extraction.caa import CAAExtractor
from steerex.extraction.gradient import GradientExtractor
from steerex.core.config import OptimizationConfig

if TYPE_CHECKING:
    from steerex.backends.base import ModelBackend


# Type alias for extraction methods
ExtractionMethod = Literal["caa", "gradient", "hybrid"]


def create_extractor(
    method: ExtractionMethod = "caa",
    **kwargs,
) -> VectorExtractor:
    """
    Create a vector extractor for the specified method.

    Args:
        method: Extraction method to use.
            - "caa": Contrastive Activation Addition (default, recommended).
            - "gradient": Gradient-based optimization.
            - "hybrid": CAA initialization + gradient refinement.
        **kwargs: Method-specific arguments.
            For CAA:
                - token_position: "mean", "last", or "last_prompt_token"
                - remove_outliers: bool, remove extreme outliers before averaging
                - outlier_std_threshold: float, std dev threshold for outliers

    Returns:
        VectorExtractor instance.

    Raises:
        ValueError: If method is unknown.

    Example:
        >>> # Create CAA extractor (default)
        >>> extractor = create_extractor("caa", token_position="mean")
        >>>
        >>> # Create CAA with outlier removal
        >>> extractor = create_extractor(
        ...     "caa",
        ...     token_position="mean",
        ...     remove_outliers=True,
        ...     outlier_std_threshold=3.0,
        ... )
        >>>
        >>> # Create gradient extractor with custom config
        >>> extractor = create_extractor(
        ...     "gradient",
        ...     config=OptimizationConfig(lr=0.1),
        ... )
        >>>
        >>> # Create hybrid extractor (CAA init + gradient refinement)
        >>> extractor = create_extractor("hybrid", max_iters=30)
    """
    if method == "caa":
        return CAAExtractor(
            token_position=kwargs.get("token_position", "mean"),
            remove_outliers=kwargs.get("remove_outliers", False),
            outlier_std_threshold=kwargs.get("outlier_std_threshold", 3.0),
        )

    elif method == "gradient":
        config = kwargs.get("config") or OptimizationConfig(
            lr=kwargs.get("lr", 0.1),
            max_iters=kwargs.get("max_iters", 50),
            starting_norm=kwargs.get("starting_norm", 1.0),
            max_norm=kwargs.get("max_norm", 5.0),
        )
        return GradientExtractor(
            config=config,
            init_from_caa=kwargs.get("init_from_caa", False),
            callbacks=kwargs.get("callbacks"),
            regularizer=kwargs.get("regularizer"),
            regularizer_weight=kwargs.get("regularizer_weight", 1.0),
        )

    elif method == "hybrid":
        # Hybrid = gradient optimization initialized from CAA
        config = kwargs.get("config") or OptimizationConfig(
            lr=kwargs.get("lr", 0.1),
            max_iters=kwargs.get("max_iters", 30),
            starting_norm=kwargs.get("starting_norm", 1.0),
            max_norm=kwargs.get("max_norm", 5.0),
        )
        return GradientExtractor(
            config=config,
            init_from_caa=True,  # Key difference from pure gradient
            callbacks=kwargs.get("callbacks"),
            regularizer=kwargs.get("regularizer"),
            regularizer_weight=kwargs.get("regularizer_weight", 1.0),
        )

    else:
        raise ValueError(
            f"Unknown extraction method: {method}. "
            f"Choose from: 'caa', 'gradient', 'hybrid'"
        )


def extract(
    backend: "ModelBackend",
    tokenizer: Any,
    pairs: List[ContrastPair],
    layer: int,
    method: ExtractionMethod = "caa",
    normalize: Optional[float] = None,
    **kwargs,
) -> ExtractionResult:
    """
    Extract a steering vector from contrast pairs.

    This is the main entry point for vector extraction. It creates an
    appropriate extractor and runs extraction in one call.

    Args:
        backend: Model backend for forward passes.
        tokenizer: Tokenizer with chat template support.
        pairs: List of positive/negative contrast pairs.
        layer: Layer to extract the vector for.
        method: Extraction method (default: "caa").
            - "caa": Fast, reliable, recommended for most cases.
            - "gradient": Slower, can overfit, use with many pairs.
            - "hybrid": Best of both - CAA init + gradient refinement.
        normalize: If set, normalize vector to this L2 norm.
        **kwargs: Method-specific arguments.

    Returns:
        ExtractionResult containing the vector and metadata.

    Example:
        >>> # Simple CAA extraction
        >>> result = extract(backend, tokenizer, pairs, layer=16)
        >>> steering = result.to_steering()
        >>>
        >>> # Gradient with normalization
        >>> result = extract(
        ...     backend, tokenizer, pairs, layer=16,
        ...     method="gradient",
        ...     normalize=1.0,
        ... )
        >>>
        >>> # Hybrid approach
        >>> result = extract(
        ...     backend, tokenizer, pairs, layer=16,
        ...     method="hybrid",
        ...     max_iters=30,
        ... )
    """
    extractor = create_extractor(method, **kwargs)
    result = extractor.extract(backend, tokenizer, pairs, layer)

    if normalize is not None:
        result = result.normalize(normalize)

    return result


def extract_from_messages(
    backend: "ModelBackend",
    tokenizer: Any,
    positive_messages: List[List[Dict[str, str]]],
    negative_messages: List[List[Dict[str, str]]],
    layer: int,
    method: ExtractionMethod = "caa",
    **kwargs,
) -> ExtractionResult:
    """
    Extract a steering vector from chat message pairs.

    Convenience function that converts message lists to ContrastPairs
    and calls extract().

    Args:
        backend: Model backend for forward passes.
        tokenizer: Tokenizer with chat template support.
        positive_messages: List of message lists for positive examples.
        negative_messages: List of message lists for negative examples.
        layer: Layer to extract the vector for.
        method: Extraction method (default: "caa").
        **kwargs: Additional arguments passed to extract().

    Returns:
        ExtractionResult containing the vector and metadata.

    Example:
        >>> positive = [
        ...     [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
        ... ]
        >>> negative = [
        ...     [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Go away."}],
        ... ]
        >>> result = extract_from_messages(
        ...     backend, tokenizer, positive, negative, layer=16
        ... )
    """
    if len(positive_messages) != len(negative_messages):
        raise ValueError(
            f"positive_messages and negative_messages must have same length. "
            f"Got {len(positive_messages)} and {len(negative_messages)}"
        )

    pairs = [
        ContrastPair.from_messages(pos, neg)
        for pos, neg in zip(positive_messages, negative_messages)
    ]

    return extract(backend, tokenizer, pairs, layer, method=method, **kwargs)
