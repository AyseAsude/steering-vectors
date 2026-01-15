"""
Steering Vectors: A research platform for LLM activation engineering.

Example usage (recommended - using CAA extraction):
    >>> from steerex import extract, ContrastPair, HuggingFaceBackend
    >>>
    >>> # Setup backend
    >>> backend = HuggingFaceBackend(model, tokenizer)
    >>>
    >>> # Define contrast pairs
    >>> pairs = [
    ...     ContrastPair.from_messages(
    ...         positive=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
    ...         negative=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Go away."}],
    ...     ),
    ... ]
    >>>
    >>> # Extract using CAA (default, recommended)
    >>> result = extract(backend, tokenizer, pairs, layer=16)
    >>> steering = result.to_steering()
    >>>
    >>> # Use for generation
    >>> output = backend.generate_with_steering(
    ...     "Hello!",
    ...     steering_mode=steering,
    ...     layers=16,
    ... )

Gradient optimization approach:
    >>> from steerex import (
    ...     SteeringOptimizer,
    ...     VectorSteering,
    ...     HuggingFaceBackend,
    ...     TrainingDatapoint,
    ...     OptimizationConfig,
    ... )
    >>>
    >>> backend = HuggingFaceBackend(model, tokenizer)
    >>> steering = VectorSteering()
    >>> config = OptimizationConfig(lr=0.1, max_iters=50)
    >>> datapoints = [TrainingDatapoint(prompt="...", dst_completions=["..."])]
    >>> optimizer = SteeringOptimizer(backend, steering, config)
    >>> result = optimizer.optimize(datapoints, layer=16)
"""

# Core data types
from steerex.core.datapoint import TrainingDatapoint
from steerex.core.config import OptimizationConfig, ManifoldConfig
from steerex.core.result import OptimizationResult

# Steering modes (how to apply vectors)
from steerex.steering.base import SteeringMode
from steerex.steering.vector import VectorSteering
from steerex.steering.clamp import ClampSteering
from steerex.steering.affine import AffineSteering

# Backends (model interfaces)
from steerex.backends.base import ModelBackend
from steerex.backends.huggingface import HuggingFaceBackend

# Extraction (recommended API)
from steerex.extraction import (
    # Base classes
    VectorExtractor,
    ExtractionResult,
    # Data format
    ContrastPair,
    # Extractors
    CAAExtractor,
    GradientExtractor,
    # Factory functions
    extract,
    create_extractor,
)

# Optimization (gradient-based approach)
from steerex.optimization.optimizer import SteeringOptimizer
from steerex.optimization.loss import (
    LossComponent,
    PromotionLoss,
    SuppressionLoss,
    CompositeLoss,
    RegularizerComponent,
    ManifoldLoss,
)
from steerex.optimization.callbacks import (
    OptimizationCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    HistoryCallback,
)

# Storage
from steerex.storage.repository import VectorRepository, SQLiteRepository
from steerex.storage.models import VectorMetadata

__version__ = "0.2.0"

__all__ = [
    # === Extraction (Recommended API) ===
    # Factory functions
    "extract",
    "create_extractor",
    # Base classes
    "VectorExtractor",
    "ExtractionResult",
    # Data format
    "ContrastPair",
    # Extractors
    "CAAExtractor",
    "GradientExtractor",
    # === Steering Modes ===
    "SteeringMode",
    "VectorSteering",
    "ClampSteering",
    "AffineSteering",
    # === Backends ===
    "ModelBackend",
    "HuggingFaceBackend",
    # === Core Types ===
    "TrainingDatapoint",
    "OptimizationConfig",
    "OptimizationResult",
    "ManifoldConfig",
    # === Optimization ===
    "SteeringOptimizer",
    "LossComponent",
    "PromotionLoss",
    "SuppressionLoss",
    "CompositeLoss",
    "RegularizerComponent",
    "ManifoldLoss",
    # Callbacks
    "OptimizationCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "HistoryCallback",
    # === Storage ===
    "VectorRepository",
    "SQLiteRepository",
    "VectorMetadata",
]
