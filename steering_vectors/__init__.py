"""
Steering Vectors: A research platform for LLM activation engineering.

Example usage (recommended - using CAA extraction):
    >>> from steering_vectors import extract, ContrastPair, HuggingFaceBackend
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

Legacy usage (gradient optimization):
    >>> from steering_vectors import (
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
from steering_vectors.core.datapoint import TrainingDatapoint
from steering_vectors.core.config import OptimizationConfig, ManifoldConfig
from steering_vectors.core.result import OptimizationResult

# Steering modes (how to apply vectors)
from steering_vectors.steering.base import SteeringMode
from steering_vectors.steering.vector import VectorSteering
from steering_vectors.steering.clamp import ClampSteering
from steering_vectors.steering.affine import AffineSteering

# Backends (model interfaces)
from steering_vectors.backends.base import ModelBackend
from steering_vectors.backends.huggingface import HuggingFaceBackend

# Extraction (recommended API)
from steering_vectors.extraction import (
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

# Optimization (legacy, use extraction module instead)
from steering_vectors.optimization.optimizer import SteeringOptimizer
from steering_vectors.optimization.loss import (
    LossComponent,
    PromotionLoss,
    SuppressionLoss,
    CompositeLoss,
    RegularizerComponent,
    ManifoldLoss,
)
from steering_vectors.optimization.callbacks import (
    OptimizationCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    HistoryCallback,
)

# Storage
from steering_vectors.storage.repository import VectorRepository, SQLiteRepository
from steering_vectors.storage.models import VectorMetadata

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
    # === Optimization (Legacy) ===
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
