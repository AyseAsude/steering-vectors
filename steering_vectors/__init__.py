"""
Steering Vectors: A research platform for LLM activation engineering.

Example usage:
    from steering_vectors import (
        SteeringOptimizer,
        VectorSteering,
        HuggingFaceBackend,
        TrainingDatapoint,
        OptimizationConfig,
    )

    # Setup
    backend = HuggingFaceBackend(model, tokenizer)
    steering = VectorSteering()
    config = OptimizationConfig(lr=0.1, max_iters=50)

    # Define training data
    datapoints = [
        TrainingDatapoint(
            prompt="My favorite animal is",
            dst_completions=[" definitely cats!"],
        )
    ]

    # Optimize
    optimizer = SteeringOptimizer(backend, steering, config)
    result = optimizer.optimize(datapoints, layer=16)

    # Use the vector
    steered_text = backend.generate_with_steering(
        "My favorite animal is",
        steering_mode=steering,
        layer=16,
    )
"""

from steering_vectors.core.datapoint import TrainingDatapoint
from steering_vectors.core.config import OptimizationConfig, ManifoldConfig
from steering_vectors.core.result import OptimizationResult

from steering_vectors.steering.base import SteeringMode
from steering_vectors.steering.vector import VectorSteering
from steering_vectors.steering.clamp import ClampSteering
from steering_vectors.steering.affine import AffineSteering

from steering_vectors.backends.base import ModelBackend
from steering_vectors.backends.huggingface import HuggingFaceBackend

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

from steering_vectors.storage.repository import VectorRepository, SQLiteRepository
from steering_vectors.storage.models import VectorMetadata

__version__ = "0.1.0"

__all__ = [
    # Core
    "TrainingDatapoint",
    "OptimizationConfig",
    "OptimizationResult",
    "ManifoldConfig",
    # Steering modes
    "SteeringMode",
    "VectorSteering",
    "ClampSteering",
    "AffineSteering",
    # Backends
    "ModelBackend",
    "HuggingFaceBackend",
    # Optimization
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
    # Storage
    "VectorRepository",
    "SQLiteRepository",
    "VectorMetadata",
]
