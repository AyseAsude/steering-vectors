"""Optimization components."""

from steering_vectors.optimization.optimizer import SteeringOptimizer
from steering_vectors.optimization.loss import (
    LossComponent,
    PromotionLoss,
    SuppressionLoss,
    CompositeLoss,
    SatisficingLoss,
)
from steering_vectors.optimization.callbacks import (
    OptimizationCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    HistoryCallback,
    NormConstraintCallback,
)
from steering_vectors.optimization.noise import (
    NoiseApplicator,
    gaussian_generator,
    antipgd_generator,
    identity_projector,
    tangent_space_projector,
    create_noise_applicator,
)

__all__ = [
    # Optimizer
    "SteeringOptimizer",
    # Loss components
    "LossComponent",
    "PromotionLoss",
    "SuppressionLoss",
    "CompositeLoss",
    "SatisficingLoss",
    # Callbacks
    "OptimizationCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "HistoryCallback",
    "NormConstraintCallback",
    # Noise
    "NoiseApplicator",
    "gaussian_generator",
    "antipgd_generator",
    "identity_projector",
    "tangent_space_projector",
    "create_noise_applicator",
]
