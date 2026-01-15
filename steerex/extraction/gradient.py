"""
Gradient-based steering vector extractor.

This module wraps the SteeringOptimizer to provide a unified extraction
interface compatible with the VectorExtractor protocol.

Note: Gradient optimization can find arbitrary directions that don't
correspond to the target behavior. Consider using CAAExtractor first,
or use GradientExtractor with init_from_caa=True to start from the
CAA direction and refine it.
"""

from typing import Any, List, Optional, TYPE_CHECKING

import torch

from steerex.extraction.base import VectorExtractor, ExtractionResult
from steerex.extraction.datapoint import ContrastPair
from steerex.core.config import OptimizationConfig
from steerex.core.datapoint import TrainingDatapoint
from steerex.steering.vector import VectorSteering
from steerex.optimization.optimizer import SteeringOptimizer
from steerex.optimization.callbacks import OptimizationCallback
from steerex.optimization.loss import RegularizerComponent

if TYPE_CHECKING:
    from steerex.backends.base import ModelBackend


class GradientExtractor(VectorExtractor):
    """
    Gradient-based steering vector extractor.

    Uses gradient descent to find a vector that increases probability
    of positive completions and decreases probability of negative ones.

    Warning:
        Gradient optimization can find arbitrary local minima that don't
        correspond to the target behavior. For more reliable results:
        - Use init_from_caa=True to initialize from CAA direction
        - Use many contrast pairs (50+ recommended)
        - Add manifold regularization

    Attributes:
        config: Optimization hyperparameters.
        init_from_caa: If True, initialize from CAA direction before optimizing.
        callbacks: Optional optimization callbacks.
        regularizer: Optional regularization component.
        regularizer_weight: Weight for regularization term.

    Example:
        >>> extractor = GradientExtractor(
        ...     config=OptimizationConfig(lr=0.1, max_iters=50),
        ...     init_from_caa=True,  # Recommended
        ... )
        >>> result = extractor.extract(backend, tokenizer, pairs, layer=16)
    """

    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        init_from_caa: bool = False,
        callbacks: Optional[List[OptimizationCallback]] = None,
        regularizer: Optional[RegularizerComponent] = None,
        regularizer_weight: float = 1.0,
    ):
        """
        Initialize gradient extractor.

        Args:
            config: Optimization configuration.
            init_from_caa: If True, initialize from CAA direction.
            callbacks: Optional optimization callbacks.
            regularizer: Optional regularization component.
            regularizer_weight: Weight for regularization term.
        """
        self.config = config or OptimizationConfig()
        self.init_from_caa = init_from_caa
        self.callbacks = callbacks
        self.regularizer = regularizer
        self.regularizer_weight = regularizer_weight

    @property
    def method_name(self) -> str:
        return "gradient"

    def extract(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        pairs: List[ContrastPair],
        layer: int,
    ) -> ExtractionResult:
        """
        Extract steering vector using gradient optimization.

        Args:
            backend: Model backend for forward passes.
            tokenizer: Tokenizer with chat template support.
            pairs: List of positive/negative contrast pairs.
            layer: Layer to extract the vector for.

        Returns:
            ExtractionResult containing the optimized vector.
        """
        self.validate_pairs(pairs)

        # Convert ContrastPairs to TrainingDatapoints
        datapoints = self._convert_to_datapoints(pairs, tokenizer)

        # Create steering mode
        steering = VectorSteering()

        # Optionally initialize from CAA
        if self.init_from_caa:
            caa_vector = self._compute_caa_init(backend, tokenizer, pairs, layer)
            steering.vector = caa_vector.clone().requires_grad_(True)
        else:
            steering.init_parameters(
                hidden_dim=backend.get_hidden_dim(),
                device=backend.get_device(),
                dtype=torch.float32,
                starting_norm=self.config.starting_norm,
            )

        # Run optimization
        optimizer = SteeringOptimizer(
            backend=backend,
            steering_mode=steering,
            config=self.config,
            callbacks=self.callbacks,
            regularizer=self.regularizer,
            regularizer_weight=self.regularizer_weight,
        )

        result = optimizer.optimize(datapoints, layer=layer)

        return ExtractionResult(
            vector=result.vector,
            layer=layer,
            method=self.method_name,
            metadata={
                "iterations": result.iterations,
                "final_loss": result.final_loss,
                "init_from_caa": self.init_from_caa,
                "num_pairs": len(pairs),
                "config": self.config.model_dump(),
            },
        )

    def _convert_to_datapoints(
        self,
        pairs: List[ContrastPair],
        tokenizer: Any,
    ) -> List[TrainingDatapoint]:
        """Convert ContrastPairs to TrainingDatapoints."""
        datapoints = []

        for pair in pairs:
            if pair.format == "messages":
                # Convert messages to prompt + completion
                pos_messages = pair.get_positive_messages()
                neg_messages = pair.get_negative_messages()

                # Prompt is everything except assistant response
                prompt_messages = pos_messages[:-1]
                prompt = tokenizer.apply_chat_template(
                    prompt_messages, tokenize=False, add_generation_prompt=True
                )

                # Completions are the assistant responses
                pos_full = tokenizer.apply_chat_template(
                    pos_messages, tokenize=False, add_generation_prompt=False
                )
                neg_full = tokenizer.apply_chat_template(
                    neg_messages, tokenize=False, add_generation_prompt=False
                )

                pos_completion = pos_full[len(prompt):]
                neg_completion = neg_full[len(prompt):]

            else:
                prompt, pos_completion = pair.get_positive_prompt_completion()
                _, neg_completion = pair.get_negative_prompt_completion()

            datapoints.append(
                TrainingDatapoint(
                    prompt=prompt,
                    dst_completions=[pos_completion],
                    src_completions=[neg_completion],
                )
            )

        return datapoints

    def _compute_caa_init(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        pairs: List[ContrastPair],
        layer: int,
    ) -> torch.Tensor:
        """Compute CAA vector for initialization."""
        from steerex.extraction.caa import CAAExtractor

        caa = CAAExtractor(token_position="mean")
        result = caa.extract(backend, tokenizer, pairs, layer)

        # Normalize to starting_norm
        vector = result.vector
        current_norm = vector.norm()
        if current_norm > 0:
            vector = vector * (self.config.starting_norm / current_norm)

        return vector.to(torch.float32)
