"""
Main steering vector optimizer.

This module contains the core optimization loop for finding steering vectors.
A steering vector is a tensor that, when added to a model's hidden states at
a specific layer, changes the model's behavior in a targeted way.

The optimization works by:
1. Starting with a random vector
2. Running the model with this vector injected via hooks
3. Computing how likely the model is to produce target completions
4. Using gradient descent to adjust the vector
5. Repeating until the vector reliably induces the desired behavior
"""

from typing import List, Optional, Union, Tuple, Dict, Any, Callable

import torch

from steering_vectors.core.datapoint import TrainingDatapoint
from steering_vectors.core.config import OptimizationConfig
from steering_vectors.core.result import OptimizationResult
from steering_vectors.core.types import LayerSpec
from steering_vectors.backends.base import ModelBackend
from steering_vectors.steering.base import SteeringMode
from steering_vectors.optimization.loss import (
    LossComponent,
    PromotionLoss,
    SuppressionLoss,
)
from steering_vectors.optimization.callbacks import OptimizationCallback
from steering_vectors.optimization.noise import (
    NoiseApplicator,
    create_noise_applicator,
    identity_projector,
)


class SteeringOptimizer:
    """
    Main optimizer for finding steering vectors.

    This class orchestrates the optimization loop, delegating to:
    - ModelBackend: Handles model-specific operations (tokenization, forward pass)
    - SteeringMode: Defines how to modify activations (additive, clamp, affine)
    - LossComponent: Computes loss from model outputs
    - OptimizationCallback: Provides hooks for logging, early stopping, etc.

    The optimization process:
    1. Initialize a random steering vector
    2. For each iteration:
       a. Create hooks that add the vector to hidden states
       b. Run the model with hooks attached
       c. Compute loss: how far from desired behavior?
       d. Backpropagate to get gradients on the vector
       e. Update the vector using Adam optimizer
    3. Return the optimized vector

    Example:
        >>> backend = HuggingFaceBackend(model, tokenizer)
        >>> steering = VectorSteering()
        >>> config = OptimizationConfig(lr=0.1, max_iters=50)
        >>> optimizer = SteeringOptimizer(backend, steering, config)
        >>> result = optimizer.optimize(datapoints, layer=16)
    """

    def __init__(
        self,
        backend: ModelBackend,
        steering_mode: SteeringMode,
        config: Optional[OptimizationConfig] = None,
        callbacks: Optional[List[OptimizationCallback]] = None,
    ):
        """
        Initialize the optimizer.

        Args:
            backend: Model backend for forward passes (e.g., HuggingFaceBackend).
            steering_mode: Strategy for modifying activations (e.g., VectorSteering).
            config: Hyperparameters (learning rate, iterations, etc.).
            callbacks: Optional callbacks for logging, early stopping, etc.
        """
        self.backend = backend
        self.steering_mode = steering_mode
        self.config = config or OptimizationConfig()
        self.callbacks = callbacks or []

        # Create noise applicator from config (None if noise disabled)
        self._noise_applicator: Optional[NoiseApplicator] = create_noise_applicator(
            self.config.noise
        )

    def optimize(
        self,
        datapoints: List[TrainingDatapoint],
        layer: LayerSpec,
    ) -> OptimizationResult:
        """
        Optimize a steering vector for the given datapoints.

        This is the main entry point. It runs the optimization loop and
        returns the optimized vector.

        Args:
            datapoints: Training examples. Each specifies:
                - prompt: The input text
                - dst_completions: Completions to make MORE likely (promote)
                - src_completions: Completions to make LESS likely (suppress)
            layer: Which layer(s) to inject the steering vector at.

        Returns:
            OptimizationResult containing:
                - vector: The optimized steering vector
                - iterations: How many steps were taken
                - final_loss: The loss at the end
                - metadata: Config and layer info
        """
        # Normalize layer to list (supports single int or list of ints)
        layers = [layer] if isinstance(layer, int) else list(layer)

        # Reset noise state if using noisy steering
        if self._noise_applicator:
            self._noise_applicator.reset()

        # =====================================================================
        # STEP 1: Initialize the steering vector with random values
        # =====================================================================
        # The steering mode owns the parameter(s). For VectorSteering, this is
        # just a single vector of shape (hidden_dim,). For AffineSteering, it
        # includes a low-rank matrix as well.
        self.steering_mode.init_parameters(
            hidden_dim=self.backend.get_hidden_dim(),
            device=self.backend.get_device(),
            dtype=self.backend.get_dtype(),
            starting_norm=self.config.starting_norm,
        )

        # Get the parameters to optimize and set up Adam
        params = self.steering_mode.parameters()
        optimizer = torch.optim.Adam(params, lr=self.config.lr)

        # =====================================================================
        # STEP 2: Precompute tokenizations
        # =====================================================================
        # Tokenize all prompts and completions once, not every iteration.
        # This is a simple optimization to avoid repeated string processing.
        tokenized_data = self._precompute_tokens(datapoints)

        # Notify callbacks that we're starting
        for callback in self.callbacks:
            callback.on_optimization_start(params, self.config)

        # =====================================================================
        # STEP 3: Main optimization loop
        # =====================================================================
        final_loss = 0.0
        step = 0

        for step in range(self.config.max_iters):
            # Clear gradients from previous iteration
            optimizer.zero_grad()

            # -----------------------------------------------------------------
            # STEP 3a: Compute loss over all datapoints
            # -----------------------------------------------------------------
            # This runs the model with the steering hook, computes log probs
            # of target completions, and returns the total loss.
            total_loss, per_completion_losses = self._compute_batch_loss(
                datapoints, tokenized_data, layers
            )

            # -----------------------------------------------------------------
            # STEP 3b: Backpropagate to get gradients
            # -----------------------------------------------------------------
            # This computes d(loss)/d(vector) using the chain rule.
            # The gradient tells us: "which direction should we move the
            # vector to reduce the loss?"
            total_loss.backward()

            # -----------------------------------------------------------------
            # STEP 3c: Update the vector using Adam
            # -----------------------------------------------------------------
            # Adam adjusts each element of the vector based on the gradient,
            # with adaptive learning rates and momentum.
            optimizer.step()

            # -----------------------------------------------------------------
            # STEP 3d: Apply constraints (e.g., max norm)
            # -----------------------------------------------------------------
            # If max_norm is set, clip the vector to prevent it from growing
            # too large (which can cause weird model behavior).
            self.steering_mode.apply_constraints(self.config.max_norm)

            final_loss = total_loss.item()

            # Run callbacks (logging, early stopping, etc.)
            should_continue = self._run_callbacks(
                step, final_loss, params, per_completion_losses
            )
            if not should_continue:
                break

        # Notify callbacks that we're done
        for callback in self.callbacks:
            callback.on_optimization_end(params, final_loss, step + 1)

        # =====================================================================
        # STEP 4: Return the result
        # =====================================================================
        return OptimizationResult(
            vector=self.steering_mode.get_vector(),
            iterations=step + 1,
            final_loss=final_loss,
            metadata={
                "config": self.config.model_dump(),
                "layers": layers,
            },
        )

    def _precompute_tokens(
        self,
        datapoints: List[TrainingDatapoint],
    ) -> List[Dict[str, Any]]:
        """
        Precompute tokenizations for all datapoints.

        Tokenizing strings is relatively slow, so we do it once upfront
        instead of every iteration.

        Returns a list where each item contains:
            - prompt_ids: Tokenized prompt
            - prompt_len: Number of tokens in prompt
            - src_tokens: List of tokenized src_completions
            - dst_tokens: List of tokenized dst_completions
        """
        result = []

        for dp in datapoints:
            # Tokenize just the prompt to find its length
            prompt_ids = self.backend.tokenize(dp.prompt)
            prompt_len = prompt_ids.shape[1]

            # Tokenize each src_completion (things to suppress)
            src_tokens = []
            for completion in dp.src_completions:
                # Tokenize prompt + completion together
                full_ids = self.backend.tokenize(dp.prompt + completion)
                # Extract just the completion tokens (after the prompt)
                completion_ids = full_ids[0, prompt_len:]
                src_tokens.append({
                    "full_ids": full_ids,
                    "completion_ids": completion_ids,
                    "prompt_len": prompt_len,
                })

            # Tokenize each dst_completion (things to promote)
            dst_tokens = []
            for completion in dp.dst_completions:
                full_ids = self.backend.tokenize(dp.prompt + completion)
                completion_ids = full_ids[0, prompt_len:]
                dst_tokens.append({
                    "full_ids": full_ids,
                    "completion_ids": completion_ids,
                    "prompt_len": prompt_len,
                })

            result.append({
                "prompt_ids": prompt_ids,
                "prompt_len": prompt_len,
                "src_tokens": src_tokens,
                "dst_tokens": dst_tokens,
                "datapoint": dp,
            })

        return result

    def _compute_batch_loss(
        self,
        datapoints: List[TrainingDatapoint],
        tokenized_data: List[Dict],
        layers: List[int],
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Compute total loss over all datapoints.

        For each datapoint:
        1. Create a steering hook with the current vector
        2. Run the model with the hook attached
        3. For dst_completions: compute -log(P(completion)) [promote]
        4. For src_completions: compute -log(1-P(completion)) [suppress]
        5. Sum all losses

        If noise is enabled, each completion is evaluated multiple times
        (noise_iters) with different noise perturbations.

        The loss represents "how far are we from the desired behavior?"
        Lower loss = model behaves more as desired.

        Returns:
            Tuple of:
                - total_loss: Sum of all completion losses (for backprop)
                - per_completion_losses: Nested list of individual losses (for logging)
        """
        device = self.backend.get_device()

        # Initialize loss tensor that will accumulate all losses
        # requires_grad=True allows gradients to flow through
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        per_completion_losses: List[List[float]] = []

        # Create loss functions for promotion (increase prob) and suppression (decrease prob)
        promotion_loss = PromotionLoss(
            normalize_by_length=self.config.normalize_by_length
        )
        suppression_loss = SuppressionLoss(
            use_one_minus=self.config.use_one_minus,
            normalize_by_length=self.config.normalize_by_length,
        )

        # Determine number of noise iterations
        noise_iters = (
            self._noise_applicator.noise_iters if self._noise_applicator else 1
        )

        # Check if we need gradient for tangent space projection
        needs_gradient = (
            self._noise_applicator is not None
            and self._noise_applicator.projector != identity_projector()
        )

        # Process each datapoint
        for dp_idx, (dp, tokens) in enumerate(zip(datapoints, tokenized_data)):
            dp_losses: List[List[float]] = [[], []]  # [src_losses, dst_losses]

            # -----------------------------------------------------------------
            # Suppression: make src_completions LESS likely
            # -----------------------------------------------------------------
            for comp_idx, comp_tokens in enumerate(tokens["src_tokens"]):
                comp_loss_sum = 0.0

                for _ in range(noise_iters):
                    # Get vector, possibly with noise applied
                    vector = self._get_vector_for_loss(
                        comp_tokens,
                        layers,
                        suppression_loss,
                        is_src=True,
                        needs_gradient=needs_gradient,
                    )

                    # Create hook with this vector
                    vector_sign = -1 if dp.negate else 1
                    hook = self._create_hook_with_vector(
                        vector, dp.token_slice, vector_sign
                    )
                    hooks = [(layer, hook) for layer in layers]

                    # Compute loss
                    with self.backend.hooks_context(hooks):
                        logits = self.backend.get_logits(comp_tokens["full_ids"])[0]

                    loss = suppression_loss.compute(
                        logits,
                        comp_tokens["full_ids"][0],
                        comp_tokens["prompt_len"],
                        coldness=self.config.coldness,
                    )

                    # Satisficing: aim for target value
                    if self.config.satisfice and dp.src_target_losses:
                        target = dp.src_target_losses[comp_idx]
                        loss = (loss - target) ** 2

                    total_loss = total_loss + loss
                    comp_loss_sum += loss.item()

                # Store average loss for this completion
                dp_losses[0].append(comp_loss_sum / noise_iters)

            # -----------------------------------------------------------------
            # Promotion: make dst_completions MORE likely
            # -----------------------------------------------------------------
            for comp_idx, comp_tokens in enumerate(tokens["dst_tokens"]):
                comp_loss_sum = 0.0

                for _ in range(noise_iters):
                    # Get vector, possibly with noise applied
                    vector = self._get_vector_for_loss(
                        comp_tokens,
                        layers,
                        promotion_loss,
                        is_src=False,
                        needs_gradient=needs_gradient,
                    )

                    # Create hook with this vector
                    vector_sign = -1 if dp.negate else 1
                    hook = self._create_hook_with_vector(
                        vector, dp.token_slice, vector_sign
                    )
                    hooks = [(layer, hook) for layer in layers]

                    # Compute loss
                    with self.backend.hooks_context(hooks):
                        logits = self.backend.get_logits(comp_tokens["full_ids"])[0]

                    loss = promotion_loss.compute(
                        logits,
                        comp_tokens["full_ids"][0],
                        comp_tokens["prompt_len"],
                        coldness=self.config.coldness,
                    )

                    # Satisficing: aim for target value
                    if self.config.satisfice and dp.dst_target_losses:
                        target = dp.dst_target_losses[comp_idx]
                        loss = (loss - target) ** 2

                    total_loss = total_loss + loss
                    comp_loss_sum += loss.item()

                # Store average loss for this completion
                dp_losses[1].append(comp_loss_sum / noise_iters)

            per_completion_losses.append(dp_losses)

        return total_loss, per_completion_losses

    def _get_vector_for_loss(
        self,
        comp_tokens: Dict,
        layers: List[int],
        loss_fn: LossComponent,
        is_src: bool,
        needs_gradient: bool,
    ) -> torch.Tensor:
        """
        Get the steering vector, optionally with noise applied.

        If noise is enabled, applies noise to the vector.
        If tangent space projection is enabled, computes gradient first.

        Args:
            comp_tokens: Tokenized completion data.
            layers: Layers to apply steering to.
            loss_fn: Loss function (for gradient computation).
            is_src: Whether this is a suppression (src) completion.
            needs_gradient: Whether gradient is needed for noise projection.

        Returns:
            The (possibly noisy) steering vector.
        """
        vector = self.steering_mode.get_vector()

        if self._noise_applicator is None:
            return vector

        gradient = None
        if needs_gradient:
            gradient = self._compute_gradient_for_noise(
                comp_tokens, layers, loss_fn
            )

        return self._noise_applicator.apply(vector, gradient)

    def _create_hook_with_vector(
        self,
        vector: torch.Tensor,
        token_slice: Any,
        strength: float,
    ) -> Callable:
        """
        Create a steering hook that uses a specific vector.

        This is needed for noisy steering where we want to inject
        a modified vector rather than the one stored in steering_mode.

        Args:
            vector: The vector to inject.
            token_slice: Which tokens to apply steering to.
            strength: Multiplier for the vector (e.g., -1 for negation).

        Returns:
            A hook function for forward_pre_hook.
        """
        idx = token_slice if token_slice is not None else slice(None)

        def hook_fn(module, args):
            hidden_states = args[0]  # Shape: [batch, seq_len, hidden_dim]
            modified = hidden_states.clone()
            modified[:, idx] = modified[:, idx] + strength * vector.to(
                modified.device, modified.dtype
            )
            return (modified,) + args[1:]

        return hook_fn

    def _compute_gradient_for_noise(
        self,
        comp_tokens: Dict,
        layers: List[int],
        loss_fn: LossComponent,
    ) -> torch.Tensor:
        """
        Compute gradient of unsteered loss for tangent space projection.

        This runs a forward pass with a zero vector and computes the gradient
        of the loss with respect to that vector. This gradient represents
        the "optimization direction" that noise should avoid.

        Args:
            comp_tokens: Tokenized completion data.
            layers: Layers for steering.
            loss_fn: Loss function to use.

        Returns:
            Gradient tensor of shape (hidden_dim,).
        """
        # Create a zero vector that requires grad
        zero_vec = torch.zeros(
            self.backend.get_hidden_dim(),
            device=self.backend.get_device(),
            dtype=self.backend.get_dtype(),
            requires_grad=True,
        )

        # Forward pass with zero steering
        hook = self._create_hook_with_vector(zero_vec, None, 1.0)
        hooks = [(layers[0], hook)]

        with self.backend.hooks_context(hooks):
            logits = self.backend.get_logits(comp_tokens["full_ids"])[0]

        # Compute loss
        loss = loss_fn.compute(
            logits,
            comp_tokens["full_ids"][0],
            comp_tokens["prompt_len"],
            coldness=self.config.coldness,
        )

        # Get gradient
        grad = torch.autograd.grad(loss, zero_vec)[0]
        return grad.detach()

    def _run_callbacks(
        self,
        step: int,
        loss: float,
        parameters: List[torch.Tensor],
        per_completion_losses: List[List[float]],
    ) -> bool:
        """
        Run all callbacks and return whether to continue.

        Callbacks can signal early stopping by returning False from on_step_end.
        """
        extra = {"per_completion_losses": per_completion_losses}

        for callback in self.callbacks:
            if not callback.on_step_end(step, loss, parameters, extra):
                return False
        return True
