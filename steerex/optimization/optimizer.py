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

Performance optimizations:
- Batched forward passes: Multiple completions in single forward pass
- Grouped by steering config: Completions with same hooks are batched together
- Configurable batch sizes: Balance memory vs speed
"""

from collections import defaultdict
from typing import List, Optional, Union, Tuple, Dict, Any

import torch

from steerex.core.datapoint import TrainingDatapoint
from steerex.core.config import OptimizationConfig
from steerex.core.result import OptimizationResult
from steerex.core.types import LayerSpec
from steerex.backends.base import ModelBackend
from steerex.steering.base import SteeringMode
from steerex.optimization.loss import (
    LossComponent,
    PromotionLoss,
    SuppressionLoss,
    RegularizerComponent,
)
from steerex.optimization.callbacks import OptimizationCallback


# Default batch size for batched optimization
DEFAULT_BATCH_SIZE = 16


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
        regularizer: Optional[RegularizerComponent] = None,
        regularizer_weight: float = 1.0,
    ):
        """
        Initialize the optimizer.

        Args:
            backend: Model backend for forward passes (e.g., HuggingFaceBackend).
            steering_mode: Strategy for modifying activations (e.g., VectorSteering).
            config: Hyperparameters (learning rate, iterations, etc.).
            callbacks: Optional callbacks for logging, early stopping, etc.
            regularizer: Optional regularization component (e.g., ManifoldLoss).
            regularizer_weight: Weight for the regularization term (lambda).
        """
        self.backend = backend
        self.steering_mode = steering_mode
        self.config = config or OptimizationConfig()
        self.callbacks = callbacks or []
        self.regularizer = regularizer
        self.regularizer_weight = regularizer_weight

    def optimize(
        self,
        datapoints: List[TrainingDatapoint],
        layer: LayerSpec,
    ) -> OptimizationResult:
        """
        Optimize a steering vector for the given datapoints.

        This is the main entry point. It runs the optimization loop and
        returns the optimized vector.

        By default, uses batched forward passes for 10-50x faster optimization.
        Set config.use_batched=False to use sequential (original) method.

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
        # Use batched optimization if enabled and backend supports it
        if self.config.use_batched and hasattr(self.backend, 'pad_sequences'):
            return self.optimize_batched(
                datapoints,
                layer,
                batch_size=self.config.batch_size,
            )

        # Normalize layer to list (supports single int or list of ints)
        layers = [layer] if isinstance(layer, int) else list(layer)

        # =====================================================================
        # STEP 1: Initialize the steering vector with random values
        # =====================================================================
        # The steering mode owns the parameter(s). For VectorSteering, this is
        # just a single vector of shape (hidden_dim,). For AffineSteering, it
        # includes a low-rank matrix as well.
        # Always use float32 for steering parameters for numerical stability.
        # Half-precision (float16/bfloat16) can cause issues with gradients.
        self.steering_mode.init_parameters(
            hidden_dim=self.backend.get_hidden_dim(),
            device=self.backend.get_device(),
            dtype=torch.float32,
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
            # STEP 3a.5: Add regularization term (if configured)
            # -----------------------------------------------------------------
            # Regularizers like ManifoldLoss add constraints on the vector
            # itself (e.g., staying on the manifold of natural activations).
            if self.regularizer is not None:
                reg_loss = self.regularizer.compute(self.steering_mode)
                total_loss = total_loss + self.regularizer_weight * reg_loss

            # -----------------------------------------------------------------
            # STEP 3b: Backpropagate to get gradients
            # -----------------------------------------------------------------
            # This computes d(loss)/d(vector) using the chain rule.
            # The gradient tells us: "which direction should we move the
            # vector to reduce the loss?"
            total_loss.backward()

            # -----------------------------------------------------------------
            # STEP 3b.5: Clip gradients to prevent NaN from explosion
            # -----------------------------------------------------------------
            # When using log(1-p) loss and p→1, gradients can explode.
            # Clipping prevents this from corrupting the optimization.
            if self.config.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.grad_clip_value)

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
            normalize_by_length=self.config.normalize_by_length,
            eps=self.config.loss_eps,
        )
        suppression_loss = SuppressionLoss(
            use_one_minus=self.config.use_one_minus,
            normalize_by_length=self.config.normalize_by_length,
            eps=self.config.loss_eps,
        )

        # Process each datapoint
        for dp_idx, (dp, tokens) in enumerate(zip(datapoints, tokenized_data)):
            dp_losses = [[], []]  # [src_losses, dst_losses]

            # -----------------------------------------------------------------
            # Create the steering hook for this datapoint
            # -----------------------------------------------------------------
            # If negate=True, we flip the vector sign (useful for contrastive training)
            vector_sign = -1 if dp.negate else 1
            hook = self.steering_mode.create_hook(
                token_slice=dp.token_slice,
                strength=float(vector_sign),
            )
            # Apply the same hook to all specified layers
            hooks = [(layer, hook) for layer in layers]

            # -----------------------------------------------------------------
            # Suppression: make src_completions LESS likely
            # -----------------------------------------------------------------
            # For each completion we want to suppress, we compute loss that
            # penalizes high probability of that completion.
            for comp_idx, comp_tokens in enumerate(tokens["src_tokens"]):
                # Run model with steering hook attached
                with self.backend.hooks_context(hooks):
                    logits = self.backend.get_logits(comp_tokens["full_ids"])[0]

                # Compute suppression loss: -log(1 - P(token)) for each token
                # High P(token) → high loss → gradient pushes to reduce P
                loss = suppression_loss.compute(
                    logits,
                    comp_tokens["full_ids"][0],
                    comp_tokens["prompt_len"],
                    coldness=self.config.coldness,
                )

                # Satisficing: instead of minimizing, aim for target value
                # Loss becomes (actual - target)² which is minimized at target
                if self.config.satisfice and dp.src_target_losses:
                    target = dp.src_target_losses[comp_idx]
                    loss = (loss - target) ** 2

                total_loss = total_loss + loss
                dp_losses[0].append(loss.item())

            # -----------------------------------------------------------------
            # Promotion: make dst_completions MORE likely
            # -----------------------------------------------------------------
            # For each completion we want to promote, we compute loss that
            # penalizes low probability of that completion.
            for comp_idx, comp_tokens in enumerate(tokens["dst_tokens"]):
                with self.backend.hooks_context(hooks):
                    logits = self.backend.get_logits(comp_tokens["full_ids"])[0]

                # Compute promotion loss: -log(P(token)) for each token
                # Low P(token) → high loss → gradient pushes to increase P
                loss = promotion_loss.compute(
                    logits,
                    comp_tokens["full_ids"][0],
                    comp_tokens["prompt_len"],
                    coldness=self.config.coldness,
                )

                # Satisficing: aim for target value instead of minimizing
                if self.config.satisfice and dp.dst_target_losses:
                    target = dp.dst_target_losses[comp_idx]
                    loss = (loss - target) ** 2

                total_loss = total_loss + loss
                dp_losses[1].append(loss.item())

            per_completion_losses.append(dp_losses)

        return total_loss, per_completion_losses

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

    def _compute_batch_loss_batched(
        self,
        datapoints: List[TrainingDatapoint],
        tokenized_data: List[Dict],
        layers: List[int],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> Tuple[torch.Tensor, List[List[float]]]:
        """
        Compute loss using batched forward passes for efficiency.

        Groups completions by their steering configuration (negate, token_slice)
        and runs them in batches. This reduces the number of forward passes
        from O(datapoints * completions) to O(batches).

        Args:
            datapoints: Training datapoints.
            tokenized_data: Pre-tokenized data for each datapoint.
            layers: Layers to apply steering hooks.
            batch_size: Maximum batch size for forward passes.

        Returns:
            Tuple of (total_loss, per_completion_losses).
        """
        device = self.backend.get_device()

        # Check if backend supports batched operations
        if not hasattr(self.backend, 'pad_sequences'):
            # Fall back to sequential processing
            return self._compute_batch_loss(datapoints, tokenized_data, layers)

        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        # Initialize with same structure as sequential: [[src_losses], [dst_losses]] per datapoint
        per_completion_losses: List[List[List[float]]] = [[[], []] for _ in datapoints]

        # Loss functions
        promotion_loss = PromotionLoss(
            normalize_by_length=self.config.normalize_by_length,
            eps=self.config.loss_eps,
        )
        suppression_loss = SuppressionLoss(
            use_one_minus=self.config.use_one_minus,
            normalize_by_length=self.config.normalize_by_length,
            eps=self.config.loss_eps,
        )

        # Group completions by steering configuration
        # Key: (negate, token_slice) -> list of completion info
        groups: Dict[tuple, List[Dict]] = defaultdict(list)

        for dp_idx, (dp, tokens) in enumerate(zip(datapoints, tokenized_data)):
            key = (dp.negate, dp.token_slice)

            # Add src completions
            for comp_idx, comp_tokens in enumerate(tokens["src_tokens"]):
                groups[key].append({
                    "type": "src",
                    "dp_idx": dp_idx,
                    "comp_idx": comp_idx,
                    "full_ids": comp_tokens["full_ids"],
                    "prompt_len": comp_tokens["prompt_len"],
                    "target": dp.src_target_losses[comp_idx] if (
                        self.config.satisfice and dp.src_target_losses
                    ) else None,
                })

            # Add dst completions
            for comp_idx, comp_tokens in enumerate(tokens["dst_tokens"]):
                groups[key].append({
                    "type": "dst",
                    "dp_idx": dp_idx,
                    "comp_idx": comp_idx,
                    "full_ids": comp_tokens["full_ids"],
                    "prompt_len": comp_tokens["prompt_len"],
                    "target": dp.dst_target_losses[comp_idx] if (
                        self.config.satisfice and dp.dst_target_losses
                    ) else None,
                })

        # Process each group
        for (negate, token_slice), completions in groups.items():
            if not completions:
                continue

            # Create steering hook for this group
            vector_sign = -1 if negate else 1
            hook = self.steering_mode.create_hook(
                token_slice=token_slice,
                strength=float(vector_sign),
            )
            hooks = [(layer, hook) for layer in layers]

            # Process in batches
            for batch_start in range(0, len(completions), batch_size):
                batch_end = min(batch_start + batch_size, len(completions))
                batch = completions[batch_start:batch_end]

                # Collect sequences for this batch
                sequences = [comp["full_ids"] for comp in batch]

                # Batched forward pass
                logits_batch, attn_mask, orig_lengths = self.backend.get_logits_batched(
                    sequences, hooks=hooks
                )

                # Compute loss for each completion in batch
                for i, comp in enumerate(batch):
                    # Get logits for this sequence (accounting for left padding)
                    max_len = logits_batch.shape[1]
                    orig_len = orig_lengths[i]
                    pad_len = max_len - orig_len

                    # Extract the relevant portion (skip padding)
                    logits = logits_batch[i, pad_len:, :]

                    # Get target IDs (also need to handle the original shape)
                    full_ids = comp["full_ids"]
                    if full_ids.dim() == 2:
                        target_ids = full_ids.squeeze(0)
                    else:
                        target_ids = full_ids

                    # Compute loss based on type
                    if comp["type"] == "src":
                        loss = suppression_loss.compute(
                            logits,
                            target_ids,
                            comp["prompt_len"],
                            coldness=self.config.coldness,
                        )
                    else:
                        loss = promotion_loss.compute(
                            logits,
                            target_ids,
                            comp["prompt_len"],
                            coldness=self.config.coldness,
                        )

                    # Apply satisficing if configured
                    if comp["target"] is not None:
                        loss = (loss - comp["target"]) ** 2

                    total_loss = total_loss + loss

                    # Record per-completion loss
                    dp_idx = comp["dp_idx"]
                    loss_list_idx = 0 if comp["type"] == "src" else 1
                    per_completion_losses[dp_idx][loss_list_idx].append(loss.item())

        return total_loss, per_completion_losses

    def optimize_batched(
        self,
        datapoints: List[TrainingDatapoint],
        layer: LayerSpec,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> OptimizationResult:
        """
        Optimize steering vector using batched forward passes.

        This is a faster version of optimize() that batches multiple
        completions into single forward passes, reducing total GPU
        operations by 10-50x depending on datapoint count.

        Args:
            datapoints: Training examples.
            layer: Which layer(s) to inject the steering vector at.
            batch_size: Maximum batch size for forward passes.
                Larger = faster but more memory. Default 16.

        Returns:
            OptimizationResult containing the optimized vector.
        """
        # Normalize layer to list
        layers = [layer] if isinstance(layer, int) else list(layer)

        # Initialize steering parameters
        self.steering_mode.init_parameters(
            hidden_dim=self.backend.get_hidden_dim(),
            device=self.backend.get_device(),
            dtype=torch.float32,
            starting_norm=self.config.starting_norm,
        )

        # Set up optimizer
        params = self.steering_mode.parameters()
        optimizer = torch.optim.Adam(params, lr=self.config.lr)

        # Precompute tokenizations
        tokenized_data = self._precompute_tokens(datapoints)

        # Notify callbacks
        for callback in self.callbacks:
            callback.on_optimization_start(params, self.config)

        # Main optimization loop
        final_loss = 0.0
        step = 0

        for step in range(self.config.max_iters):
            optimizer.zero_grad()

            # Use batched loss computation
            total_loss, per_completion_losses = self._compute_batch_loss_batched(
                datapoints, tokenized_data, layers, batch_size
            )

            # Add regularization
            if self.regularizer is not None:
                reg_loss = self.regularizer.compute(self.steering_mode)
                total_loss = total_loss + self.regularizer_weight * reg_loss

            # Backprop and update
            total_loss.backward()

            # Clip gradients to prevent NaN from explosion
            if self.config.grad_clip_value is not None:
                torch.nn.utils.clip_grad_value_(params, self.config.grad_clip_value)

            optimizer.step()
            self.steering_mode.apply_constraints(self.config.max_norm)

            final_loss = total_loss.item()

            # Run callbacks
            should_continue = self._run_callbacks(
                step, final_loss, params, per_completion_losses
            )
            if not should_continue:
                break

        # Notify completion
        for callback in self.callbacks:
            callback.on_optimization_end(params, final_loss, step + 1)

        return OptimizationResult(
            vector=self.steering_mode.get_vector(),
            iterations=step + 1,
            final_loss=final_loss,
            metadata={
                "config": self.config.model_dump(),
                "layers": layers,
                "batched": True,
                "batch_size": batch_size,
            },
        )
