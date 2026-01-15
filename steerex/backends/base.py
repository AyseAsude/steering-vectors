"""Abstract base class for model backends."""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Tuple
from contextlib import contextmanager

import torch

from steerex.steering.base import SteeringMode
from steerex.core.types import LayerSpec, TokenSpec


class ModelBackend(ABC):
    """
    Abstract interface for different model implementations.

    Provides a unified API for:
    - HuggingFace Transformers
    - TransformerLens (future)
    - Other implementations

    This abstraction allows the optimizer to work with any
    model backend without knowing implementation details.
    """

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return the hidden dimension size."""
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """Return the number of transformer layers."""
        ...

    @abstractmethod
    def get_device(self) -> str:
        """Return the device the model is on."""
        ...

    @abstractmethod
    def get_dtype(self) -> torch.dtype:
        """Return the model's dtype."""
        ...

    @abstractmethod
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text to input IDs.

        Args:
            text: Input text.

        Returns:
            Tensor of token IDs (1, seq_len).
        """
        ...

    @abstractmethod
    def decode(self, token_ids: torch.Tensor) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Tensor of token IDs.

        Returns:
            Decoded text string.
        """
        ...

    @abstractmethod
    def get_logits(
        self,
        input_ids: torch.Tensor,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
    ) -> torch.Tensor:
        """
        Run forward pass and return logits.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            hooks: List of (layer, hook_fn) pairs to apply.

        Returns:
            Logits tensor (batch, seq_len, vocab_size).
        """
        ...

    @abstractmethod
    def register_hook(self, layer: int, hook_fn: Callable) -> Any:
        """
        Register a forward pre-hook at the specified layer.

        This captures the INPUT to the layer (residual stream before the layer).
        Used for steering (modifying activations before they enter the layer).

        Args:
            layer: Layer index.
            hook_fn: Hook function with signature (module, args) -> args.

        Returns:
            Hook handle for later removal.
        """
        ...

    @abstractmethod
    def register_output_hook(self, layer: int, hook_fn: Callable) -> Any:
        """
        Register a forward hook at the specified layer to capture output.

        This captures the OUTPUT of the layer (residual stream after the layer).
        Used for extracting activations.

        Args:
            layer: Layer index.
            hook_fn: Hook function with signature (module, args, output) -> output or None.

        Returns:
            Hook handle for later removal.
        """
        ...

    @abstractmethod
    def remove_hook(self, handle: Any) -> None:
        """
        Remove a previously registered hook.

        Args:
            handle: Hook handle from register_hook or register_output_hook.
        """
        ...

    @contextmanager
    def hooks_context(self, hook_infos: List[Tuple[int, Callable]]):
        """
        Context manager for temporary pre-hooks (for steering).

        Args:
            hook_infos: List of (layer, hook_fn) pairs.

        Yields:
            None. Hooks are active within the context.
        """
        handles = []
        try:
            for layer, hook_fn in hook_infos:
                handle = self.register_hook(layer, hook_fn)
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                self.remove_hook(handle)

    @contextmanager
    def output_hooks_context(self, hook_infos: List[Tuple[int, Callable]]):
        """
        Context manager for temporary output hooks (for extraction).

        Args:
            hook_infos: List of (layer, hook_fn) pairs.

        Yields:
            None. Hooks are active within the context.
        """
        handles = []
        try:
            for layer, hook_fn in hook_infos:
                handle = self.register_output_hook(layer, hook_fn)
                handles.append(handle)
            yield
        finally:
            for handle in handles:
                self.remove_hook(handle)

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts in a batch.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy.
            hooks: Optional hooks to apply during generation.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts (one per prompt).
        """
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> str:
        """
        Generate text with optional steering.

        Args:
            prompt: Input prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy.
            hooks: Optional hooks to apply during generation.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text (excluding prompt).
        """
        ...

    def generate_with_steering(
        self,
        prompt: str,
        steering_mode: SteeringMode,
        layers: LayerSpec,
        strength: float = 1.0,
        token_slice: TokenSpec = None,
        **kwargs,
    ) -> str:
        """
        Generate text with steering applied.

        Convenience method that wraps generate() with proper hooks.

        Args:
            prompt: Input prompt.
            steering_mode: The steering mode to apply.
            layers: Layer(s) to apply steering at.
            strength: Steering strength multiplier.
            token_slice: Which tokens to steer.
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        if isinstance(layers, int):
            layers = [layers]

        hooks = [
            (layer, steering_mode.create_hook(token_slice, strength))
            for layer in layers
        ]

        return self.generate(prompt, hooks=hooks, **kwargs)

    def generate_with_steering_batch(
        self,
        prompts: List[str],
        steering_mode: SteeringMode,
        layers: LayerSpec,
        strength: float = 1.0,
        token_slice: TokenSpec = None,
        **kwargs,
    ) -> List[str]:
        """
        Generate text for multiple prompts with steering applied.

        Convenience method that wraps generate_batch() with proper hooks.

        Args:
            prompts: List of input prompts.
            steering_mode: The steering mode to apply.
            layers: Layer(s) to apply steering at.
            strength: Steering strength multiplier.
            token_slice: Which tokens to steer.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts.
        """
        if isinstance(layers, int):
            layers = [layers]

        hooks = [
            (layer, steering_mode.create_hook(token_slice, strength))
            for layer in layers
        ]

        return self.generate_batch(prompts, hooks=hooks, **kwargs)

    @abstractmethod
    def get_completion_probability(
        self,
        prompt: str,
        completion: str,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        coldness: float = 1.0,
        log_prob: bool = True,
    ) -> float:
        """
        Compute probability of completion given prompt.

        Args:
            prompt: Input prompt.
            completion: Target completion.
            hooks: Optional hooks to apply.
            coldness: Inverse temperature.
            log_prob: If True, return log probability.

        Returns:
            (Log) probability of the completion.
        """
        ...
