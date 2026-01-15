"""HuggingFace Transformers backend.

Thread-Safe Steering Architecture:
----------------------------------
This backend supports fully concurrent steered generation without locks by using
thread-local storage for steering parameters. Each thread has its own isolated
steering configuration that doesn't interfere with other threads.

How it works:
1. Persistent hooks are registered ONCE at initialization for all layers
2. Each hook checks thread-local storage for steering parameters
3. If no steering is set for the current thread, hooks are no-ops (pass-through)
4. Each thread sets its steering config, generates, then clears it
5. No locks needed - thread-local storage provides isolation

This enables true parallel steered generation across multiple threads.
"""

import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steerex.backends.base import ModelBackend

if TYPE_CHECKING:
    from steerex.steering.base import SteeringMode
    from steerex.core.types import LayerSpec, TokenSpec


@dataclass
class ThreadSteeringConfig:
    """Per-thread steering configuration for a single layer."""
    vector: torch.Tensor
    strength: float
    token_slice: Optional[slice] = None


@dataclass
class ThreadOutputHookConfig:
    """Per-thread output hook configuration for a single layer."""
    callback: Callable
    # Store captured outputs here (the callback can append to this list)
    captured: List[torch.Tensor] = None

    def __post_init__(self):
        if self.captured is None:
            self.captured = []


class HuggingFaceBackend(ModelBackend):
    """
    Backend for HuggingFace Transformers models.

    Supports Llama-like architectures with model.model.layers structure.

    Thread Safety:
        This backend supports fully concurrent steered generation. Multiple
        threads can generate with different steering vectors simultaneously
        without interference or locks. Each thread's steering is isolated
        via thread-local storage.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> backend = HuggingFaceBackend(model, tokenizer)
        >>>
        >>> # Concurrent steered generation from multiple threads - no locks!
        >>> import concurrent.futures
        >>> with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        ...     futures = [
        ...         executor.submit(backend.generate_with_steering, prompt, steering, layer, strength)
        ...         for prompt, steering, layer, strength in jobs
        ...     ]
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
        gradient_checkpointing: bool = False,
    ):
        """
        Initialize HuggingFace backend.

        Args:
            model: A HuggingFace causal LM.
            tokenizer: The corresponding tokenizer.
            device: Device override. If None, uses model's device.
            gradient_checkpointing: Enable gradient checkpointing to reduce
                memory usage at the cost of ~30% slower backward pass.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device
        self._gradient_checkpointing = gradient_checkpointing

        # Thread-local storage for steering parameters
        # Each thread has its own isolated steering config
        self._steering_local = threading.local()

        # Thread-local storage for output hooks (used in extraction)
        # Each thread has its own isolated output hook callbacks
        self._output_hook_local = threading.local()

        # Track if persistent hooks have been registered
        self._steering_hooks_registered = False
        self._steering_hook_handles: List[Any] = []
        self._output_hooks_registered = False
        self._output_hook_handles: List[Any] = []

        # Ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        # Setup padding token for batched operations
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Register persistent steering hooks for all layers
        self._register_persistent_steering_hooks()

        # Register persistent output hooks for all layers (for extraction)
        self._register_persistent_output_hooks()

    def get_hidden_dim(self) -> int:
        """Return hidden dimension from config.
        
        Handles different model architectures:
        - Standard models: config.hidden_size
        - Multimodal models (e.g., Gemma3): config.text_config.hidden_size
        """
        config = self.model.config
        
        # Try standard hidden_size first
        if hasattr(config, 'hidden_size'):
            return config.hidden_size
        
        # For multimodal models like Gemma3, hidden_size is in text_config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'hidden_size'):
            return config.text_config.hidden_size
        
        raise AttributeError(
            f"Cannot find hidden dimension in {type(config).__name__}. "
            f"Expected 'hidden_size' or 'text_config.hidden_size' attribute."
        )

    def get_num_layers(self) -> int:
        """Return number of layers from config.
        
        Handles different model architectures:
        - Standard models: config.num_hidden_layers
        - Multimodal models (e.g., Gemma3): config.text_config.num_hidden_layers
        """
        config = self.model.config
        
        # Try standard num_hidden_layers first
        if hasattr(config, 'num_hidden_layers'):
            return config.num_hidden_layers
        
        # For multimodal models like Gemma3, num_hidden_layers is in text_config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
            return config.text_config.num_hidden_layers
        
        raise AttributeError(
            f"Cannot find number of layers in {type(config).__name__}. "
            f"Expected 'num_hidden_layers' or 'text_config.num_hidden_layers' attribute."
        )

    def get_device(self) -> str:
        """Return model device."""
        if self._device:
            return self._device
        return str(next(self.model.parameters()).device)

    def get_dtype(self) -> torch.dtype:
        """Return model dtype."""
        return next(self.model.parameters()).dtype

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to tensor."""
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.get_device())

    def tokenize_batch(
        self,
        texts: List[str],
        padding: bool = True,
        return_attention_mask: bool = True,
    ) -> dict:
        """Tokenize multiple texts with padding.

        Args:
            texts: List of texts to tokenize.
            padding: Whether to pad sequences to same length.
            return_attention_mask: Whether to return attention mask.

        Returns:
            Dictionary with 'input_ids' and optionally 'attention_mask'.
        """
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=padding,
            truncation=True,
            return_attention_mask=return_attention_mask,
        )
        return {k: v.to(self.get_device()) for k, v in encoded.items()}

    def pad_sequences(
        self,
        sequences: List[torch.Tensor],
        padding_value: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a list of token sequences to the same length.

        Args:
            sequences: List of tensors, each (1, seq_len) or (seq_len,).
            padding_value: Value to use for padding. Defaults to pad_token_id.

        Returns:
            Tuple of (padded_ids, attention_mask), each (batch, max_len).
        """
        if padding_value is None:
            padding_value = self.tokenizer.pad_token_id

        # Normalize to 1D tensors
        seqs_1d = []
        for seq in sequences:
            if seq.dim() == 2:
                seq = seq.squeeze(0)
            seqs_1d.append(seq)

        # Find max length
        max_len = max(len(s) for s in seqs_1d)

        # Pad sequences (left padding for causal LMs)
        padded = []
        masks = []
        for seq in seqs_1d:
            pad_len = max_len - len(seq)
            if pad_len > 0:
                padding = torch.full((pad_len,), padding_value, dtype=seq.dtype, device=seq.device)
                padded_seq = torch.cat([padding, seq])
                mask = torch.cat([
                    torch.zeros(pad_len, dtype=torch.long, device=seq.device),
                    torch.ones(len(seq), dtype=torch.long, device=seq.device),
                ])
            else:
                padded_seq = seq
                mask = torch.ones(len(seq), dtype=torch.long, device=seq.device)
            padded.append(padded_seq)
            masks.append(mask)

        return torch.stack(padded), torch.stack(masks)

    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _get_layer(self, layer_idx: int):
        """Get the transformer layer module.
        
        Handles different model architectures:
        - Standard models (Llama, Qwen, Mistral): model.model.layers
        - GPTNeoX models: model.gpt_neox.layers
        - Multimodal models (e.g., Gemma3): model.language_model.model.layers
        """
        # Try standard structure first (Llama, Qwen, Mistral, etc.)
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers[layer_idx]
        
        # For GPTNeoX models
        if hasattr(self.model, 'gpt_neox') and hasattr(self.model.gpt_neox, 'layers'):
            return self.model.gpt_neox.layers[layer_idx]
        
        # For multimodal models like Gemma3, layers are in language_model
        if hasattr(self.model, 'language_model'):
            if hasattr(self.model.language_model, 'model') and hasattr(self.model.language_model.model, 'layers'):
                return self.model.language_model.model.layers[layer_idx]
            if hasattr(self.model.language_model, 'layers'):
                return self.model.language_model.layers[layer_idx]
        
        raise AttributeError(
            f"Cannot find layers in {type(self.model).__name__}. "
            f"Expected 'model.layers', 'gpt_neox.layers', 'language_model.model.layers', or 'language_model.layers'."
        )

    # =========================================================================
    # Thread-Local Steering System
    # =========================================================================
    # These methods enable fully concurrent steered generation without locks.
    # Each thread has isolated steering config via threading.local().

    def _register_persistent_steering_hooks(self) -> None:
        """Register persistent steering hooks on all layers.

        These hooks are registered ONCE at initialization and remain active
        for the lifetime of the backend. They check thread-local storage
        to determine if steering should be applied for the current thread.

        This design eliminates the need for per-generation hook registration,
        which was the source of thread-safety issues requiring locks.
        """
        if self._steering_hooks_registered:
            return

        num_layers = self.get_num_layers()

        for layer_idx in range(num_layers):
            hook = self._create_thread_local_steering_hook(layer_idx)
            layer_module = self._get_layer(layer_idx)
            handle = layer_module.register_forward_pre_hook(hook)
            self._steering_hook_handles.append(handle)

        self._steering_hooks_registered = True

    def _create_thread_local_steering_hook(self, layer_idx: int) -> Callable:
        """Create a steering hook that reads from thread-local storage.

        The hook is a closure that captures:
        - layer_idx: The layer this hook is for
        - self._steering_local: Reference to thread-local storage

        At runtime, the hook:
        1. Checks if steering is configured for this thread
        2. If not, returns unchanged (fast no-op path)
        3. If yes, checks if this layer should be steered
        4. If yes, applies the steering transformation

        Args:
            layer_idx: The layer index this hook is registered on.

        Returns:
            A hook function compatible with register_forward_pre_hook.
        """
        steering_local = self._steering_local  # Capture reference

        def steering_hook(module, args):
            # Fast path: check if any steering is configured for this thread
            config = getattr(steering_local, 'config', None)
            if config is None:
                return args  # No steering, pass through unchanged

            # Check if this layer should be steered
            layer_config = config.get(layer_idx)
            if layer_config is None:
                return args  # This layer not steered

            # Apply steering transformation
            hidden_states = args[0]  # Shape: [batch, seq_len, hidden_dim]
            vector = layer_config.vector
            strength = layer_config.strength
            token_slice = layer_config.token_slice

            # Determine which tokens to steer
            if token_slice is not None:
                # Steer only specified tokens
                steered = hidden_states.clone()
                steered[:, token_slice, :] = steered[:, token_slice, :] + strength * vector
                return (steered,) + args[1:]
            else:
                # Steer all tokens
                return (hidden_states + strength * vector,) + args[1:]

        return steering_hook

    def _set_thread_steering(
        self,
        layers: List[int],
        vector: torch.Tensor,
        strength: float = 1.0,
        token_slice: Optional[slice] = None,
    ) -> None:
        """Set steering configuration for the current thread.

        This configures which layers to steer and with what parameters.
        The steering will only affect generation calls made from this thread.

        Thread Safety:
            Uses threading.local() which guarantees complete isolation.
            Each thread has its own 'config' attribute that is invisible
            to other threads. This is a Python language guarantee.

        Args:
            layers: List of layer indices to apply steering at.
            vector: The steering vector to add to activations.
            strength: Multiplier for the steering vector.
            token_slice: Optional slice specifying which token positions to steer.
        """
        # Detach and clone vector to ensure complete isolation
        # This prevents any gradient graph sharing between threads
        isolated_vector = vector.detach().clone()

        config: Dict[int, ThreadSteeringConfig] = {}
        for layer_idx in layers:
            config[layer_idx] = ThreadSteeringConfig(
                vector=isolated_vector,
                strength=strength,
                token_slice=token_slice,
            )
        self._steering_local.config = config

    def _clear_thread_steering(self) -> None:
        """Clear steering configuration for the current thread.

        Must be called after generation to prevent steering from leaking
        to subsequent operations on the same thread.
        """
        self._steering_local.config = None

    def _get_thread_steering(self) -> Optional[Dict[int, ThreadSteeringConfig]]:
        """Get the current thread's steering configuration.

        Returns:
            The steering config dict, or None if no steering is configured.
        """
        return getattr(self._steering_local, 'config', None)

    @contextmanager
    def thread_steering_context(
        self,
        layers: List[int],
        vector: torch.Tensor,
        strength: float = 1.0,
        token_slice: Optional[slice] = None,
    ):
        """Context manager for thread-local steering.

        Safely sets and clears steering configuration for the current thread.
        Use this to ensure steering is always cleaned up, even on exceptions.

        Args:
            layers: List of layer indices to apply steering at.
            vector: The steering vector to add to activations.
            strength: Multiplier for the steering vector.
            token_slice: Optional slice specifying which token positions to steer.

        Yields:
            None. Steering is active within the context.

        Example:
            >>> with backend.thread_steering_context([16], vector, strength=1.5):
            ...     output = backend.generate_batch(prompts)
        """
        self._set_thread_steering(layers, vector, strength, token_slice)
        try:
            yield
        finally:
            self._clear_thread_steering()

    # =========================================================================
    # Thread-Local Output Hooks (for extraction)
    # =========================================================================
    # These methods enable fully concurrent extraction without locks.

    def _register_persistent_output_hooks(self) -> None:
        """Register persistent output hooks on all layers.

        These hooks are registered ONCE at initialization and check thread-local
        storage to determine if output capture is needed for the current thread.
        """
        if self._output_hooks_registered:
            return

        num_layers = self.get_num_layers()

        for layer_idx in range(num_layers):
            hook = self._create_thread_local_output_hook(layer_idx)
            layer_module = self._get_layer(layer_idx)
            handle = layer_module.register_forward_hook(hook)
            self._output_hook_handles.append(handle)

        self._output_hooks_registered = True

    def _create_thread_local_output_hook(self, layer_idx: int) -> Callable:
        """Create an output hook that checks thread-local storage.

        Args:
            layer_idx: The layer index this hook is registered on.

        Returns:
            A hook function compatible with register_forward_hook.
        """
        output_hook_local = self._output_hook_local  # Capture reference

        def output_hook(module, args, output):
            # Fast path: check if any output capture is configured for this thread
            config = getattr(output_hook_local, 'config', None)
            if config is None:
                return output  # No capture configured

            # Check if this layer should capture
            layer_config = config.get(layer_idx)
            if layer_config is None:
                return output  # This layer not capturing

            # Call the thread's callback with the output
            try:
                result = layer_config.callback(module, args, output)
                return result if result is not None else output
            except Exception:
                # Don't let callback errors crash the forward pass
                return output

        return output_hook

    def _set_thread_output_hooks(
        self,
        hook_infos: List[Tuple[int, Callable]],
    ) -> None:
        """Set output hook callbacks for the current thread.

        Args:
            hook_infos: List of (layer_idx, callback) pairs.
        """
        config: Dict[int, ThreadOutputHookConfig] = {}
        for layer_idx, callback in hook_infos:
            config[layer_idx] = ThreadOutputHookConfig(callback=callback)
        self._output_hook_local.config = config

    def _clear_thread_output_hooks(self) -> None:
        """Clear output hook callbacks for the current thread."""
        self._output_hook_local.config = None

    @contextmanager
    def thread_output_hooks_context(
        self,
        hook_infos: List[Tuple[int, Callable]],
    ):
        """Context manager for thread-local output hooks.

        Thread-safe replacement for output_hooks_context. Multiple threads
        can use this concurrently without interference.

        Args:
            hook_infos: List of (layer_idx, callback) pairs.

        Yields:
            None. Hooks are active within the context.
        """
        self._set_thread_output_hooks(hook_infos)
        try:
            yield
        finally:
            self._clear_thread_output_hooks()

    @contextmanager
    def output_hooks_context(self, hook_infos: List[Tuple[int, Callable]]):
        """Context manager for temporary output hooks (thread-safe override).

        This overrides the base class method to use thread-local storage,
        enabling fully concurrent extraction operations.

        Args:
            hook_infos: List of (layer, hook_fn) pairs.

        Yields:
            None. Hooks are active within the context.
        """
        # Use thread-local hooks for thread safety
        self._set_thread_output_hooks(hook_infos)
        try:
            yield
        finally:
            self._clear_thread_output_hooks()

    def register_hook(self, layer: int, hook_fn: Callable) -> Any:
        """Register forward pre-hook at layer."""
        layer_module = self._get_layer(layer)
        return layer_module.register_forward_pre_hook(hook_fn)

    def register_output_hook(self, layer: int, hook_fn: Callable) -> Any:
        """
        Register forward hook at layer to capture output.

        Unlike register_hook (pre-hook), this captures the layer's output.
        Hook signature: hook_fn(module, args, output) -> output or None

        Args:
            layer: Layer index.
            hook_fn: Hook function receiving (module, args, output).

        Returns:
            Hook handle for later removal.
        """
        layer_module = self._get_layer(layer)
        return layer_module.register_forward_hook(hook_fn)

    def remove_hook(self, handle: Any) -> None:
        """Remove hook by handle."""
        handle.remove()

    def get_logits(
        self,
        input_ids: torch.Tensor,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run forward pass with optional hooks.

        Args:
            input_ids: Input token IDs (batch, seq_len).
            hooks: List of (layer, hook_fn) pairs to apply.
            attention_mask: Optional attention mask (batch, seq_len).
                Required for padded batches.

        Returns:
            Logits tensor (batch, seq_len, vocab_size).
        """
        hooks = hooks or []

        with self.hooks_context(hooks):
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs.logits

    def get_logits_batched(
        self,
        sequences: List[torch.Tensor],
        hooks: Optional[List[Tuple[int, Callable]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """Run batched forward pass on variable-length sequences.

        Pads sequences and runs a single forward pass for efficiency.

        Args:
            sequences: List of token ID tensors, each (1, seq_len) or (seq_len,).
            hooks: List of (layer, hook_fn) pairs to apply.

        Returns:
            Tuple of (logits, attention_mask, original_lengths).
            logits: (batch, max_len, vocab).
            attention_mask: (batch, max_len).
            original_lengths: Original length of each sequence.
        """
        hooks = hooks or []

        # Record original lengths
        original_lengths = []
        for seq in sequences:
            if seq.dim() == 2:
                original_lengths.append(seq.shape[1])
            else:
                original_lengths.append(len(seq))

        # Pad sequences
        padded_ids, attention_mask = self.pad_sequences(sequences)

        # Single forward pass
        logits = self.get_logits(padded_ids, hooks=hooks, attention_mask=attention_mask)

        return logits, attention_mask, original_lengths

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

        For custom hooks (non-steering), this method uses traditional hook
        registration. For steered generation, use generate_with_steering_batch()
        which uses thread-local steering for full parallelism.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy.
            hooks: Optional custom hooks to apply during generation.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts (one per prompt).
        """
        # Use the internal generation method
        return self._generate_batch_internal(
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            custom_hooks=hooks,
            **kwargs,
        )

    def _generate_batch_internal(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        custom_hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> List[str]:
        """Internal batch generation with thread-local steering support.

        This method handles both:
        1. Thread-local steering (via _steering_local) - no lock needed
        2. Custom hooks (via custom_hooks param) - uses hooks_context

        The thread-local steering hooks are always active but are no-ops
        when no steering is configured for the current thread.

        Args:
            prompts: List of input prompts.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy.
            custom_hooks: Optional custom hooks (extraction, etc.) - NOT for steering.
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts (one per prompt).
        """
        custom_hooks = custom_hooks or []

        # Set up padding for batch generation
        original_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.get_device())

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            **kwargs,
        }

        # Generate with optional custom hooks
        # Note: Thread-local steering hooks are ALWAYS active (registered at init)
        # They check _steering_local and apply steering only if configured
        if custom_hooks:
            # Custom hooks need the context manager (for extraction, etc.)
            with self.hooks_context(custom_hooks):
                output_ids = self.model.generate(**inputs, **generation_kwargs)
        else:
            # No custom hooks - thread-local steering still works via persistent hooks
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )

    def generate_with_steering_batch(
        self,
        prompts: List[str],
        steering_mode: "SteeringMode",
        layers: "LayerSpec",
        strength: float = 1.0,
        token_slice: "TokenSpec" = None,
        **kwargs,
    ) -> List[str]:
        """Generate text for multiple prompts with steering applied.

        This method uses thread-local steering for full parallelism.
        Multiple threads can call this concurrently with different steering
        vectors without interference - no locks required.

        Args:
            prompts: List of input prompts.
            steering_mode: The steering mode containing the vector.
            layers: Layer(s) to apply steering at.
            strength: Steering strength multiplier.
            token_slice: Which tokens to steer (None = all).
            **kwargs: Additional generation arguments.

        Returns:
            List of generated texts.
        """
        # Normalize layers to list
        if isinstance(layers, int):
            layers = [layers]

        # Get the steering vector from the steering mode
        vector = steering_mode.vector

        # Use thread-local steering context for isolation
        with self.thread_steering_context(layers, vector, strength, token_slice):
            return self._generate_batch_internal(prompts, **kwargs)

    def generate_with_steering(
        self,
        prompt: str,
        steering_mode: "SteeringMode",
        layers: "LayerSpec",
        strength: float = 1.0,
        token_slice: "TokenSpec" = None,
        **kwargs,
    ) -> str:
        """Generate text with steering applied.

        This method uses thread-local steering for full parallelism.
        Multiple threads can call this concurrently with different steering
        vectors without interference - no locks required.

        Args:
            prompt: Input prompt.
            steering_mode: The steering mode containing the vector.
            layers: Layer(s) to apply steering at.
            strength: Steering strength multiplier.
            token_slice: Which tokens to steer (None = all).
            **kwargs: Additional generation arguments.

        Returns:
            Generated text.
        """
        return self.generate_with_steering_batch(
            [prompt], steering_mode, layers, strength, token_slice, **kwargs
        )[0]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> str:
        """Generate text with optional steering."""
        return self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            hooks=hooks,
            **kwargs,
        )[0]

    def get_completion_probability(
        self,
        prompt: str,
        completion: str,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        coldness: float = 1.0,
        log_prob: bool = True,
    ) -> float:
        """
        Compute (log) probability of completion given prompt (vectorized).

        Matches steering_opt.get_completion_logprob_hf behavior.
        """
        hooks = hooks or []

        prompt_ids = self.tokenize(prompt)
        full_ids = self.tokenize(prompt + completion)

        prompt_len = prompt_ids.shape[1]
        total_len = full_ids.shape[1]
        completion_len = total_len - prompt_len

        # Handle empty completion
        if completion_len <= 0:
            return 0.0

        with self.hooks_context(hooks):
            logits = self.get_logits(full_ids, hooks=[])[0].float()

        probs = torch.softmax(logits * coldness, dim=-1)

        # Get completion token IDs
        completion_ids = full_ids[0, prompt_len:total_len]

        # Get probabilities for completion tokens (shifted by 1 for autoregressive)
        completion_probs = probs[prompt_len - 1 : total_len - 1]

        # Gather target token probabilities
        target_probs = completion_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Compute log probability sum
        total_log_prob = torch.log(target_probs + 1e-10).sum().item()

        return total_log_prob if log_prob else torch.exp(torch.tensor(total_log_prob)).item()

    def get_completion_probability_one_minus(
        self,
        prompt: str,
        completion: str,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        coldness: float = 1.0,
    ) -> float:
        """
        Compute log(1 - P(completion)) for suppression loss (vectorized).

        Used when suppressing completions.
        """
        hooks = hooks or []

        prompt_ids = self.tokenize(prompt)
        full_ids = self.tokenize(prompt + completion)

        prompt_len = prompt_ids.shape[1]
        total_len = full_ids.shape[1]
        completion_len = total_len - prompt_len

        # Handle empty completion
        if completion_len <= 0:
            return 0.0

        with self.hooks_context(hooks):
            logits = self.get_logits(full_ids, hooks=[])[0].float()

        probs = torch.softmax(logits * coldness, dim=-1)

        # Get completion token IDs
        completion_ids = full_ids[0, prompt_len:total_len]

        # Get probabilities for completion tokens (shifted by 1 for autoregressive)
        completion_probs = probs[prompt_len - 1 : total_len - 1]

        # Gather target token probabilities
        target_probs = completion_probs.gather(
            dim=-1, index=completion_ids.unsqueeze(-1)
        ).squeeze(-1)

        # Compute log(1 - p) sum
        total_log_prob = torch.log(1 - target_probs + 1e-10).sum().item()

        return total_log_prob
