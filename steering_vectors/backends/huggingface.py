"""HuggingFace Transformers backend."""

from typing import Any, Callable, List, Optional, Tuple
from contextlib import contextmanager

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from steering_vectors.backends.base import ModelBackend


class HuggingFaceBackend(ModelBackend):
    """
    Backend for HuggingFace Transformers models.

    Supports Llama-like architectures with model.model.layers structure.

    Example:
        >>> from transformers import AutoModelForCausalLM, AutoTokenizer
        >>> model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        >>> backend = HuggingFaceBackend(model, tokenizer)
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: Optional[str] = None,
    ):
        """
        Initialize HuggingFace backend.

        Args:
            model: A HuggingFace causal LM.
            tokenizer: The corresponding tokenizer.
            device: Device override. If None, uses model's device.
        """
        self.model = model
        self.tokenizer = tokenizer
        self._device = device

        # Ensure model is in eval mode and frozen
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

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
    ) -> torch.Tensor:
        """Run forward pass with optional hooks."""
        hooks = hooks or []

        with self.hooks_context(hooks):
            outputs = self.model(input_ids, use_cache=False)
            return outputs.logits

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
        hooks = hooks or []

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

        with self.hooks_context(hooks):
            output_ids = self.model.generate(**inputs, **generation_kwargs)

        # Restore original padding side
        self.tokenizer.padding_side = original_padding_side

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.batch_decode(
            output_ids[:, input_len:], skip_special_tokens=True
        )

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
        Compute (log) probability of completion given prompt.

        Matches steering_opt.get_completion_logprob_hf behavior.
        """
        hooks = hooks or []

        prompt_ids = self.tokenize(prompt)
        full_ids = self.tokenize(prompt + completion)

        prompt_len = prompt_ids.shape[1]
        total_len = full_ids.shape[1]

        with self.hooks_context(hooks):
            logits = self.get_logits(full_ids, hooks=[])[0].float()

        probs = torch.softmax(logits * coldness, dim=-1)

        total_log_prob = 0.0
        for i in range(prompt_len, total_len):
            target_token = full_ids[0, i]
            prob = probs[i - 1, target_token]
            total_log_prob += torch.log(prob + 1e-10).item()

        return total_log_prob if log_prob else torch.exp(torch.tensor(total_log_prob)).item()

    def get_completion_probability_one_minus(
        self,
        prompt: str,
        completion: str,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        coldness: float = 1.0,
    ) -> float:
        """
        Compute log(1 - P(completion)) for suppression loss.

        Used when suppressing completions.
        """
        hooks = hooks or []

        prompt_ids = self.tokenize(prompt)
        full_ids = self.tokenize(prompt + completion)

        prompt_len = prompt_ids.shape[1]
        total_len = full_ids.shape[1]

        with self.hooks_context(hooks):
            logits = self.get_logits(full_ids, hooks=[])[0].float()

        probs = torch.softmax(logits * coldness, dim=-1)

        total_log_prob = 0.0
        for i in range(prompt_len, total_len):
            target_token = full_ids[0, i]
            prob = 1 - probs[i - 1, target_token]
            total_log_prob += torch.log(prob + 1e-10).item()

        return total_log_prob
