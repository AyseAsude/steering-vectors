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
        """Return hidden dimension from config."""
        return self.model.config.hidden_size

    def get_num_layers(self) -> int:
        """Return number of layers from config."""
        return self.model.config.num_hidden_layers

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
        """Get the transformer layer module."""
        # Works for Llama, Qwen, Mistral, etc.
        return self.model.model.layers[layer_idx]

    def register_hook(self, layer: int, hook_fn: Callable) -> Any:
        """Register forward pre-hook at layer."""
        layer_module = self._get_layer(layer)
        return layer_module.register_forward_pre_hook(hook_fn)

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
        input_ids = self.tokenize(prompt)
        hooks = hooks or []

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
            **kwargs,
        }

        with self.hooks_context(hooks):
            output_ids = self.model.generate(input_ids, **generation_kwargs)

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
