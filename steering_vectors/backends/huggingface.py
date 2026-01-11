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
