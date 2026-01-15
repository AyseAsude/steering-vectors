"""TransformerLens backend (placeholder for future implementation)."""

from typing import Any, Callable, List, Optional, Tuple

import torch

from steerex.backends.base import ModelBackend


class TransformerLensBackend(ModelBackend):
    """
    Backend for TransformerLens models.

    TODO: Implement when TransformerLens support is needed.

    TransformerLens uses a different hook API:
    - model.run_with_hooks()
    - Hook points like 'blocks.{layer}.hook_resid_pre'
    """

    def __init__(self, model):
        """
        Initialize TransformerLens backend.

        Args:
            model: A TransformerLens HookedTransformer.
        """
        raise NotImplementedError(
            "TransformerLens backend not yet implemented. "
            "Use HuggingFaceBackend for now."
        )

    def get_hidden_dim(self) -> int:
        raise NotImplementedError

    def get_num_layers(self) -> int:
        raise NotImplementedError

    def get_device(self) -> str:
        raise NotImplementedError

    def get_dtype(self) -> torch.dtype:
        raise NotImplementedError

    def tokenize(self, text: str) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, token_ids: torch.Tensor) -> str:
        raise NotImplementedError

    def get_logits(
        self,
        input_ids: torch.Tensor,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def register_hook(self, layer: int, hook_fn: Callable) -> Any:
        raise NotImplementedError

    def register_output_hook(self, layer: int, hook_fn: Callable) -> Any:
        raise NotImplementedError

    def remove_hook(self, handle: Any) -> None:
        raise NotImplementedError

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> List[str]:
        raise NotImplementedError

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError

    def get_completion_probability(
        self,
        prompt: str,
        completion: str,
        hooks: Optional[List[Tuple[int, Callable]]] = None,
        coldness: float = 1.0,
        log_prob: bool = True,
    ) -> float:
        raise NotImplementedError
