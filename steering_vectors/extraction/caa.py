"""
Contrastive Activation Addition (CAA) extractor.

This module implements the CAA method for extracting steering vectors:
    vector = mean(positive_activations) - mean(negative_activations)

CAA is the recommended extraction method because:
- It's geometrically constrained (only one solution)
- Fast (single forward pass per example)
- Reliable (doesn't get stuck in local minima)
- No hyperparameters to tune
"""

from typing import Any, List, Literal, TYPE_CHECKING

import torch

from steering_vectors.extraction.base import VectorExtractor, ExtractionResult
from steering_vectors.extraction.datapoint import ContrastPair

if TYPE_CHECKING:
    from steering_vectors.backends.base import ModelBackend


class CAAExtractor(VectorExtractor):
    """
    Contrastive Activation Addition (CAA) extractor.

    Computes steering vectors as the difference between mean activations
    of positive and negative examples. This is the recommended method
    for most use cases.

    Attributes:
        token_position: Which token position to extract activations from.
            - "mean": Mean of all response token activations (default).
            - "last": Activation at the last token of the response.
            - "last_prompt_token": Activation at the last prompt token.

    Example:
        >>> extractor = CAAExtractor(token_position="mean")
        >>> result = extractor.extract(backend, tokenizer, pairs, layer=16)
        >>> steering = result.to_steering()
    """

    def __init__(
        self,
        token_position: Literal["mean", "last", "last_prompt_token"] = "mean",
    ):
        """
        Initialize CAA extractor.

        Args:
            token_position: Which token position to extract from.
        """
        self.token_position = token_position

    @property
    def method_name(self) -> str:
        return "caa"

    def extract(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        pairs: List[ContrastPair],
        layer: int,
    ) -> ExtractionResult:
        """
        Extract steering vector using CAA.

        Args:
            backend: Model backend for forward passes.
            tokenizer: Tokenizer with chat template support.
            pairs: List of positive/negative contrast pairs.
            layer: Layer to extract the vector for.

        Returns:
            ExtractionResult containing the CAA vector.
        """
        self.validate_pairs(pairs)

        positive_activations = []
        negative_activations = []

        for pair in pairs:
            if pair.format == "messages":
                pos_act = self._extract_from_messages(
                    backend, tokenizer, pair.get_positive_messages(), layer
                )
                neg_act = self._extract_from_messages(
                    backend, tokenizer, pair.get_negative_messages(), layer
                )
            else:
                pos_prompt, pos_completion = pair.get_positive_prompt_completion()
                neg_prompt, neg_completion = pair.get_negative_prompt_completion()
                pos_act = self._extract_from_completion(
                    backend, tokenizer, pos_prompt, pos_completion, layer
                )
                neg_act = self._extract_from_completion(
                    backend, tokenizer, neg_prompt, neg_completion, layer
                )

            positive_activations.append(pos_act)
            negative_activations.append(neg_act)

        # Compute mean difference
        pos_mean = torch.stack(positive_activations).mean(dim=0)
        neg_mean = torch.stack(negative_activations).mean(dim=0)
        vector = pos_mean - neg_mean

        return ExtractionResult(
            vector=vector,
            layer=layer,
            method=self.method_name,
            metadata={
                "token_position": self.token_position,
                "num_pairs": len(pairs),
                "vector_norm": vector.norm().item(),
            },
        )

    def _extract_from_messages(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        messages: List[dict],
        layer: int,
    ) -> torch.Tensor:
        """Extract activation from chat messages."""
        # Get full conversation text
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # Get prompt-only text (everything before assistant response)
        prompt_messages = messages[:-1]
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Verify prompt_text is a prefix of full_text
        if not full_text.startswith(prompt_text):
            raise ValueError(
                "Chat template produced inconsistent results: "
                "prompt_text is not a prefix of full_text"
            )

        # Find boundary using offset mapping
        prompt_char_end = len(prompt_text)
        encoding = tokenizer(
            full_text,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        full_ids = encoding["input_ids"].to(backend.get_device())
        offsets = encoding["offset_mapping"][0]

        # Find first token that starts at or after prompt_char_end
        prompt_len = len(offsets)
        for i, (start, end) in enumerate(offsets):
            if start >= prompt_char_end:
                prompt_len = i
                break

        full_len = full_ids.shape[1]

        if full_len <= prompt_len:
            raise ValueError(
                f"No response tokens found. prompt_len={prompt_len}, full_len={full_len}"
            )

        # Extract activations
        activations = self._extract_layer_activations(backend, full_ids, layer)

        return self._select_activation(activations, prompt_len, full_len)

    def _extract_from_completion(
        self,
        backend: "ModelBackend",
        tokenizer: Any,
        prompt: str,
        completion: str,
        layer: int,
    ) -> torch.Tensor:
        """Extract activation from prompt/completion pair."""
        prompt_ids = backend.tokenize(prompt)
        full_ids = backend.tokenize(prompt + completion)

        prompt_len = prompt_ids.shape[1]
        full_len = full_ids.shape[1]

        if full_len <= prompt_len:
            raise ValueError("Completion is empty after tokenization")

        activations = self._extract_layer_activations(backend, full_ids, layer)

        return self._select_activation(activations, prompt_len, full_len)

    def _select_activation(
        self,
        activations: torch.Tensor,
        prompt_len: int,
        full_len: int,
    ) -> torch.Tensor:
        """Select activation based on token_position strategy."""
        if self.token_position == "last_prompt_token":
            return activations[prompt_len - 1]
        elif self.token_position == "last":
            return activations[full_len - 1]
        elif self.token_position == "mean":
            response_activations = activations[prompt_len:full_len]
            return response_activations.mean(dim=0)
        else:
            raise ValueError(f"Unknown token_position: {self.token_position}")

    def _extract_layer_activations(
        self,
        backend: "ModelBackend",
        input_ids: torch.Tensor,
        layer: int,
    ) -> torch.Tensor:
        """Extract layer output activations."""
        captured = []

        def capture_hook(module, args, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            captured.append(hidden_states.detach().clone())
            return output

        with backend.output_hooks_context([(layer, capture_hook)]):
            _ = backend.get_logits(input_ids)

        return captured[0].squeeze(0)  # [seq_len, hidden_dim]
