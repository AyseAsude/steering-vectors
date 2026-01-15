"""
Contrastive Activation Addition (CAA) extractor.

This module implements the CAA method for extracting steering vectors:
    vector = mean(positive_activations) - mean(negative_activations)

CAA is the recommended extraction method because:
- It's geometrically constrained (only one solution)
- Fast (single forward pass per example)
- Reliable (doesn't get stuck in local minima)
- No hyperparameters to tune

Supports optional extreme outlier removal for cleaner vectors.
"""

from typing import Any, List, Literal, Optional, Tuple, TYPE_CHECKING

import torch

from steerex.extraction.base import VectorExtractor, ExtractionResult
from steerex.extraction.datapoint import ContrastPair

if TYPE_CHECKING:
    from steerex.backends.base import ModelBackend


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
        remove_outliers: If True, remove extreme outliers before averaging.
        outlier_std_threshold: Std dev threshold for outlier removal (default 3.0).

    Example:
        >>> extractor = CAAExtractor(token_position="mean", remove_outliers=True)
        >>> result = extractor.extract(backend, tokenizer, pairs, layer=16)
        >>> steering = result.to_steering()
    """

    def __init__(
        self,
        token_position: Literal["mean", "last", "last_prompt_token"] = "mean",
        remove_outliers: bool = False,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Initialize CAA extractor.

        Args:
            token_position: Which token position to extract from.
            remove_outliers: If True, remove extreme outliers (> threshold std devs).
            outlier_std_threshold: Number of standard deviations for outlier detection.
        """
        self.token_position = token_position
        self.remove_outliers = remove_outliers
        self.outlier_std_threshold = outlier_std_threshold

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

        # Optional: Remove extreme outliers before averaging
        outliers_removed = 0
        if self.remove_outliers and len(positive_activations) > 3:
            positive_activations, negative_activations, outliers_removed = (
                self._remove_extreme_outliers(positive_activations, negative_activations)
            )

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
                "num_pairs_used": len(positive_activations),
                "outliers_removed": outliers_removed,
                "vector_norm": vector.norm().item(),
            },
        )

    def _remove_extreme_outliers(
        self,
        positive_acts: List[torch.Tensor],
        negative_acts: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """Remove extreme outliers based on activation difference norms.

        Identifies pairs where the (positive - negative) activation difference
        has an L2 norm that is more than `outlier_std_threshold` standard
        deviations from the mean. Only removes EXTREME outliers.

        Args:
            positive_acts: List of positive activation tensors.
            negative_acts: List of negative activation tensors.

        Returns:
            Tuple of (filtered_positive, filtered_negative, num_removed).
        """
        # Compute per-pair difference norms
        differences = [p - n for p, n in zip(positive_acts, negative_acts)]
        norms = torch.tensor([d.norm().item() for d in differences])

        # Compute statistics
        mean_norm = norms.mean()
        std_norm = norms.std()

        # Identify non-outliers (within threshold)
        threshold = mean_norm + self.outlier_std_threshold * std_norm
        mask = norms <= threshold

        # Filter
        filtered_pos = [p for p, keep in zip(positive_acts, mask) if keep]
        filtered_neg = [n for n, keep in zip(negative_acts, mask) if keep]
        num_removed = len(positive_acts) - len(filtered_pos)

        # Safety: keep at least 3 pairs
        if len(filtered_pos) < 3:
            return positive_acts, negative_acts, 0

        return filtered_pos, filtered_neg, num_removed

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
        """Select activation based on token_position strategy.

        Args:
            activations: Hidden states tensor of shape [seq_len, hidden_dim].
            prompt_len: Number of prompt tokens.
            full_len: Total sequence length (prompt + response).

        Returns:
            Selected activation tensor of shape [hidden_dim].

        Note:
            Uses bounds checking to handle cases where activations tensor
            may be shorter than expected (due to model truncation, etc.).
        """
        actual_len = activations.shape[0]

        # Bounds check: ensure indices are within activations tensor
        if actual_len < prompt_len:
            # Fallback: use last available activation
            import logging
            logging.warning(
                f"Activation length ({actual_len}) < prompt_len ({prompt_len}). "
                "Using last available activation."
            )
            return activations[-1]

        if self.token_position == "last_prompt_token":
            # Last token of prompt
            idx = min(prompt_len - 1, actual_len - 1)
            return activations[idx]
        elif self.token_position == "last":
            # Last token of full sequence
            idx = min(full_len - 1, actual_len - 1)
            return activations[idx]
        elif self.token_position == "mean":
            # Mean of response tokens
            end_idx = min(full_len, actual_len)
            start_idx = min(prompt_len, end_idx)
            if start_idx >= end_idx:
                # No response tokens available, use last prompt token
                return activations[min(prompt_len - 1, actual_len - 1)]
            response_activations = activations[start_idx:end_idx]
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
