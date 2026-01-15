"""Probability-based evaluation utilities."""

from typing import List, Optional, Dict, Tuple

import torch

from steerex.backends.base import ModelBackend
from steerex.steering.base import SteeringMode


def get_completion_probability(
    backend: ModelBackend,
    prompt: str,
    completion: str,
    steering_mode: Optional[SteeringMode] = None,
    layer: Optional[int] = None,
    strength: float = 1.0,
    coldness: float = 1.0,
    log_prob: bool = True,
) -> float:
    """
    Compute probability of completion given prompt.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        completion: Target completion.
        steering_mode: Optional steering to apply.
        layer: Layer for steering.
        strength: Steering strength.
        coldness: Inverse temperature.
        log_prob: If True, return log probability.

    Returns:
        (Log) probability of completion.
    """
    if steering_mode and layer is not None:
        hooks = [(layer, steering_mode.create_hook(strength=strength))]
    else:
        hooks = None

    return backend.get_completion_probability(
        prompt,
        completion,
        hooks=hooks,
        coldness=coldness,
        log_prob=log_prob,
    )


def compare_probabilities(
    backend: ModelBackend,
    prompt: str,
    completions: List[str],
    steering_mode: Optional[SteeringMode] = None,
    layer: Optional[int] = None,
    strength: float = 1.0,
    coldness: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Compare probabilities of multiple completions.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        completions: List of target completions.
        steering_mode: Optional steering.
        layer: Layer for steering.
        strength: Steering strength.
        coldness: Inverse temperature.

    Returns:
        Dictionary mapping completion to {"log_prob": ..., "prob": ...}.
    """
    results = {}

    for completion in completions:
        log_prob = get_completion_probability(
            backend,
            prompt,
            completion,
            steering_mode=steering_mode,
            layer=layer,
            strength=strength,
            coldness=coldness,
            log_prob=True,
        )
        results[completion] = {
            "log_prob": log_prob,
            "prob": torch.exp(torch.tensor(log_prob)).item(),
        }

    return results


def probability_shift(
    backend: ModelBackend,
    prompt: str,
    completions: List[str],
    steering_mode: SteeringMode,
    layer: int,
    strength: float = 1.0,
    coldness: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Measure how steering changes completion probabilities.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        completions: Target completions to measure.
        steering_mode: Steering to apply.
        layer: Layer for steering.
        strength: Steering strength.
        coldness: Inverse temperature.

    Returns:
        Dictionary mapping completion to probability shift info.
    """
    results = {}

    for completion in completions:
        # Baseline (unsteered)
        baseline_log_prob = get_completion_probability(
            backend, prompt, completion,
            coldness=coldness, log_prob=True,
        )

        # Steered
        steered_log_prob = get_completion_probability(
            backend, prompt, completion,
            steering_mode=steering_mode,
            layer=layer,
            strength=strength,
            coldness=coldness,
            log_prob=True,
        )

        results[completion] = {
            "baseline_log_prob": baseline_log_prob,
            "steered_log_prob": steered_log_prob,
            "log_prob_diff": steered_log_prob - baseline_log_prob,
            "prob_ratio": torch.exp(
                torch.tensor(steered_log_prob - baseline_log_prob)
            ).item(),
        }

    return results


def rank_completions(
    backend: ModelBackend,
    prompt: str,
    completions: List[str],
    steering_mode: Optional[SteeringMode] = None,
    layer: Optional[int] = None,
    strength: float = 1.0,
    coldness: float = 1.0,
) -> List[Tuple[str, float]]:
    """
    Rank completions by probability.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        completions: Completions to rank.
        steering_mode: Optional steering.
        layer: Layer for steering.
        strength: Steering strength.
        coldness: Inverse temperature.

    Returns:
        List of (completion, log_prob) sorted by probability descending.
    """
    probs = compare_probabilities(
        backend, prompt, completions,
        steering_mode, layer, strength, coldness
    )

    ranked = [
        (comp, probs[comp]["log_prob"])
        for comp in completions
    ]
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked
