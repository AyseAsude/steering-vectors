"""Generation utilities for evaluation."""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

import torch

from steerex.backends.base import ModelBackend
from steerex.steering.base import SteeringMode
from steerex.core.types import LayerSpec


@dataclass
class GenerationResult:
    """Result from a generation experiment."""

    prompt: str
    generations: List[str]
    steering_strength: float
    layer: int
    metadata: Dict[str, Any]


def generate_samples(
    backend: ModelBackend,
    prompt: str,
    steering_mode: Optional[SteeringMode] = None,
    layer: Optional[int] = None,
    strength: float = 1.0,
    n_samples: int = 10,
    batch_size: int = 5,
    **generation_kwargs,
) -> List[str]:
    """
    Generate multiple samples, optionally with steering.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        steering_mode: Optional steering to apply.
        layer: Layer to apply steering at.
        strength: Steering strength.
        n_samples: Number of samples to generate.
        batch_size: Samples per batch.
        **generation_kwargs: Args for backend.generate().

    Returns:
        List of generated texts.
    """
    generations = []

    for _ in range(0, n_samples, batch_size):
        batch_count = min(batch_size, n_samples - len(generations))

        for _ in range(batch_count):
            if steering_mode and layer is not None:
                text = backend.generate_with_steering(
                    prompt,
                    steering_mode=steering_mode,
                    layers=layer,
                    strength=strength,
                    **generation_kwargs,
                )
            else:
                text = backend.generate(prompt, **generation_kwargs)

            generations.append(text)

    return generations


def generate_with_multiple_vectors(
    backend: ModelBackend,
    prompt: str,
    steering_modes: List[Tuple[SteeringMode, int]],
    strengths: List[float],
    n_samples: int = 5,
    **generation_kwargs,
) -> Dict[str, List[str]]:
    """
    Generate with multiple steering vectors and strengths.

    Useful for comparing different vectors or strength levels.

    Args:
        backend: Model backend.
        prompt: Input prompt.
        steering_modes: List of (steering_mode, layer) tuples.
        strengths: List of steering strengths to try.
        n_samples: Samples per configuration.
        **generation_kwargs: Generation arguments.

    Returns:
        Dictionary mapping config description to generations.
    """
    results = {}

    # Unsteered baseline
    baseline = generate_samples(
        backend, prompt, n_samples=n_samples, **generation_kwargs
    )
    results["unsteered"] = baseline

    # Steered
    for i, (steering_mode, layer) in enumerate(steering_modes):
        for strength in strengths:
            key = f"vector_{i}_layer_{layer}_strength_{strength}"
            gens = generate_samples(
                backend,
                prompt,
                steering_mode=steering_mode,
                layer=layer,
                strength=strength,
                n_samples=n_samples,
                **generation_kwargs,
            )
            results[key] = gens

    return results


def compare_steered_vs_unsteered(
    backend: ModelBackend,
    prompts: List[str],
    steering_mode: SteeringMode,
    layer: int,
    strength: float = 1.0,
    n_samples: int = 5,
    **generation_kwargs,
) -> Dict[str, Dict[str, List[str]]]:
    """
    Compare steered vs unsteered generations for multiple prompts.

    Args:
        backend: Model backend.
        prompts: List of prompts.
        steering_mode: Steering to apply.
        layer: Layer for steering.
        strength: Steering strength.
        n_samples: Samples per prompt per condition.
        **generation_kwargs: Generation arguments.

    Returns:
        Dictionary mapping prompt to {"steered": [...], "unsteered": [...]}.
    """
    results = {}

    for prompt in prompts:
        unsteered = generate_samples(
            backend, prompt, n_samples=n_samples, **generation_kwargs
        )
        steered = generate_samples(
            backend,
            prompt,
            steering_mode=steering_mode,
            layer=layer,
            strength=strength,
            n_samples=n_samples,
            **generation_kwargs,
        )
        results[prompt] = {
            "unsteered": unsteered,
            "steered": steered,
        }

    return results


def batch_generate(
    backend: ModelBackend,
    prompts: List[str],
    steering_mode: Optional[SteeringMode] = None,
    layer: Optional[int] = None,
    strength: float = 1.0,
    **generation_kwargs,
) -> List[str]:
    """
    Generate one completion per prompt (batch mode).

    Args:
        backend: Model backend.
        prompts: List of prompts.
        steering_mode: Optional steering.
        layer: Layer for steering.
        strength: Steering strength.
        **generation_kwargs: Generation arguments.

    Returns:
        List of generated texts.
    """
    # Note: This is a simple implementation that doesn't do true batching.
    # For efficiency with many prompts, implement batch generation in backend.
    results = []

    for prompt in prompts:
        if steering_mode and layer is not None:
            text = backend.generate_with_steering(
                prompt,
                steering_mode=steering_mode,
                layers=layer,
                strength=strength,
                **generation_kwargs,
            )
        else:
            text = backend.generate(prompt, **generation_kwargs)
        results.append(text)

    return results
