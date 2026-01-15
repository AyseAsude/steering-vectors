# steerex

> **Attribution**: This library is a modular reimplementation of [llm-steering-opt](https://github.com/jacobdunefsky/llm-steering-opt) by Jacob Dunefsky. Full credit goes to the original author for the steering vector optimization algorithms.

A research platform for LLM activation engineering and steering vector extraction.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### CAA Extraction

Extract steering vectors using Contrastive Activation Addition (difference of means):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from steerex import extract, ContrastPair, HuggingFaceBackend

# Load model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
backend = HuggingFaceBackend(model, tokenizer)

# Define contrast pairs
pairs = [
    ContrastPair.from_messages(
        positive=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help?"},
        ],
        negative=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Go away."},
        ],
    ),
]

# Extract and use
result = extract(backend, tokenizer, pairs, layer=16)
steering = result.to_steering()

output = backend.generate_with_steering(
    "Hello!",
    steering_mode=steering,
    layers=16,
    max_new_tokens=50,
)
```

### Gradient Optimization

Learn steering vectors through optimization:

```python
from steerex import (
    SteeringOptimizer,
    VectorSteering,
    HuggingFaceBackend,
    TrainingDatapoint,
    OptimizationConfig,
)

backend = HuggingFaceBackend(model, tokenizer)
steering = VectorSteering()
config = OptimizationConfig(lr=0.1, max_iters=50)

datapoint = TrainingDatapoint(
    prompt="My favorite animal is",
    dst_completions=[" definitely cats!"],  # Promote
    src_completions=[" definitely dogs!"],  # Suppress
)

optimizer = SteeringOptimizer(backend, steering, config)
result = optimizer.optimize([datapoint], layer=10)

output = backend.generate_with_steering(
    "My favorite animal is",
    steering_mode=steering,
    layers=10,
    max_new_tokens=30,
)
```

## Extraction Methods

| Method | How it works |
|--------|--------------|
| CAA | `mean(positive) - mean(negative)` activations |
| Gradient | Optimizes vector to promote/suppress completions |

## Steering Modes

```python
from steerex import VectorSteering, ClampSteering, AffineSteering

VectorSteering()   # Additive: activation += strength * vector
ClampSteering()    # Projects activations toward vector direction
AffineSteering()   # Learned affine transformation
```
