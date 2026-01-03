# steering-vectors

> **Attribution**: This library is a modular reimplementation of [llm-steering-opt](https://github.com/jacobdunefsky/llm-steering-opt) by Jacob Dunefsky. Full credit goes to the original author for the steering vector optimization algorithms.

A research platform for LLM activation engineering and steering vector optimization.

## Installation

```bash
uv pip install -e ".[dev]"
```

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import (
    SteeringOptimizer,
    HuggingFaceBackend,
    VectorSteering,
    TrainingDatapoint,
    OptimizationConfig,
)

# Load model
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# Setup
backend = HuggingFaceBackend(model, tokenizer)
steering = VectorSteering()
config = OptimizationConfig(lr=0.1, max_iters=50)

# Define training data
datapoint = TrainingDatapoint(
    prompt="My favorite animal is",
    dst_completions=[" definitely cats!"],  # Promote
    src_completions=[" definitely dogs!"],  # Suppress
)

# Optimize
optimizer = SteeringOptimizer(backend, steering, config)
result = optimizer.optimize([datapoint], layer=10)

# Generate with steering
hook = steering.create_hook()
with backend.hooks_context([(10, hook)]):
    output = model.generate(...)
```