"""Type aliases and common types."""

from typing import Union, List

import torch

# Layer specification: single layer or multiple layers
LayerSpec = Union[int, List[int]]

# Token specification: which tokens to apply steering to
TokenSpec = Union[slice, int, None]

# Tensor type alias for clarity
Tensor = torch.Tensor
