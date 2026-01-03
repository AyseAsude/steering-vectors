"""Data models for vector storage."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import uuid


class VectorMetadata(BaseModel):
    """
    Metadata for a stored steering vector.

    This captures everything needed to understand and reproduce
    a steering vector.
    """

    # Identity
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None

    # Model info
    model_id: str  # e.g., "Qwen/Qwen2.5-14B-Instruct"
    layer: int
    hidden_dim: int

    # Training info
    training_prompts: Optional[List[str]] = None
    training_completions: Optional[Dict[str, List[str]]] = None  # {"dst": [...], "src": [...]}
    training_config: Optional[Dict[str, Any]] = None  # Serialized OptimizationConfig

    # Organization
    tags: List[str] = Field(default_factory=list)

    # Metrics
    final_loss: Optional[float] = None
    norm: Optional[float] = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # File reference
    tensor_path: Optional[str] = None  # Relative path to .pt file

    class Config:
        frozen = False


class VectorRecord(BaseModel):
    """
    Full record including metadata and tensor path.

    Used internally by the repository.
    """

    metadata: VectorMetadata
    tensor_path: str

    def __repr__(self):
        return f"VectorRecord(name={self.metadata.name!r}, model={self.metadata.model_id!r})"
