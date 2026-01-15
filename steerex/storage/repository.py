"""Vector repository for storage and retrieval."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import json
import sqlite3
from datetime import datetime

import torch

from steerex.storage.models import VectorMetadata, VectorRecord
from steerex.core.result import OptimizationResult


class VectorRepository(ABC):
    """
    Abstract base class for vector storage.

    Provides CRUD operations for steering vectors with metadata.
    """

    @abstractmethod
    def save(
        self,
        vector: torch.Tensor,
        name: str,
        model_id: str,
        layer: int,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """
        Save a vector with metadata.

        Args:
            vector: The steering vector tensor.
            name: Human-readable name.
            model_id: HuggingFace model identifier.
            layer: Layer the vector was trained for.
            tags: Organizational tags.
            description: Optional description.
            training_config: Serialized training configuration.
            **kwargs: Additional metadata fields.

        Returns:
            The ID of the saved vector.
        """
        ...

    @abstractmethod
    def save_result(
        self,
        result: OptimizationResult,
        name: str,
        model_id: str,
        layer: int,
        **kwargs,
    ) -> str:
        """
        Save an OptimizationResult directly.

        Convenience method that extracts vector and metadata from result.
        """
        ...

    @abstractmethod
    def get(self, vector_id: str) -> torch.Tensor:
        """
        Retrieve a vector by ID.

        Args:
            vector_id: The vector's unique ID.

        Returns:
            The steering vector tensor.

        Raises:
            KeyError: If vector not found.
        """
        ...

    @abstractmethod
    def get_by_name(self, name: str) -> torch.Tensor:
        """
        Retrieve a vector by name.

        Args:
            name: The vector's name.

        Returns:
            The steering vector tensor.

        Raises:
            KeyError: If vector not found.
        """
        ...

    @abstractmethod
    def get_metadata(self, vector_id: str) -> VectorMetadata:
        """
        Get metadata for a vector.

        Args:
            vector_id: The vector's unique ID.

        Returns:
            VectorMetadata object.
        """
        ...

    @abstractmethod
    def list(
        self,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorMetadata]:
        """
        List vectors matching criteria.

        Args:
            model_id: Filter by model.
            layer: Filter by layer.
            tags: Filter by tags (AND logic).
            limit: Maximum results.
            offset: Skip first N results.

        Returns:
            List of matching VectorMetadata.
        """
        ...

    @abstractmethod
    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector.

        Args:
            vector_id: The vector's unique ID.

        Returns:
            True if deleted, False if not found.
        """
        ...

    @abstractmethod
    def update_metadata(
        self,
        vector_id: str,
        **updates,
    ) -> VectorMetadata:
        """
        Update vector metadata.

        Args:
            vector_id: The vector's unique ID.
            **updates: Fields to update.

        Returns:
            Updated VectorMetadata.
        """
        ...

    # Similarity operations

    @abstractmethod
    def get_similar(
        self,
        vector: torch.Tensor,
        top_k: int = 5,
        model_id: Optional[str] = None,
    ) -> List[tuple]:
        """
        Find vectors similar to the given vector.

        Args:
            vector: Query vector.
            top_k: Number of results.
            model_id: Optionally filter by model.

        Returns:
            List of (VectorMetadata, similarity_score) tuples.
        """
        ...


class SQLiteRepository(VectorRepository):
    """
    SQLite-based vector repository.

    Stores metadata in SQLite, tensors as .pt files.

    Directory structure:
        base_path/
            vectors.db       # SQLite database
            tensors/
                <uuid>.pt    # Tensor files
    """

    def __init__(self, base_path: Union[str, Path]):
        """
        Initialize repository.

        Args:
            base_path: Directory for database and tensors.
        """
        self.base_path = Path(base_path)
        self.db_path = self.base_path / "vectors.db"
        self.tensors_path = self.base_path / "tensors"

        # Create directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.tensors_path.mkdir(exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    model_id TEXT NOT NULL,
                    layer INTEGER NOT NULL,
                    hidden_dim INTEGER NOT NULL,
                    tensor_path TEXT NOT NULL,
                    training_config TEXT,
                    final_loss REAL,
                    norm REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS vector_tags (
                    vector_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    PRIMARY KEY (vector_id, tag),
                    FOREIGN KEY (vector_id) REFERENCES vectors(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_model ON vectors(model_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_vectors_layer ON vectors(layer)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_tags_tag ON vector_tags(tag)
            """)

    def save(
        self,
        vector: torch.Tensor,
        name: str,
        model_id: str,
        layer: int,
        tags: Optional[List[str]] = None,
        description: Optional[str] = None,
        training_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Save vector to repository."""
        import uuid

        vector_id = str(uuid.uuid4())
        tensor_filename = f"{vector_id}.pt"
        tensor_path = self.tensors_path / tensor_filename

        # Save tensor
        torch.save(vector.detach().cpu(), tensor_path)

        # Save metadata
        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO vectors
                (id, name, description, model_id, layer, hidden_dim, tensor_path,
                 training_config, final_loss, norm, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    vector_id,
                    name,
                    description,
                    model_id,
                    layer,
                    vector.shape[0],
                    tensor_filename,
                    json.dumps(training_config) if training_config else None,
                    kwargs.get("final_loss"),
                    vector.norm().item(),
                    now,
                    now,
                ),
            )

            # Save tags
            if tags:
                conn.executemany(
                    "INSERT INTO vector_tags (vector_id, tag) VALUES (?, ?)",
                    [(vector_id, tag) for tag in tags],
                )

        return vector_id

    def save_result(
        self,
        result: OptimizationResult,
        name: str,
        model_id: str,
        layer: int,
        **kwargs,
    ) -> str:
        """Save optimization result."""
        return self.save(
            vector=result.vector,
            name=name,
            model_id=model_id,
            layer=layer,
            training_config=result.metadata.get("config"),
            final_loss=result.final_loss,
            **kwargs,
        )

    def get(self, vector_id: str) -> torch.Tensor:
        """Get vector by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT tensor_path FROM vectors WHERE id = ?",
                (vector_id,),
            ).fetchone()

        if not row:
            raise KeyError(f"Vector not found: {vector_id}")

        tensor_path = self.tensors_path / row[0]
        return torch.load(tensor_path, weights_only=True)

    def get_by_name(self, name: str) -> torch.Tensor:
        """Get vector by name."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT tensor_path FROM vectors WHERE name = ?",
                (name,),
            ).fetchone()

        if not row:
            raise KeyError(f"Vector not found: {name}")

        tensor_path = self.tensors_path / row[0]
        return torch.load(tensor_path, weights_only=True)

    def get_metadata(self, vector_id: str) -> VectorMetadata:
        """Get metadata by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM vectors WHERE id = ?",
                (vector_id,),
            ).fetchone()

            if not row:
                raise KeyError(f"Vector not found: {vector_id}")

            tags = [
                r[0]
                for r in conn.execute(
                    "SELECT tag FROM vector_tags WHERE vector_id = ?",
                    (vector_id,),
                ).fetchall()
            ]

        return VectorMetadata(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            model_id=row["model_id"],
            layer=row["layer"],
            hidden_dim=row["hidden_dim"],
            tensor_path=row["tensor_path"],
            training_config=json.loads(row["training_config"])
            if row["training_config"]
            else None,
            final_loss=row["final_loss"],
            norm=row["norm"],
            tags=tags,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def list(
        self,
        model_id: Optional[str] = None,
        layer: Optional[int] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[VectorMetadata]:
        """List vectors matching criteria."""
        conditions = []
        params = []

        if model_id:
            conditions.append("model_id = ?")
            params.append(model_id)

        if layer is not None:
            conditions.append("layer = ?")
            params.append(layer)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if tags:
                # Filter by tags using subquery
                tag_placeholders = ",".join("?" * len(tags))
                query = f"""
                    SELECT v.* FROM vectors v
                    WHERE {where_clause}
                    AND v.id IN (
                        SELECT vector_id FROM vector_tags
                        WHERE tag IN ({tag_placeholders})
                        GROUP BY vector_id
                        HAVING COUNT(DISTINCT tag) = ?
                    )
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """
                params.extend(tags)
                params.append(len(tags))
            else:
                query = f"""
                    SELECT * FROM vectors
                    WHERE {where_clause}
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """

            params.extend([limit, offset])
            rows = conn.execute(query, params).fetchall()

            results = []
            for row in rows:
                tags_for_vector = [
                    r[0]
                    for r in conn.execute(
                        "SELECT tag FROM vector_tags WHERE vector_id = ?",
                        (row["id"],),
                    ).fetchall()
                ]
                results.append(
                    VectorMetadata(
                        id=row["id"],
                        name=row["name"],
                        description=row["description"],
                        model_id=row["model_id"],
                        layer=row["layer"],
                        hidden_dim=row["hidden_dim"],
                        tensor_path=row["tensor_path"],
                        final_loss=row["final_loss"],
                        norm=row["norm"],
                        tags=tags_for_vector,
                        created_at=datetime.fromisoformat(row["created_at"]),
                        updated_at=datetime.fromisoformat(row["updated_at"]),
                    )
                )

        return results

    def delete(self, vector_id: str) -> bool:
        """Delete vector by ID."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT tensor_path FROM vectors WHERE id = ?",
                (vector_id,),
            ).fetchone()

            if not row:
                return False

            # Delete tensor file
            tensor_path = self.tensors_path / row[0]
            if tensor_path.exists():
                tensor_path.unlink()

            # Delete from database (tags deleted via CASCADE)
            conn.execute("DELETE FROM vectors WHERE id = ?", (vector_id,))

        return True

    def update_metadata(
        self,
        vector_id: str,
        **updates,
    ) -> VectorMetadata:
        """Update metadata fields."""
        allowed_fields = {"name", "description", "tags"}
        updates = {k: v for k, v in updates.items() if k in allowed_fields}

        if not updates:
            return self.get_metadata(vector_id)

        now = datetime.utcnow().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Handle tags separately
            if "tags" in updates:
                conn.execute(
                    "DELETE FROM vector_tags WHERE vector_id = ?",
                    (vector_id,),
                )
                conn.executemany(
                    "INSERT INTO vector_tags (vector_id, tag) VALUES (?, ?)",
                    [(vector_id, tag) for tag in updates.pop("tags")],
                )

            # Update other fields
            if updates:
                set_clause = ", ".join(f"{k} = ?" for k in updates)
                params = list(updates.values()) + [now, vector_id]
                conn.execute(
                    f"UPDATE vectors SET {set_clause}, updated_at = ? WHERE id = ?",
                    params,
                )

        return self.get_metadata(vector_id)

    def get_similar(
        self,
        vector: torch.Tensor,
        top_k: int = 5,
        model_id: Optional[str] = None,
    ) -> List[tuple]:
        """Find similar vectors by cosine similarity."""
        # Get all vectors (optionally filtered by model)
        candidates = self.list(model_id=model_id, limit=1000)

        if not candidates:
            return []

        # Compute similarities
        query_norm = vector / vector.norm()
        similarities = []

        for meta in candidates:
            stored_vector = self.get(meta.id)
            stored_norm = stored_vector / stored_vector.norm()
            sim = torch.dot(query_norm.cpu(), stored_norm.cpu()).item()
            similarities.append((meta, sim))

        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]
