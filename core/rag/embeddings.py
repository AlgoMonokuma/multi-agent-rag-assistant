"""Text embedding utilities."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from core.log import logger
from core.rag.parser import ParsedDocument


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingException(Exception):
    """Text embedding error."""


class SentenceTransformerEmbedder:
    """Wrap the sentence-transformers embedding workflow."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        expected_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        model: Any | None = None,
    ) -> None:
        """Initialize the embedder."""
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._model = model

    def embed_documents(self, documents: Sequence[ParsedDocument]) -> np.ndarray:
        """Embed chunk documents into a float32 matrix."""
        texts = [document.page_content for document in documents]
        return self.embed_texts(texts)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed text values into a float32 matrix."""
        model = self._model or self._load_model()

        try:
            vectors = model.encode(list(texts))
        except Exception as error:
            logger.error("Embedding generation failed: %s", error)
            raise EmbeddingException("Embedding generation failed.") from error

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise EmbeddingException("Embedding vectors must be a two-dimensional matrix.")

        if matrix.shape[1] != self._expected_dimension:
            raise EmbeddingException(
                f"Embedding dimension mismatch: expected {self._expected_dimension}, "
                f"got {matrix.shape[1]}."
            )

        logger.info(
            "Generated %s embeddings with dimension %s.",
            matrix.shape[0],
            matrix.shape[1],
        )
        return matrix

    def _load_model(self) -> Any:
        """Lazy-load the default sentence-transformers model."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingException(
                "sentence-transformers is not installed; embeddings cannot be generated."
            ) from error

        self._model = SentenceTransformer(self._model_name)
        return self._model
