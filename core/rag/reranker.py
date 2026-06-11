"""Cross-Encoder re-ranking for initial hybrid search results."""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

from core.log import logger


class RerankerException(Exception):
    """Re-ranking error."""


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """Minimal Cross-Encoder interface for dependency injection."""

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """Return a relevance score for each [query, passage] pair."""


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Re-rank initial retrieval results with a Cross-Encoder model."""

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        cross_encoder: CrossEncoderProtocol | None = None,
    ) -> None:
        """Initialize the Cross-Encoder re-ranker."""
        self._model_name = model_name
        self._cross_encoder = cross_encoder

    def _load_model(self) -> CrossEncoderProtocol:
        """Lazy-load the Cross-Encoder model."""
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except (ImportError, TypeError) as error:
            raise RerankerException(
                "sentence_transformers is not installed; Cross-Encoder model loading "
                "is unavailable. Run: uv add sentence-transformers"
            ) from error

        try:
            model = CrossEncoder(self._model_name)
        except Exception as error:
            logger.error("Cross-Encoder model loading failed: %s", error)
            raise RerankerException(
                f"Cross-Encoder model '{self._model_name}' failed to load."
            ) from error

        return model  # type: ignore[return-value]

    def rerank(
        self,
        query: str,
        chunks: Sequence[object],
        top_n: int = 3,
    ) -> list[object]:
        """Run Cross-Encoder re-ranking over initial retrieval chunks."""
        if not chunks:
            return []

        if top_n <= 0:
            logger.info("Re-Ranking skipped because top_n <= 0: %d", top_n)
            return []

        if self._cross_encoder is None:
            self._cross_encoder = self._load_model()
        cross_encoder = self._cross_encoder

        pairs = [[query, chunk.page_content] for chunk in chunks]  # type: ignore[union-attr]

        try:
            raw_scores = cross_encoder.predict(pairs)
        except Exception as error:
            logger.error(
                "Cross-Encoder prediction failed for model %s "
                "(top_n=%s, chunk_count=%s, query_length=%s): %s",
                self._model_name,
                top_n,
                len(chunks),
                len(query),
                error,
            )
            raise RerankerException(
                "Cross-Encoder prediction failed for model "
                f"{self._model_name} (top_n={top_n}, "
                f"chunk_count={len(chunks)}, query_length={len(query)})."
            ) from error

        scored = sorted(
            zip(raw_scores, chunks),
            key=lambda pair: float(pair[0]),
            reverse=True,
        )

        reranked: list[object] = []
        from dataclasses import replace  # noqa: PLC0415

        for new_rank, (score, chunk) in enumerate(scored[:top_n], start=1):
            updated_chunk = replace(  # type: ignore[call-overload]
                chunk,
                rerank_score=float(score),
                rank=new_rank,
            )
            reranked.append(updated_chunk)

        logger.info(
            "Re-Ranking completed with %d input chunks and %d returned chunks.",
            len(chunks),
            len(reranked),
        )
        return reranked
