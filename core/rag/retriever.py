"""Session-scoped hybrid retrieval with vector and keyword search."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

from core.log import logger

if TYPE_CHECKING:
    from core.rag.reranker import CrossEncoderReranker


class RetrieverException(Exception):
    """Hybrid retrieval error."""


class EmbedderProtocol(Protocol):
    """Minimal embedder interface."""

    def embed_texts(self, texts: Sequence[str]) -> Any:
        """Embed text values into vectors."""


class IndexerProtocol(Protocol):
    """Minimal SessionIndexer interface."""

    def get_session(self, session_id: str) -> Any:
        """Return a session record."""

    def get_chunk_id_by_ordinal(self, session_id: str, ordinal: int) -> str:
        """Return chunk_id for a vector ordinal."""

    def get_chunk_document(self, session_id: str, chunk_id: str) -> Any:
        """Return the chunk ParsedDocument."""

    def get_chunk_metadata(self, session_id: str, chunk_id: str) -> Dict[str, Any]:
        """Return chunk metadata."""

    def list_chunk_documents(self, session_id: str) -> List[Any]:
        """Return all chunk ParsedDocument objects for a session."""

    def list_vector_chunk_ids(self, session_id: str) -> List[str]:
        """Return the vector ordinal mapping for a session."""


@dataclass
class RetrievedChunk:
    """Single retrieval result with scores and citation metadata."""

    chunk_id: str
    page_content: str
    metadata: Dict[str, Any]
    vector_score: float
    keyword_score: float
    merged_score: float
    rank: int
    rerank_score: Optional[float] = None


@dataclass(slots=True)
class HybridSearchResult:
    """Complete hybrid search response."""

    query: str
    session_id: str
    results: List[RetrievedChunk]
    total_found: int


DEFAULT_TOP_K = 10
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3


class HybridRetriever:
    """Combine vector and keyword search for one session."""

    def __init__(
        self,
        session_indexer: Any,
        embedder: Any,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        reranker: Optional["CrossEncoderReranker"] = None,
        final_top_n: int = 3,
    ) -> None:
        """Initialize the hybrid retriever."""
        self._indexer = session_indexer
        self._embedder = embedder
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight
        self._reranker = reranker
        self._final_top_n = final_top_n

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        vector_weight: float | None = None,
        keyword_weight: float | None = None,
    ) -> HybridSearchResult:
        """Run hybrid retrieval and return ranked top-k results."""
        self._validate_query(query)
        self._validate_session(session_id)

        record = self._indexer.get_session(session_id)

        if vector_weight is not None:
            effective_vector_weight = vector_weight
        elif getattr(record, "vector_weight", None) is not None:
            effective_vector_weight = record.vector_weight
        else:
            effective_vector_weight = self._vector_weight

        if keyword_weight is not None:
            effective_keyword_weight = keyword_weight
        elif getattr(record, "keyword_weight", None) is not None:
            effective_keyword_weight = record.keyword_weight
        else:
            effective_keyword_weight = self._keyword_weight

        if top_k <= 0:
            return HybridSearchResult(
                query=query,
                session_id=session_id,
                results=[],
                total_found=0,
            )

        if not record.vector_chunk_ids and not record.chunk_map:
            logger.info("Session %s has no ingested chunks; returning empty results.", session_id)
            return HybridSearchResult(
                query=query,
                session_id=session_id,
                results=[],
                total_found=0,
            )

        vector_hits = self._vector_search(session_id, record, query, top_k)
        keyword_hits = self._keyword_search(session_id, record, query, top_k)

        merged = self._merge_results(
            session_id=session_id,
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            top_k=top_k,
            vector_weight=effective_vector_weight,
            keyword_weight=effective_keyword_weight,
        )

        initial_total = len(merged)

        if self._reranker is not None:
            from core.rag.reranker import RerankerException  # noqa: PLC0415

            try:
                merged = self._reranker.rerank(  # type: ignore[assignment]
                    query=query,
                    chunks=merged,
                    top_n=self._final_top_n,
                )
            except RerankerException as error:
                logger.error(
                    "Session %s Re-Ranking failed; returning hybrid search results: %s",
                    session_id,
                    error,
                )

        logger.info(
            "Session %s hybrid search completed for query '%s' with %s returned "
            "results from %s initial results.",
            session_id,
            query,
            len(merged),
            initial_total,
        )

        return HybridSearchResult(
            query=query,
            session_id=session_id,
            results=merged,
            total_found=initial_total,
        )

    def _vector_search(
        self,
        session_id: str,
        record: Any,
        query: str,
        top_k: int,
    ) -> Dict[str, float]:
        """Search the FAISS index and return normalized vector scores."""
        if not record.vector_chunk_ids:
            return {}

        try:
            query_vector = self._embedder.embed_texts([query])
        except Exception as error:
            logger.error("Session %s query embedding failed: %s", session_id, error)
            raise RetrieverException(
                f"Session {session_id} query embedding failed."
            ) from error

        actual_k = min(top_k, len(record.vector_chunk_ids))

        try:
            distances, indices = record.index.search(query_vector, actual_k)
        except Exception as error:
            logger.error("Session %s FAISS search failed: %s", session_id, error)
            raise RetrieverException(
                f"Session {session_id} FAISS search failed."
            ) from error

        hits: Dict[str, float] = {}
        for i in range(actual_k):
            ordinal = int(indices[0, i])
            distance = float(distances[0, i])

            if ordinal < 0:
                continue

            try:
                chunk_id = self._indexer.get_chunk_id_by_ordinal(session_id, ordinal)
            except Exception:
                logger.warning(
                    "Session %s ordinal %s could not be mapped to chunk_id; skipping.",
                    session_id,
                    ordinal,
                )
                continue

            score = 1.0 / (1.0 + distance)
            hits[chunk_id] = score

        return hits

    def _keyword_search(
        self,
        session_id: str,
        record: Any,
        query: str,
        top_k: int,
    ) -> Dict[str, float]:
        """Search session chunk text with a lightweight BM25-style score."""
        if not record.chunk_map:
            return {}

        query_terms = self._tokenize(query)
        if not query_terms:
            return {}

        unique_query_terms = set(query_terms)
        chunk_entries: List[tuple[str, str]] = []
        for chunk_id, doc in record.chunk_map.items():
            chunk_entries.append((chunk_id, doc.page_content))

        doc_count = len(chunk_entries)
        doc_freqs: Counter[str] = Counter()
        doc_term_counts: List[tuple[str, Counter[str]]] = []

        for chunk_id, text in chunk_entries:
            terms = self._tokenize(text)
            term_counter = Counter(terms)
            doc_term_counts.append((chunk_id, term_counter))
            for term in set(terms):
                doc_freqs[term] += 1

        avg_dl = sum(sum(tc.values()) for _, tc in doc_term_counts) / max(doc_count, 1)
        k1 = 1.5
        b = 0.75

        scores: Dict[str, float] = {}
        for chunk_id, term_counter in doc_term_counts:
            doc_len = sum(term_counter.values())
            score = 0.0

            for q_term in unique_query_terms:
                tf = term_counter.get(q_term, 0)
                if tf == 0:
                    continue

                df = doc_freqs.get(q_term, 0)
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1))
                )
                score += idf * tf_norm

            if score > 0.0:
                scores[chunk_id] = score

        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {cid: s / max_score for cid, s in scores.items()}

        return scores

    def _merge_results(
        self,
        session_id: str,
        vector_hits: Dict[str, float],
        keyword_hits: Dict[str, float],
        top_k: int,
        vector_weight: float,
        keyword_weight: float,
    ) -> List[RetrievedChunk]:
        """Merge vector and keyword hits into deterministic ranked results."""
        all_chunk_ids = set(vector_hits.keys()) | set(keyword_hits.keys())

        if not all_chunk_ids:
            return []

        merged_entries: List[tuple[str, float, float, float]] = []
        for chunk_id in all_chunk_ids:
            v_score = vector_hits.get(chunk_id, 0.0)
            k_score = keyword_hits.get(chunk_id, 0.0)
            m_score = vector_weight * v_score + keyword_weight * k_score
            merged_entries.append((chunk_id, v_score, k_score, m_score))

        merged_entries.sort(key=lambda entry: (-entry[3], entry[0]))
        merged_entries = merged_entries[:top_k]

        results: List[RetrievedChunk] = []
        for rank, (chunk_id, v_score, k_score, m_score) in enumerate(
            merged_entries, start=1
        ):
            try:
                doc = self._indexer.get_chunk_document(session_id, chunk_id)
                metadata = self._build_citation_metadata(
                    session_id=session_id,
                    chunk_id=chunk_id,
                    metadata=doc.metadata,
                )
            except Exception:
                logger.warning(
                    "Session %s chunk %s metadata could not be loaded; skipping.",
                    session_id,
                    chunk_id,
                )
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    page_content=doc.page_content,
                    metadata=metadata,
                    vector_score=v_score,
                    keyword_score=k_score,
                    merged_score=m_score,
                    rank=rank,
                )
            )

        return results

    @staticmethod
    def _build_citation_metadata(
        session_id: str,
        chunk_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build citation metadata and fill required fallback values."""
        citation_metadata = dict(metadata)
        if not citation_metadata.get("source"):
            citation_metadata["source"] = "unknown"
        citation_metadata["chunk_id"] = chunk_id
        citation_metadata["session_id"] = session_id
        return citation_metadata

    def _validate_query(self, query: str) -> None:
        """Validate that the query is not blank."""
        if not query or not query.strip():
            raise RetrieverException("query must not be blank.")

    def _validate_session(self, session_id: str) -> None:
        """Validate that the session exists."""
        try:
            self._indexer.get_session(session_id)
        except Exception as error:
            raise RetrieverException(f"Session not found: {session_id}") from error

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text for lightweight keyword search."""
        tokens: List[str] = []

        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)

        ascii_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens.extend(ascii_tokens)

        return tokens
