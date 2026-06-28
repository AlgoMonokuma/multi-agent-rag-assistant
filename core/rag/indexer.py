"""Session-isolated RAG index management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

import numpy as np

from core.log import logger
from core.rag.parser import ParsedDocument


class IndexerException(Exception):
    """Session index error."""


@dataclass(slots=True)
class SessionIndexRecord:
    """Index and metadata state for one session."""

    session_id: str
    index: Any
    created_at: datetime
    chunk_map: Dict[str, ParsedDocument] = field(default_factory=dict)
    metadata_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vector_chunk_ids: List[str] = field(default_factory=list)
    vector_weight: float | None = None
    keyword_weight: float | None = None


class SessionIndexer:
    """Manage per-session FAISS indexes and chunk mappings."""

    def __init__(
        self,
        index_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Initialize the session indexer."""
        self._index_factory = index_factory or self._create_default_faiss_index
        self._sessions: Dict[str, SessionIndexRecord] = {}

    def create_session(self) -> SessionIndexRecord:
        """Create a new session index record."""
        session_id = str(uuid4())

        try:
            index = self._index_factory()
        except Exception as error:
            logger.error("Session index creation failed: %s", error)
            raise IndexerException("Unable to create session index.") from error

        record = SessionIndexRecord(
            session_id=session_id,
            index=index,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = record
        logger.info("Created session index: %s", session_id)
        return record

    def get_session(self, session_id: str) -> SessionIndexRecord:
        """Return the record for a session."""
        return self._require_session(session_id)

    def update_session_weights(
        self, session_id: str, vector_weight: float, keyword_weight: float
    ) -> None:
        """Update retrieval weights for a session."""
        if not (0.0 <= vector_weight <= 1.0) or not (0.0 <= keyword_weight <= 1.0):
            raise IndexerException(
                f"Invalid weight range: vector={vector_weight}, keyword={keyword_weight}. "
                "Weights must be between 0.0 and 1.0."
            )

        record = self._require_session(session_id)
        record.vector_weight = vector_weight
        record.keyword_weight = keyword_weight

    def store_documents(
        self,
        session_id: str,
        documents: Sequence[ParsedDocument],
    ) -> List[str]:
        """Store chunk documents and metadata without writing vectors."""
        record = self._require_session(session_id)
        stored_chunk_ids: List[str] = []

        for document in documents:
            chunk_id = self._build_next_chunk_id(record)
            stored_document = self._build_stored_document(
                session_id=session_id,
                chunk_id=chunk_id,
                document=document,
            )
            self._store_chunk(record, chunk_id, stored_document)
            stored_chunk_ids.append(chunk_id)

        logger.info(
            "Session %s stored %s chunk metadata records.",
            session_id,
            len(stored_chunk_ids),
        )
        return stored_chunk_ids

    def ingest_chunk_embeddings(
        self,
        session_id: str,
        documents: Sequence[ParsedDocument],
        embeddings: Sequence[Sequence[float]],
    ) -> List[str]:
        """Atomically store chunk metadata and matching vectors."""
        record = self._require_session(session_id)
        embedding_matrix = self._normalize_embeddings(embeddings)

        if len(documents) != int(embedding_matrix.shape[0]):
            raise IndexerException("Document count must match embedding count.")

        prepared_documents: List[tuple[str, ParsedDocument]] = []
        for document in documents:
            chunk_id = self._build_next_chunk_id(record, extra_offset=len(prepared_documents))
            prepared_documents.append(
                (
                    chunk_id,
                    self._build_stored_document(
                        session_id=session_id,
                        chunk_id=chunk_id,
                        document=document,
                    ),
                )
            )

        try:
            record.index.add(embedding_matrix)
        except Exception as error:
            logger.error("Session %s FAISS index write failed: %s", session_id, error)
            raise IndexerException("FAISS index write failed.") from error

        for chunk_id, stored_document in prepared_documents:
            self._store_chunk(record, chunk_id, stored_document)
            record.vector_chunk_ids.append(chunk_id)

        logger.info(
            "Session %s stored %s chunks and vectors.",
            session_id,
            len(prepared_documents),
        )
        return [chunk_id for chunk_id, _ in prepared_documents]

    def get_chunk_metadata(self, session_id: str, chunk_id: str) -> Dict[str, Any]:
        """Return metadata for one chunk in a session."""
        record = self._require_session(session_id)

        if chunk_id not in record.metadata_map:
            logger.error("Session %s missing chunk_id: %s", session_id, chunk_id)
            raise IndexerException(f"Session {session_id} has no chunk_id: {chunk_id}")

        return dict(record.metadata_map[chunk_id])

    def list_chunk_metadata(self, session_id: str) -> List[Dict[str, Any]]:
        """Return metadata for all chunks in a session."""
        record = self._require_session(session_id)
        return [dict(metadata) for metadata in record.metadata_map.values()]

    def get_chunk_id_by_ordinal(self, session_id: str, ordinal: int) -> str:
        """Return the chunk_id for a FAISS vector ordinal."""
        record = self._require_session(session_id)

        if ordinal < 0:
            logger.error("Session %s received invalid ordinal: %s", session_id, ordinal)
            raise IndexerException(f"Session {session_id} has no ordinal: {ordinal}")

        try:
            return record.vector_chunk_ids[ordinal]
        except IndexError as error:
            logger.error("Session %s missing ordinal: %s", session_id, ordinal)
            raise IndexerException(f"Session {session_id} has no ordinal: {ordinal}") from error

    def list_vector_chunk_ids(self, session_id: str) -> List[str]:
        """Return the current vector ordinal mapping for a session."""
        record = self._require_session(session_id)
        return list(record.vector_chunk_ids)

    def get_chunk_document(self, session_id: str, chunk_id: str) -> ParsedDocument:
        """Return the full ParsedDocument for one chunk in a session."""
        record = self._require_session(session_id)

        if chunk_id not in record.chunk_map:
            logger.error("Session %s missing chunk_id: %s", session_id, chunk_id)
            raise IndexerException(f"Session {session_id} has no chunk_id: {chunk_id}")

        return record.chunk_map[chunk_id]

    def list_chunk_documents(self, session_id: str) -> List[ParsedDocument]:
        """Return all chunk ParsedDocument objects for a session."""
        record = self._require_session(session_id)
        return list(record.chunk_map.values())

    def cleanup_session(self, session_id: str) -> None:
        """Remove session index and metadata state."""
        self._require_session(session_id)
        del self._sessions[session_id]
        logger.info("Cleaned up session index: %s", session_id)

    def _require_session(self, session_id: str) -> SessionIndexRecord:
        """Validate that a session exists and return its record."""
        record = self._sessions.get(session_id)
        if record is None:
            logger.error("Session index not found: %s", session_id)
            raise IndexerException(f"Session index not found: {session_id}")
        return record

    def _build_next_chunk_id(
        self,
        record: SessionIndexRecord,
        extra_offset: int = 0,
    ) -> str:
        """Build the next chunk_id for the current record state."""
        return f"chunk-{len(record.chunk_map) + extra_offset + 1}"

    def _build_stored_document(
        self,
        session_id: str,
        chunk_id: str,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """Create a chunk document controlled by the indexer."""
        metadata = dict(document.metadata)
        metadata["chunk_id"] = chunk_id
        metadata["session_id"] = session_id

        return ParsedDocument(
            page_content=document.page_content,
            metadata=metadata,
        )

    def _store_chunk(
        self,
        record: SessionIndexRecord,
        chunk_id: str,
        document: ParsedDocument,
    ) -> None:
        """Store chunk content and metadata in a session record."""
        record.chunk_map[chunk_id] = document
        record.metadata_map[chunk_id] = dict(document.metadata)

    @staticmethod
    def _normalize_embeddings(
        embeddings: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Convert embeddings into the matrix format accepted by FAISS."""
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except Exception as error:
            raise IndexerException("Embeddings cannot be converted to float32 matrix.") from error

        if matrix.ndim != 2:
            raise IndexerException("Embeddings must be a two-dimensional matrix.")

        return matrix

    @staticmethod
    def _create_default_faiss_index() -> Any:
        """Create the default in-memory FAISS index."""
        try:
            import faiss
        except ImportError as error:
            raise IndexerException("faiss is not installed; install faiss-cpu first.") from error

        return faiss.IndexFlatL2(384)


_default_session_indexer: SessionIndexer | None = None


def get_default_session_indexer() -> SessionIndexer:
    """Return the shared session indexer used by the default runtime path.

    The agent workflow and any future ingestion entry points should use this
    singleton so they operate on the same in-memory session registry.
    """
    global _default_session_indexer
    if _default_session_indexer is None:
        _default_session_indexer = SessionIndexer()
    return _default_session_indexer
