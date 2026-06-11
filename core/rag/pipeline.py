"""Chunking, embedding, and session indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.log import logger
from core.rag.chunker import (
    CHUNKING_PROFILES,
    ChunkingException,
    ChunkingProfile,
    DocumentType,
    TextChunker,
)
from core.rag.embeddings import EmbeddingException, SentenceTransformerEmbedder
from core.rag.indexer import IndexerException, SessionIndexer
from core.rag.parser import ParsedDocument


@dataclass(slots=True)
class IngestionResult:
    """Summary for one session ingestion run."""

    session_id: str
    chunk_ids: list[str]
    chunk_count: int
    embedding_dimension: int


def ingest_documents(
    session_indexer: SessionIndexer,
    session_id: str,
    documents: Sequence[ParsedDocument],
    chunker: TextChunker | None = None,
    embedder: SentenceTransformerEmbedder | None = None,
    document_type: DocumentType | None = None,
) -> IngestionResult:
    """Write documents into a session-scoped chunk and vector index."""
    profile: ChunkingProfile = CHUNKING_PROFILES[
        document_type if document_type is not None else DocumentType.SEMANTIC
    ]
    if chunker is None:
        resolved_chunker = TextChunker(profile=profile)
    else:
        resolved_chunker = chunker
        if document_type is not None:
            logger.warning(
                "Session %s received a custom chunker and document_type %s. "
                "Chunking will use the custom chunker, while retrieval weights "
                "will use the selected document type profile.",
                session_id,
                document_type,
            )

    resolved_embedder = embedder or SentenceTransformerEmbedder()

    try:
        chunked_documents = resolved_chunker.chunk_documents(
            documents=documents,
            session_id=session_id,
        )
    except Exception as error:
        logger.error(
            "Session %s chunking failed during ingestion: %s",
            session_id,
            error,
        )
        raise ChunkingException(
            f"Session {session_id} chunking failed during ingestion: {error}"
        ) from error

    if not chunked_documents:
        logger.info("Session %s ingestion produced no chunks.", session_id)
        return IngestionResult(
            session_id=session_id,
            chunk_ids=[],
            chunk_count=0,
            embedding_dimension=0,
        )

    try:
        embeddings = resolved_embedder.embed_documents(chunked_documents)
    except Exception as error:
        logger.error(
            "Session %s embedding failed during ingestion: %s",
            session_id,
            error,
        )
        raise EmbeddingException(
            f"Session {session_id} embedding failed during ingestion: {error}"
        ) from error

    try:
        chunk_ids = session_indexer.ingest_chunk_embeddings(
            session_id=session_id,
            documents=chunked_documents,
            embeddings=embeddings,
        )
    except IndexerException as error:
        logger.error("Session %s indexing failed during ingestion: %s", session_id, error)
        raise IndexerException(
            f"Session {session_id} indexing failed during ingestion: {error}"
        ) from error

    session_indexer.update_session_weights(
        session_id=session_id,
        vector_weight=profile.vector_weight,
        keyword_weight=profile.keyword_weight,
    )

    logger.info("Session %s completed ingestion with %s chunks.", session_id, len(chunk_ids))
    return IngestionResult(
        session_id=session_id,
        chunk_ids=chunk_ids,
        chunk_count=len(chunk_ids),
        embedding_dimension=int(embeddings.shape[1]),
    )
