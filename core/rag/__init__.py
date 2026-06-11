"""RAG core package exports."""

from core.rag.chunker import (
    CHUNKING_PROFILES,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ChunkingException,
    ChunkingProfile,
    DocumentType,
    TextChunker,
)
from core.rag.embeddings import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingException,
    SentenceTransformerEmbedder,
)
from core.rag.indexer import IndexerException, SessionIndexRecord, SessionIndexer
from core.rag.pipeline import IngestionResult, ingest_documents
from core.rag.reranker import CrossEncoderReranker, RerankerException
from core.rag.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrievedChunk,
    RetrieverException,
)

__all__ = [
    "CHUNKING_PROFILES",
    "ChunkingException",
    "ChunkingProfile",
    "CrossEncoderReranker",
    "DEFAULT_CHUNK_OVERLAP",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_EMBEDDING_DIMENSION",
    "DEFAULT_EMBEDDING_MODEL",
    "DocumentType",
    "EmbeddingException",
    "HybridRetriever",
    "HybridSearchResult",
    "IndexerException",
    "IngestionResult",
    "RerankerException",
    "RetrievedChunk",
    "RetrieverException",
    "SentenceTransformerEmbedder",
    "SessionIndexRecord",
    "SessionIndexer",
    "TextChunker",
    "ingest_documents",
]
