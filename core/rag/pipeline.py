"""串接 Chunking、Embedding 與 Session 索引流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.log import logger
from core.rag.chunker import TextChunker
from core.rag.embeddings import SentenceTransformerEmbedder
from core.rag.indexer import SessionIndexer
from core.rag.parser import ParsedDocument


@dataclass(slots=True)
class IngestionResult:
    """保存單次 Session ingestion 的輸出摘要。"""

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
) -> IngestionResult:
    """將文件寫入指定 Session 的 chunk 與向量索引。"""
    resolved_chunker = chunker or TextChunker()
    resolved_embedder = embedder or SentenceTransformerEmbedder()

    chunked_documents = resolved_chunker.chunk_documents(
        documents=documents,
        session_id=session_id,
    )
    if not chunked_documents:
        logger.info("Session %s ingestion 未產生任何 Chunk。", session_id)
        return IngestionResult(
            session_id=session_id,
            chunk_ids=[],
            chunk_count=0,
            embedding_dimension=0,
        )

    embeddings = resolved_embedder.embed_documents(chunked_documents)
    chunk_ids = session_indexer.ingest_chunk_embeddings(
        session_id=session_id,
        documents=chunked_documents,
        embeddings=embeddings,
    )

    logger.info("Session %s 完成 ingestion，共 %s 筆 Chunk。", session_id, len(chunk_ids))
    return IngestionResult(
        session_id=session_id,
        chunk_ids=chunk_ids,
        chunk_count=len(chunk_ids),
        embedding_dimension=int(embeddings.shape[1]),
    )
