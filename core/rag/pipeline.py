"""串接 Chunking、Embedding 與 Session 索引流程。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from core.log import logger
from core.rag.chunker import (
    CHUNKING_PROFILES,
    ChunkingProfile,
    DocumentType,
    TextChunker,
)
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
    document_type: DocumentType | None = None,
) -> IngestionResult:
    """將文件寫入指定 Session 的 chunk 與向量索引。

    Args:
        session_indexer: 管理 Session 索引的 SessionIndexer 實例。
        session_id: 目標 Session 的唯一識別碼。
        documents: 待攝入的 ParsedDocument 序列。
        chunker: 可注入的自訂 TextChunker；None 時依 document_type 建立。
        embedder: 可注入的自訂嵌入器；None 時使用預設 SentenceTransformerEmbedder。
        document_type: 文件類型；None 時套用 SEMANTIC Profile（向下相容）。
    """
    profile: ChunkingProfile = CHUNKING_PROFILES[
        document_type if document_type is not None else DocumentType.SEMANTIC
    ]
    if chunker is None:
        resolved_chunker = TextChunker(profile=profile)
    else:
        # 若外部已注入自訂 chunker，則尊重該實例設定
        # 但仍會依照 document_type (或預設的 SEMANTIC) 更新 Session 權重
        resolved_chunker = chunker
        if document_type is not None:
            logger.warning(
                "Session %s: 同時提供了自訂 chunker 與 document_type (%s)。"
                "將使用自訂 chunker 進行分塊，但檢索權重會套用該類型 Profile 的設定。",
                session_id, document_type
            )

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

    # 只有當所有步驟都成功後，最後才更新 Session 權重 (AC-4 & Bug A Fix)
    session_indexer.update_session_weights(
        session_id=session_id,
        vector_weight=profile.vector_weight,
        keyword_weight=profile.keyword_weight,
    )

    logger.info("Session %s 完成 ingestion，共 %s 筆 Chunk。", session_id, len(chunk_ids))
    return IngestionResult(
        session_id=session_id,
        chunk_ids=chunk_ids,
        chunk_count=len(chunk_ids),
        embedding_dimension=int(embeddings.shape[1]),
    )
