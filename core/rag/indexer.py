"""提供 Session 隔離的 RAG 索引管理能力。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

import numpy as np

from core.log import logger
from core.rag.parser import ParsedDocument


class IndexerException(Exception):
    """Session 索引相關錯誤。"""


@dataclass(slots=True)
class SessionIndexRecord:
    """保存單一 Session 的索引與中繼資料狀態。"""

    session_id: str
    index: Any
    created_at: datetime
    chunk_map: Dict[str, ParsedDocument] = field(default_factory=dict)
    metadata_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    vector_chunk_ids: List[str] = field(default_factory=list)


class SessionIndexer:
    """管理 Session 專屬的 FAISS 索引與 Chunk 對應資訊。"""

    def __init__(
        self,
        index_factory: Callable[[], Any] | None = None,
    ) -> None:
        """初始化 Session 索引器。"""
        self._index_factory = index_factory or self._create_default_faiss_index
        self._sessions: Dict[str, SessionIndexRecord] = {}

    def create_session(self) -> SessionIndexRecord:
        """建立新的 Session 索引記錄。"""
        session_id = str(uuid4())

        try:
            index = self._index_factory()
        except Exception as error:
            logger.error("建立 Session 索引失敗: %s", error)
            raise IndexerException("無法建立 Session 索引。") from error

        record = SessionIndexRecord(
            session_id=session_id,
            index=index,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = record
        logger.info("已建立 Session 索引: %s", session_id)
        return record

    def get_session(self, session_id: str) -> SessionIndexRecord:
        """取得指定 Session 記錄。"""
        return self._require_session(session_id)

    def store_documents(
        self,
        session_id: str,
        documents: Sequence[ParsedDocument],
    ) -> List[str]:
        """只儲存 Chunk 與 metadata，不寫入向量索引。"""
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
            "Session %s 已暫存 %s 筆 Chunk metadata。",
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
        """以原子方式寫入 Chunk metadata 與對應向量。"""
        record = self._require_session(session_id)
        embedding_matrix = self._normalize_embeddings(embeddings)

        if len(documents) != int(embedding_matrix.shape[0]):
            raise IndexerException("文件數量與向量數量不一致。")

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
            logger.error("Session %s 寫入 FAISS 索引失敗: %s", session_id, error)
            raise IndexerException("寫入 FAISS 索引失敗。") from error

        for chunk_id, stored_document in prepared_documents:
            self._store_chunk(record, chunk_id, stored_document)
            record.vector_chunk_ids.append(chunk_id)

        logger.info(
            "Session %s 已寫入 %s 筆 Chunk 與向量。",
            session_id,
            len(prepared_documents),
        )
        return [chunk_id for chunk_id, _ in prepared_documents]

    def get_chunk_metadata(self, session_id: str, chunk_id: str) -> Dict[str, Any]:
        """取得指定 Session 內單一 Chunk 的 metadata。"""
        record = self._require_session(session_id)

        if chunk_id not in record.metadata_map:
            logger.error("Session %s 查無 chunk_id: %s", session_id, chunk_id)
            raise IndexerException(
                f"Session {session_id} 查無對應的 chunk_id: {chunk_id}"
            )

        return dict(record.metadata_map[chunk_id])

    def list_chunk_metadata(self, session_id: str) -> List[Dict[str, Any]]:
        """列出指定 Session 的所有 Chunk metadata。"""
        record = self._require_session(session_id)
        return [dict(metadata) for metadata in record.metadata_map.values()]

    def get_chunk_id_by_ordinal(self, session_id: str, ordinal: int) -> str:
        """依照 FAISS 向量順序取得對應的 chunk_id。"""
        record = self._require_session(session_id)

        if ordinal < 0:
            logger.error("Session %s 收到無效 ordinal: %s", session_id, ordinal)
            raise IndexerException(
                f"Session {session_id} 查無對應的 ordinal: {ordinal}"
            )

        try:
            return record.vector_chunk_ids[ordinal]
        except IndexError as error:
            logger.error("Session %s 查無 ordinal: %s", session_id, ordinal)
            raise IndexerException(
                f"Session {session_id} 查無對應的 ordinal: {ordinal}"
            ) from error

    def list_vector_chunk_ids(self, session_id: str) -> List[str]:
        """列出指定 Session 目前的向量順序映射。"""
        record = self._require_session(session_id)
        return list(record.vector_chunk_ids)

    def cleanup_session(self, session_id: str) -> None:
        """清除 Session 索引與 metadata。"""
        self._require_session(session_id)
        del self._sessions[session_id]
        logger.info("已清除 Session 索引: %s", session_id)

    def _require_session(self, session_id: str) -> SessionIndexRecord:
        """驗證 Session 是否存在。"""
        record = self._sessions.get(session_id)
        if record is None:
            logger.error("查無 Session 索引: %s", session_id)
            raise IndexerException(f"查無 Session 索引: {session_id}")
        return record

    def _build_next_chunk_id(
        self,
        record: SessionIndexRecord,
        extra_offset: int = 0,
    ) -> str:
        """依照目前記錄產生下一個 chunk_id。"""
        return f"chunk-{len(record.chunk_map) + extra_offset + 1}"

    def _build_stored_document(
        self,
        session_id: str,
        chunk_id: str,
        document: ParsedDocument,
    ) -> ParsedDocument:
        """建立由索引器控管的 Chunk 文件。"""
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
        """將 Chunk 與 metadata 寫入 Session 記錄。"""
        record.chunk_map[chunk_id] = document
        record.metadata_map[chunk_id] = dict(document.metadata)

    @staticmethod
    def _normalize_embeddings(
        embeddings: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """將嵌入向量轉為 FAISS 可接受的矩陣格式。"""
        try:
            matrix = np.asarray(embeddings, dtype=np.float32)
        except Exception as error:
            raise IndexerException("嵌入向量無法轉為 float32 矩陣。") from error

        if matrix.ndim != 2:
            raise IndexerException("嵌入向量必須是二維矩陣。")

        return matrix

    @staticmethod
    def _create_default_faiss_index() -> Any:
        """建立預設的記憶體內 FAISS 索引。"""
        try:
            import faiss
        except ImportError as error:
            raise IndexerException(
                "尚未安裝 faiss，請先安裝 faiss-cpu 依賴。"
            ) from error

        return faiss.IndexFlatL2(384)
