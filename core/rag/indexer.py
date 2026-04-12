"""管理 Session 隔離 RAG 索引的核心模組。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Sequence
from uuid import uuid4

from core.log import logger
from core.rag.parser import ParsedDocument


class IndexerException(Exception):
    """Session 索引器相關錯誤。"""


@dataclass(slots=True)
class SessionIndexRecord:
    """描述單一 Session 的索引與 metadata 狀態。"""

    session_id: str
    index: Any
    created_at: datetime
    chunk_map: Dict[str, ParsedDocument] = field(default_factory=dict)
    metadata_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class SessionIndexer:
    """提供 Session 級別的索引建立、metadata 存取與清理。"""

    def __init__(
        self,
        index_factory: Callable[[], Any] | None = None,
    ) -> None:
        """初始化 Session 索引器。"""
        self._index_factory = index_factory or self._create_default_faiss_index
        self._sessions: Dict[str, SessionIndexRecord] = {}

    def create_session(self) -> SessionIndexRecord:
        """建立新的 Session 與對應索引。"""
        session_id = str(uuid4())

        try:
            index = self._index_factory()
        except Exception as error:
            logger.error("建立 Session 索引失敗: %s", error)
            raise IndexerException("無法建立 Session 專屬索引。") from error

        record = SessionIndexRecord(
            session_id=session_id,
            index=index,
            created_at=datetime.now(timezone.utc),
        )
        self._sessions[session_id] = record
        logger.info("已建立 Session 索引: %s", session_id)
        return record

    def get_session(self, session_id: str) -> SessionIndexRecord:
        """取得既有 Session 記錄。"""
        return self._require_session(session_id)

    def store_documents(
        self,
        session_id: str,
        documents: Sequence[ParsedDocument],
    ) -> List[str]:
        """儲存文件與 metadata 映射，並自動附加 Session 資訊。"""
        record = self._require_session(session_id)
        stored_chunk_ids: List[str] = []

        for document in documents:
            chunk_id = f"chunk-{len(record.chunk_map) + 1}"
            metadata = dict(document.metadata)
            metadata["chunk_id"] = chunk_id
            metadata["session_id"] = session_id

            stored_document = ParsedDocument(
                page_content=document.page_content,
                metadata=metadata,
            )
            record.chunk_map[chunk_id] = stored_document
            record.metadata_map[chunk_id] = dict(metadata)
            stored_chunk_ids.append(chunk_id)

        logger.info(
            "Session %s 已儲存 %s 筆文件 metadata。",
            session_id,
            len(stored_chunk_ids),
        )
        return stored_chunk_ids

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

    def cleanup_session(self, session_id: str) -> None:
        """移除 Session 索引與相關 metadata。"""
        self._require_session(session_id)
        del self._sessions[session_id]
        logger.info("已清理 Session 索引: %s", session_id)

    def _require_session(self, session_id: str) -> SessionIndexRecord:
        """驗證 Session 是否存在。"""
        record = self._sessions.get(session_id)
        if record is None:
            logger.error("查無 Session 索引: %s", session_id)
            raise IndexerException(f"查無 Session 索引: {session_id}")
        return record

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
