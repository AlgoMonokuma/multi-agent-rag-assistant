"""提供文字分塊功能。"""

from __future__ import annotations

from typing import List, Protocol, Sequence

from core.log import logger
from core.rag.parser import ParsedDocument


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingException(Exception):
    """文字分塊相關錯誤。"""


class TextSplitter(Protocol):
    """定義可供注入的文字分塊器介面。"""

    def split_text(self, text: str) -> List[str]:
        """將文字分割為多個片段。"""


class TextChunker:
    """將 ParsedDocument 轉為可嵌入的 Chunk 文件。"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        splitter: TextSplitter | None = None,
    ) -> None:
        """初始化文字分塊器。"""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._splitter = splitter

    def chunk_documents(
        self,
        documents: Sequence[ParsedDocument],
        session_id: str,
    ) -> List[ParsedDocument]:
        """將多份 ParsedDocument 轉成帶 metadata 的 chunk 文件。"""
        splitter = self._splitter or self._create_default_splitter()
        chunked_documents: List[ParsedDocument] = []

        for document_index, document in enumerate(documents):
            text_chunks = splitter.split_text(document.page_content)
            base_metadata = dict(document.metadata)
            parent_source = str(base_metadata.get("source", "unknown"))

            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk_metadata = dict(base_metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["document_index"] = document_index
                chunk_metadata["parent_source"] = parent_source
                chunk_metadata["session_id"] = session_id

                chunked_documents.append(
                    ParsedDocument(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

        logger.info("Session %s 完成 %s 筆 Chunk 分塊。", session_id, len(chunked_documents))
        return chunked_documents

    def _create_default_splitter(self) -> TextSplitter:
        """建立預設的 RecursiveCharacterTextSplitter。"""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as error:
            raise ChunkingException(
                "尚未安裝 langchain-text-splitters，無法執行文字分塊。"
            ) from error

        return RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
