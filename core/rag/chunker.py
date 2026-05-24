"""提供文字分塊功能。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Sequence

from core.log import logger
from core.rag.parser import ParsedDocument


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingException(Exception):
    """文字分塊相關錯誤。"""


# ---------------------------------------------------------------------------
# 文件類型感知 Chunking Profile
# ---------------------------------------------------------------------------


class DocumentType(str, Enum):
    """文件類型枚舉，用於選擇最佳分塊與檢索參數。

    使用 str 作為基底類別，方便 FastAPI 從 Query Parameter 直接解析。
    """

    SEMANTIC = "semantic"  # 語意理解型：長文、報告、書籍
    PRECISE = "precise"   # 精確查找型：FAQ、規格書、技術文件
    CODE = "code"         # 程式碼型：原始碼、Notebook、設定檔


@dataclass(frozen=True)
class ChunkingProfile:
    """描述分塊與混合檢索行為的不可變設定物件。

    使用 frozen=True 防止意外修改 CHUNKING_PROFILES 字典中的值。
    """

    chunk_size: int
    chunk_overlap: int
    vector_weight: float
    keyword_weight: float


# 三種 DocumentType 的預設 ChunkingProfile 常數字典
CHUNKING_PROFILES: dict[DocumentType, ChunkingProfile] = {
    DocumentType.SEMANTIC: ChunkingProfile(
        chunk_size=1000,
        chunk_overlap=200,
        vector_weight=0.7,
        keyword_weight=0.3,
    ),
    DocumentType.PRECISE: ChunkingProfile(
        chunk_size=400,
        chunk_overlap=100,
        vector_weight=0.4,
        keyword_weight=0.6,
    ),
    DocumentType.CODE: ChunkingProfile(
        chunk_size=600,
        chunk_overlap=50,
        vector_weight=0.6,
        keyword_weight=0.4,
    ),
}


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
        profile: ChunkingProfile | None = None,
    ) -> None:
        """初始化文字分塊器。

        Args:
            chunk_size: 每個 Chunk 的字元數上限（profile 未提供時使用）。
            chunk_overlap: 相鄰 Chunk 的重疊字元數（profile 未提供時使用）。
            splitter: 可注入的自訂文字分塊器；None 時使用 RecursiveCharacterTextSplitter。
            profile: ChunkingProfile 設定物件；若提供，優先以 profile 的
                     chunk_size / chunk_overlap 覆蓋個別參數。
        """
        if profile is not None:
            self._chunk_size = profile.chunk_size
            self._chunk_overlap = profile.chunk_overlap
        else:
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
