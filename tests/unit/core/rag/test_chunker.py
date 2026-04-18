"""文字分塊與嵌入測試。"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from core.rag.chunker import ChunkingException, TextChunker
from core.rag.embeddings import EmbeddingException, SentenceTransformerEmbedder
from core.rag.parser import ParsedDocument


class FakeSplitter:
    """模擬文字分塊器。"""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = outputs

    def split_text(self, text: str) -> List[str]:
        return list(self.outputs)


class FakeModel:
    """模擬嵌入模型。"""

    def __init__(self, outputs: list[list[float]]) -> None:
        self.outputs = outputs

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.outputs


def test_chunk_documents_preserves_metadata_and_adds_chunk_fields() -> None:
    """分塊後應保留原始 metadata 並補上追蹤欄位。"""
    chunker = TextChunker(splitter=FakeSplitter(outputs=["第一段", "第二段"]))

    chunks = chunker.chunk_documents(
        documents=[
            ParsedDocument(
                page_content="原始內容",
                metadata={"source": "guide.md", "title": "指南"},
            )
        ],
        session_id="session-1",
    )

    assert [chunk.page_content for chunk in chunks] == ["第一段", "第二段"]
    assert chunks[0].metadata == {
        "source": "guide.md",
        "title": "指南",
        "chunk_index": 0,
        "document_index": 0,
        "parent_source": "guide.md",
        "session_id": "session-1",
    }
    assert chunks[1].metadata["chunk_index"] == 1


def test_chunker_raises_when_dependency_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少文字分塊依賴時應丟出明確錯誤。"""

    def fail_loader(self: TextChunker) -> object:
        raise ChunkingException("缺少套件")

    monkeypatch.setattr(TextChunker, "_create_default_splitter", fail_loader)

    chunker = TextChunker()

    with pytest.raises(ChunkingException, match="缺少套件"):
        chunker.chunk_documents(
            documents=[ParsedDocument(page_content="text", metadata={})],
            session_id="session-1",
        )


def test_embed_documents_returns_float32_matrix_with_expected_dimension() -> None:
    """嵌入器應輸出 float32 且符合預期維度。"""
    model = FakeModel(outputs=[[0.1] * 384, [0.2] * 384])
    embedder = SentenceTransformerEmbedder(model=model)

    matrix = embedder.embed_documents(
        [
            ParsedDocument(page_content="A", metadata={}),
            ParsedDocument(page_content="B", metadata={}),
        ]
    )

    assert matrix.dtype == np.float32
    assert matrix.shape == (2, 384)


def test_embedder_raises_when_dimension_does_not_match() -> None:
    """嵌入維度錯誤時應拒絕處理。"""
    embedder = SentenceTransformerEmbedder(model=FakeModel(outputs=[[0.1] * 8]))

    with pytest.raises(EmbeddingException, match="嵌入維度錯誤"):
        embedder.embed_documents([ParsedDocument(page_content="A", metadata={})])
