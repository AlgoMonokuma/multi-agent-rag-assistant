"""Test behavior."""

from __future__ import annotations

from typing import List

import numpy as np
import pytest

from core.rag.chunker import ChunkingException, TextChunker
from core.rag.embeddings import EmbeddingException, SentenceTransformerEmbedder
from core.rag.parser import ParsedDocument


class FakeSplitter:
    """Test behavior."""

    def __init__(self, outputs: List[str]) -> None:
        self.outputs = outputs

    def split_text(self, text: str) -> List[str]:
        return list(self.outputs)


class FakeModel:
    """Test behavior."""

    def __init__(self, outputs: list[list[float]]) -> None:
        self.outputs = outputs

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.outputs


def test_chunk_documents_preserves_metadata_and_adds_chunk_fields() -> None:
    """Test behavior."""
    chunker = TextChunker(splitter=FakeSplitter(outputs=['test content', 'test content']))

    chunks = chunker.chunk_documents(
        documents=[
            ParsedDocument(
                page_content='test content',
                metadata={"source": "guide.md", "title": 'test content'},
            )
        ],
        session_id="session-1",
    )

    assert [chunk.page_content for chunk in chunks] == ['test content', 'test content']
    assert chunks[0].metadata == {
        "source": "guide.md",
        "title": 'test content',
        "chunk_index": 0,
        "document_index": 0,
        "parent_source": "guide.md",
        "session_id": "session-1",
    }
    assert chunks[1].metadata["chunk_index"] == 1


def test_chunker_raises_when_dependency_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""

    def fail_loader(self: TextChunker) -> object:
        raise ChunkingException('test content')

    monkeypatch.setattr(TextChunker, "_create_default_splitter", fail_loader)

    chunker = TextChunker()

    with pytest.raises(ChunkingException, match='test content'):
        chunker.chunk_documents(
            documents=[ParsedDocument(page_content="text", metadata={})],
            session_id="session-1",
        )


def test_embed_documents_returns_float32_matrix_with_expected_dimension() -> None:
    """Test behavior."""
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
    """Test behavior."""
    embedder = SentenceTransformerEmbedder(model=FakeModel(outputs=[[0.1] * 8]))

    with pytest.raises(EmbeddingException, match="Embedding dimension mismatch"):
        embedder.embed_documents([ParsedDocument(page_content="A", metadata={})])
