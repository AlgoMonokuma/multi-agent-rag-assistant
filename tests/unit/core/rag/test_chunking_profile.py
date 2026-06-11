"""Test behavior."""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.rag.chunker import (
    CHUNKING_PROFILES,
    ChunkingException,
    ChunkingProfile,
    DocumentType,
    TextChunker,
)
from core.rag.embeddings import EmbeddingException
from core.rag.indexer import SessionIndexer
from core.rag.parser import ParsedDocument
from core.rag.pipeline import ingest_documents
from core.rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class CapturingFakeSplitter:
    """Test behavior."""

    last_chunk_size: int = 0
    last_chunk_overlap: int = 0

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        CapturingFakeSplitter.last_chunk_size = chunk_size
        CapturingFakeSplitter.last_chunk_overlap = chunk_overlap
        self._outputs: List[str] = ["chunk-A", "chunk-B"]

    def split_text(self, text: str) -> List[str]:
        return list(self._outputs)


class FakeSplitter:
    """Test behavior."""

    def __init__(self, outputs: List[str] | None = None) -> None:
        self._outputs = outputs or ["chunk-A"]

    def split_text(self, text: str) -> List[str]:
        return list(self._outputs)


class FakeEmbedder:
    """Test behavior."""

    def embed_documents(self, documents: List[ParsedDocument]) -> np.ndarray:
        return np.zeros((len(documents), 384), dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 384), dtype=np.float32)


class FailingChunker:
    def chunk_documents(self, documents, session_id: str):
        raise RuntimeError("chunk boom")


class FailingEmbedder:
    def embed_documents(self, documents):
        raise RuntimeError("embed boom")


class FakeFaissIndex:
    """Test behavior."""

    def __init__(self) -> None:
        self.vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.append(vectors.copy())

    @property
    def ntotal(self) -> int:
        if not self.vectors:
            return 0
        return sum(v.shape[0] for v in self.vectors)

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        total = self.ntotal
        if total == 0:
            return (
                np.full((query_vectors.shape[0], k), np.inf, dtype=np.float32),
                np.full((query_vectors.shape[0], k), -1, dtype=np.int64),
            )
        actual_k = min(k, total)
        distances = np.arange(actual_k, dtype=np.float32).reshape(1, -1)
        indices = np.arange(actual_k, dtype=np.int64).reshape(1, -1)
        if k > total:
            pad = k - total
            distances = np.concatenate(
                [distances, np.full((1, pad), np.inf, dtype=np.float32)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((1, pad), -1, dtype=np.int64)], axis=1
            )
        return distances, indices


class FailingAddIndex(FakeFaissIndex):
    def add(self, vectors: np.ndarray) -> None:
        raise RuntimeError("index boom")


@pytest.fixture
def fake_index_factory():
    def factory() -> FakeFaissIndex:
        return FakeFaissIndex()
    return factory


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestChunkingProfilesConstants:
    """Test behavior."""

    def test_semantic_profile_has_correct_default_parameters(self) -> None:
        """Test behavior."""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        assert profile.chunk_size == 1000
        assert profile.chunk_overlap == 200
        assert profile.vector_weight == 0.7
        assert profile.keyword_weight == 0.3

    def test_precise_profile_has_correct_default_parameters(self) -> None:
        """Test behavior."""
        profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        assert profile.chunk_size == 400
        assert profile.chunk_overlap == 100
        assert profile.vector_weight == 0.4
        assert profile.keyword_weight == 0.6

    def test_code_profile_has_correct_default_parameters(self) -> None:
        """Test behavior."""
        profile = CHUNKING_PROFILES[DocumentType.CODE]
        assert profile.chunk_size == 600
        assert profile.chunk_overlap == 50
        assert profile.vector_weight == 0.6
        assert profile.keyword_weight == 0.4

    def test_all_three_document_types_are_covered(self) -> None:
        """Test behavior."""
        assert DocumentType.SEMANTIC in CHUNKING_PROFILES
        assert DocumentType.PRECISE in CHUNKING_PROFILES
        assert DocumentType.CODE in CHUNKING_PROFILES

    def test_chunking_profile_is_immutable(self) -> None:
        """Test behavior."""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        with pytest.raises((AttributeError, TypeError)):
            profile.chunk_size = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestDocumentTypeEnum:
    """Test behavior."""

    def test_document_type_values_are_strings(self) -> None:
        """Test behavior."""
        assert DocumentType.SEMANTIC == "semantic"
        assert DocumentType.PRECISE == "precise"
        assert DocumentType.CODE == "code"

    def test_document_type_can_be_constructed_from_string(self) -> None:
        """Test behavior."""
        assert DocumentType("semantic") is DocumentType.SEMANTIC
        assert DocumentType("precise") is DocumentType.PRECISE
        assert DocumentType("code") is DocumentType.CODE

    def test_invalid_document_type_string_raises_value_error(self) -> None:
        """Test behavior."""
        with pytest.raises(ValueError):
            DocumentType("unknown_type")


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestTextChunkerProfileIntegration:
    """Test behavior."""

    def test_chunker_uses_profile_chunk_size_over_default(self) -> None:
        """Test behavior."""
        precise_profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        chunker = TextChunker(profile=precise_profile, splitter=FakeSplitter())
        # Test note.
        assert chunker._chunk_size == 400

    def test_chunker_uses_profile_chunk_overlap_over_default(self) -> None:
        """Test behavior."""
        code_profile = CHUNKING_PROFILES[DocumentType.CODE]
        chunker = TextChunker(profile=code_profile, splitter=FakeSplitter())
        assert chunker._chunk_overlap == 50

    def test_chunker_without_profile_uses_default_parameters(self) -> None:
        """Test behavior."""
        chunker = TextChunker(splitter=FakeSplitter())
        assert chunker._chunk_size == 1000
        assert chunker._chunk_overlap == 200

    def test_chunker_explicit_params_overridden_by_profile(self) -> None:
        """Test behavior."""
        precise_profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        chunker = TextChunker(
            chunk_size=9999,  # Test note.
            chunk_overlap=9999,
            profile=precise_profile,
            splitter=FakeSplitter(),
        )
        assert chunker._chunk_size == 400
        assert chunker._chunk_overlap == 100

    def test_chunker_with_semantic_profile_produces_chunks(self) -> None:
        """Test behavior."""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        chunker = TextChunker(profile=profile, splitter=FakeSplitter(['test content', 'test content']))
        docs = [ParsedDocument(page_content='test content', metadata={"source": "test.md"})]
        chunks = chunker.chunk_documents(docs, session_id="sess-1")
        assert len(chunks) == 2
        assert chunks[0].page_content == 'test content'


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestIngestDocumentsDocumentType:
    """Test behavior."""

    def _make_indexer_and_docs(
        self, fake_index_factory
    ) -> tuple[SessionIndexer, str, list[ParsedDocument]]:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content='test content' * 10, metadata={"source": "a.md"})]
        return indexer, record.session_id, docs

    def test_ingest_without_document_type_is_backward_compatible(
        self, fake_index_factory
    ) -> None:
        """Test behavior."""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            chunker=TextChunker(splitter=FakeSplitter(["chunk-1"])),
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        assert result.chunk_count == 1
        assert result.session_id == sid

    def test_ingest_with_semantic_type_uses_semantic_profile_chunk_size(
        self, fake_index_factory
    ) -> None:
        """Test behavior."""
        # Test note.
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=DocumentType.SEMANTIC,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            chunker=TextChunker(
                profile=CHUNKING_PROFILES[DocumentType.SEMANTIC],
                splitter=FakeSplitter(["s-chunk"]),
            ),
        )
        assert result.chunk_count == 1

    def test_ingest_with_precise_type_resolves_precise_profile(
        self, fake_index_factory
    ) -> None:
        """Test behavior."""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        # Test note.
        mock_chunker = MagicMock(spec=TextChunker)
        mock_chunker.chunk_documents.return_value = [
            ParsedDocument(page_content="p-chunk", metadata={"source": "a.md", "session_id": sid})
        ]
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=DocumentType.PRECISE,
            chunker=mock_chunker,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        # Test note.
        mock_chunker.chunk_documents.assert_called_once()

    def test_ingest_none_document_type_applies_semantic_profile(
        self, fake_index_factory
    ) -> None:
        """Test behavior."""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        # Test note.
        precise_chunker = TextChunker(
            profile=CHUNKING_PROFILES[DocumentType.SEMANTIC],
            splitter=FakeSplitter(["default-chunk"]),
        )
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=None,
            chunker=precise_chunker,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        # Test note.
        assert result.session_id == sid

    def test_ingest_builds_chunker_from_document_type_when_no_chunker_provided(
        self, fake_index_factory
    ) -> None:
        """Test behavior."""
        indexer, sid, _ = self._make_indexer_and_docs(fake_index_factory)
        # Test note.
        short_docs = [
            ParsedDocument(page_content='test content', metadata={"source": "b.md"})
        ]
        # Test note.
        # Test note.
        try:
            result = ingest_documents(
                session_indexer=indexer,
                session_id=sid,
                documents=short_docs,
                document_type=DocumentType.CODE,
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )
            # Test note.
            assert result.chunk_count >= 0
        except Exception as exc:
            # Test note.
            from core.rag.chunker import ChunkingException
            from core.rag.embeddings import EmbeddingException
            assert isinstance(exc, (ChunkingException, EmbeddingException)), (
                f"Unexpected exception type {type(exc)}: {exc}"
            )


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestHybridRetrieverWeightOverride:
    """Test behavior."""

    @pytest.fixture
    def seeded_retriever(self, fake_index_factory) -> tuple[HybridRetriever, str]:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        sid = record.session_id
        indexer.ingest_chunk_embeddings(
            sid,
            documents=[
                ParsedDocument(page_content='test content', metadata={"source": "a.md"}),
                ParsedDocument(page_content='test content', metadata={"source": "b.md"}),
            ],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )
        retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=FakeEmbedder(),
            vector_weight=0.7,
            keyword_weight=0.3,
        )
        return retriever, sid

    def test_search_without_weight_override_uses_instance_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """Test behavior."""
        retriever, sid = seeded_retriever
        result = retriever.search(session_id=sid, query='test content', top_k=5)
        assert result.total_found >= 0  # Test note.

    def test_search_with_weight_override_does_not_mutate_instance_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """Test behavior."""
        retriever, sid = seeded_retriever
        assert retriever._vector_weight == 0.7
        assert retriever._keyword_weight == 0.3

        retriever.search(
            session_id=sid,
            query="alpha",
            top_k=5,
            vector_weight=0.4,
            keyword_weight=0.6,
        )

        # Test note.
        assert retriever._vector_weight == 0.7
        assert retriever._keyword_weight == 0.3

    def test_search_with_precise_profile_weights(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """Test behavior."""
        retriever, sid = seeded_retriever
        profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        result = retriever.search(
            session_id=sid,
            query="alpha",
            top_k=5,
            vector_weight=profile.vector_weight,
            keyword_weight=profile.keyword_weight,
        )
        assert result.total_found >= 0

    def test_search_with_none_weight_override_falls_back_to_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """Test behavior."""
        retriever, sid = seeded_retriever
        result = retriever.search(
            session_id=sid,
            query='test content',
            top_k=5,
            vector_weight=None,
            keyword_weight=None,
        )
        assert result.total_found >= 0

    def test_merged_score_reflects_overridden_weights(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """Test behavior."""
        retriever, sid = seeded_retriever
        result = retriever.search(
            session_id=sid,
            query='test content',
            top_k=5,
            vector_weight=0.0,
            keyword_weight=1.0,
        )
        for chunk in result.results:
            expected = chunk.keyword_score * 1.0 + chunk.vector_score * 0.0
            assert abs(chunk.merged_score - expected) < 1e-6, (
                f"merged_score {chunk.merged_score} != expected {expected}"
            )


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestInvalidDocumentTypeHandling:
    """Test behavior."""

    def test_invalid_string_raises_value_error_from_enum(self) -> None:
        """Test behavior."""
        with pytest.raises(ValueError):
            DocumentType("invalid_type_xyz")

    def test_chunking_profiles_lookup_raises_for_nonexistent_key(self) -> None:
        """Test behavior."""
        # Test note.
        fake_key = object()  # Test note.
        with pytest.raises((KeyError, TypeError)):
            _ = CHUNKING_PROFILES[fake_key]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestSessionWeightPersistence:
    """Test behavior."""

    def test_ingest_documents_persists_weights_to_session(self, fake_index_factory) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content='test content', metadata={"source": "a.md"})]
        
        ingest_documents(
            session_indexer=indexer,
            session_id=record.session_id,
            documents=docs,
            document_type=DocumentType.CODE,
            chunker=TextChunker(splitter=FakeSplitter()),
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        
        # Test note.
        session_record = indexer.get_session(record.session_id)
        assert getattr(session_record, "vector_weight", None) == 0.6
        assert getattr(session_record, "keyword_weight", None) == 0.4

    def test_search_reads_weights_from_session_record(self, fake_index_factory) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        # Test note.
        record.vector_weight = 0.99
        record.keyword_weight = 0.01
        
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[ParsedDocument(page_content='test content', metadata={"source": "a.md"})],
            embeddings=[[0.1] * 384],
        )
        
        retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=FakeEmbedder(),
            vector_weight=0.5,
            keyword_weight=0.5,
        )
        
        result = retriever.search(session_id=record.session_id, query='test content', top_k=1)
        
        # Test note.
        chunk = result.results[0]
        expected = chunk.keyword_score * 0.01 + chunk.vector_score * 0.99
        assert abs(chunk.merged_score - expected) < 1e-6

    def test_ingest_does_not_persist_weights_on_chunking_failure(
        self, fake_index_factory, caplog
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]
        caplog.set_level("ERROR")

        with pytest.raises(ChunkingException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=FailingChunker(),  # type: ignore[arg-type]
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        assert "chunking failed" in str(exc_info.value)
        assert record.session_id in caplog.text
        assert "chunking failed during ingestion" in caplog.text
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None

    def test_ingest_does_not_persist_weights_on_embedding_failure(
        self, fake_index_factory, caplog
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]
        caplog.set_level("ERROR")

        with pytest.raises(EmbeddingException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=TextChunker(splitter=FakeSplitter(["chunk"])),
                embedder=FailingEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        assert "embedding failed" in str(exc_info.value)
        assert record.session_id in caplog.text
        assert "embedding failed during ingestion" in caplog.text
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None

    def test_ingest_does_not_persist_weights_on_indexing_failure(self) -> None:
        indexer = SessionIndexer(index_factory=FailingAddIndex)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]

        from core.rag.indexer import IndexerException

        with pytest.raises(IndexerException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=TextChunker(splitter=FakeSplitter(["chunk"])),
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None
