"""Test behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.rag.indexer import IndexerException, SessionIndexer
from core.rag.parser import ParsedDocument
from core.rag.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrievedChunk,
    RetrieverException,
)


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

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
        """Test behavior."""
        total = self.ntotal
        if total == 0:
            empty_d = np.full((query_vectors.shape[0], k), np.inf, dtype=np.float32)
            empty_i = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)
            return empty_d, empty_i

        # Test note.
        actual_k = min(k, total)
        distances = np.arange(actual_k, dtype=np.float32).reshape(1, -1)
        indices = np.arange(actual_k, dtype=np.int64).reshape(1, -1)
        # Test note.
        if k > total:
            pad = k - total
            distances = np.concatenate(
                [distances, np.full((1, pad), np.inf, dtype=np.float32)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((1, pad), -1, dtype=np.int64)], axis=1
            )
        return distances, indices


class FakeEmbedder:
    """Test behavior."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32) * 0.5


class CountingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        return super().embed_texts(texts)


@pytest.fixture
def fake_index_factory() -> Any:
    """Test behavior."""
    def factory() -> FakeFaissIndex:
        return FakeFaissIndex()
    return factory


@pytest.fixture
def seeded_indexer(fake_index_factory: Any) -> tuple[SessionIndexer, str]:
    """Test behavior."""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()
    sid = record.session_id

    indexer.ingest_chunk_embeddings(
        sid,
        documents=[
            ParsedDocument(
                page_content='test content',
                metadata={"source": "weather.pdf", "page": 1, "title": 'test content'},
            ),
            ParsedDocument(
                page_content='test content',
                metadata={"source": "ai_trend.md", "title": 'test content'},
            ),
            ParsedDocument(
                page_content='test content',
                metadata={"source": "economy.pdf", "page": 3, "title": 'test content'},
            ),
        ],
        embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
    )
    return indexer, sid


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestRetrievedChunkModel:
    """Test behavior."""

    def test_retrieved_chunk_contains_required_fields(self) -> None:
        """Test behavior."""
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            page_content='test content',
            metadata={"source": "test.md", "session_id": "sid-1"},
            vector_score=0.85,
            keyword_score=0.0,
            merged_score=0.85,
            rank=1,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.page_content == 'test content'
        assert chunk.metadata["source"] == "test.md"
        assert chunk.vector_score == 0.85
        assert chunk.keyword_score == 0.0
        assert chunk.merged_score == 0.85
        assert chunk.rank == 1

    def test_retrieved_chunk_preserves_citation_metadata(self) -> None:
        """Test behavior."""
        meta = {
            "source": "doc.pdf",
            "chunk_id": "chunk-5",
            "session_id": "sid-abc",
            "page": 2,
            "title": 'test content',
            "parent_source": "doc.pdf",
            "chunk_index": 4,
        }
        chunk = RetrievedChunk(
            chunk_id="chunk-5",
            page_content='test content',
            metadata=meta,
            vector_score=0.9,
            keyword_score=0.5,
            merged_score=0.7,
            rank=2,
        )
        for key in ("source", "chunk_id", "session_id", "page", "title",
                     "parent_source", "chunk_index"):
            assert key in chunk.metadata


class TestHybridSearchResultModel:
    """Test behavior."""

    def test_hybrid_search_result_holds_result_list(self) -> None:
        """Test behavior."""
        result = HybridSearchResult(
            query='test content',
            session_id="sid-1",
            results=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    page_content='test content',
                    metadata={},
                    vector_score=0.9,
                    keyword_score=0.3,
                    merged_score=0.6,
                    rank=1,
                ),
            ],
            total_found=1,
        )
        assert result.query == 'test content'
        assert result.session_id == "sid-1"
        assert len(result.results) == 1
        assert result.total_found == 1


class TestRetrieverException:
    """Test behavior."""

    def test_retriever_exception_is_a_standard_exception(self) -> None:
        """Test behavior."""
        with pytest.raises(RetrieverException, match='test content'):
            raise RetrieverException('test content')


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestVectorRetrieval:
    """Test behavior."""

    def test_vector_search_returns_chunks_with_ordinal_mapping(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=2)

        assert len(result.results) == 2
        for chunk in result.results:
            assert chunk.chunk_id.startswith("chunk-")
            assert chunk.vector_score >= 0.0
            assert chunk.page_content != ""
            assert "source" in chunk.metadata

    def test_vector_search_respects_session_isolation(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        rec_a = indexer.create_session()
        rec_b = indexer.create_session()

        indexer.ingest_chunk_embeddings(
            rec_a.session_id,
            documents=[
                ParsedDocument(page_content='test content', metadata={"source": "a.md"}),
            ],
            embeddings=[[0.1] * 384],
        )
        indexer.ingest_chunk_embeddings(
            rec_b.session_id,
            documents=[
                ParsedDocument(page_content='test content', metadata={"source": "b.md"}),
            ],
            embeddings=[[0.2] * 384],
        )

        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )
        result_a = retriever.search(session_id=rec_a.session_id, query='test content', top_k=5)
        result_b = retriever.search(session_id=rec_b.session_id, query='test content', top_k=5)

        # Test note.
        # Test note.
        for c in result_a.results:
            assert c.metadata.get("session_id") == rec_a.session_id
            assert c.metadata.get("source") != "b.md"
        for c in result_b.results:
            assert c.metadata.get("session_id") == rec_b.session_id
            assert c.metadata.get("source") != "a.md"


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestKeywordRetrieval:
    """Test behavior."""

    def test_keyword_search_finds_matching_chunks(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=5)

        # Test note.
        matching_contents = [
            c.page_content for c in result.results if c.keyword_score > 0.0
        ]
        assert len(matching_contents) >= 1  # Test note.


class TestSessionIndexerChunkAccessor:
    """Test behavior."""

    def test_get_chunk_document_returns_stored_parsed_document(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(page_content='test content', metadata={"source": "a.md"}),
            ],
            embeddings=[[0.1] * 384],
        )

        doc = indexer.get_chunk_document(record.session_id, "chunk-1")

        assert doc.page_content == 'test content'
        assert doc.metadata["source"] == "a.md"

    def test_get_chunk_document_raises_on_unknown_chunk(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()

        with pytest.raises(IndexerException, match="chunk_id"):
            indexer.get_chunk_document(record.session_id, "nonexistent")

    def test_list_chunk_documents_returns_all_stored_documents(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(page_content='test content', metadata={"source": "a.md"}),
                ParsedDocument(page_content='test content', metadata={"source": "b.md"}),
            ],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )

        docs = indexer.list_chunk_documents(record.session_id)

        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert contents == {'test content', 'test content'}


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestHybridMerge:
    """Test behavior."""

    def test_merge_deduplicates_by_chunk_id(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=10)

        chunk_ids = [c.chunk_id for c in result.results]
        assert len(chunk_ids) == len(set(chunk_ids)), 'test content'

    def test_merge_results_contain_all_three_scores(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=5)

        for chunk in result.results:
            assert isinstance(chunk.vector_score, float)
            assert isinstance(chunk.keyword_score, float)
            assert isinstance(chunk.merged_score, float)
            assert chunk.vector_score >= 0.0
            assert chunk.keyword_score >= 0.0
            assert chunk.merged_score >= 0.0

    def test_merge_results_have_deterministic_ordering(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result_1 = retriever.search(session_id=sid, query='test content', top_k=5)
        result_2 = retriever.search(session_id=sid, query='test content', top_k=5)

        ids_1 = [c.chunk_id for c in result_1.results]
        ids_2 = [c.chunk_id for c in result_2.results]
        assert ids_1 == ids_2

    def test_merge_results_have_sequential_ranks(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=5)

        ranks = [c.rank for c in result.results]
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected

    def test_results_have_complete_citation_metadata(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=3)

        for chunk in result.results:
            assert "source" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "session_id" in chunk.metadata
            assert chunk.metadata["session_id"] == sid

    def test_search_results_include_required_citation_metadata(
        self, fake_index_factory: Any
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[ParsedDocument(page_content="citation content", metadata={})],
            embeddings=[[0.1] * 384],
        )
        retriever = HybridRetriever(session_indexer=indexer, embedder=FakeEmbedder())

        result = retriever.search(
            session_id=record.session_id, query="citation", top_k=1
        )

        metadata = result.results[0].metadata
        assert metadata["source"] == "unknown"
        assert metadata["chunk_id"] == "chunk-1"
        assert metadata["session_id"] == record.session_id

    def test_search_results_normalize_empty_citation_source(
        self, fake_index_factory: Any
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(page_content="none source", metadata={"source": None}),
                ParsedDocument(page_content="blank source", metadata={"source": ""}),
            ],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )
        retriever = HybridRetriever(session_indexer=indexer, embedder=FakeEmbedder())

        result = retriever.search(session_id=record.session_id, query="source", top_k=2)

        assert {chunk.metadata["source"] for chunk in result.results} == {"unknown"}

    def test_reranked_results_preserve_optional_citation_metadata(
        self, fake_index_factory: Any
    ) -> None:
        class PassthroughReranker:
            def rerank(self, query, chunks, top_n):
                return chunks[:top_n]

        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(
                    page_content="optional citation content",
                    metadata={
                        "source": "doc.pdf",
                        "page": 7,
                        "title": "Title",
                        "parent_source": "parent.pdf",
                        "chunk_index": 3,
                    },
                )
            ],
            embeddings=[[0.1] * 384],
        )
        retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=FakeEmbedder(),
            reranker=PassthroughReranker(),  # type: ignore[arg-type]
            final_top_n=1,
        )

        result = retriever.search(
            session_id=record.session_id, query="optional", top_k=1
        )

        metadata = result.results[0].metadata
        for key in ("page", "title", "parent_source", "chunk_index"):
            assert key in metadata


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestScopeBoundary:
    """Test behavior."""

    def test_retriever_does_not_expose_reranking_api(self) -> None:
        """Test behavior."""
        assert not hasattr(HybridRetriever, "rerank")
        assert not hasattr(HybridRetriever, "cross_encode")
        assert not hasattr(HybridRetriever, "compress")


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test behavior."""

    def test_empty_query_raises_retriever_exception(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="query"):
            retriever.search(session_id=sid, query="", top_k=5)

    def test_whitespace_only_query_raises_retriever_exception(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="query"):
            retriever.search(session_id=sid, query="   ", top_k=5)

    def test_unknown_session_raises_retriever_exception(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="Session"):
            retriever.search(session_id="nonexistent", query='test content', top_k=5)

    def test_empty_index_returns_empty_results(
        self, fake_index_factory: Any
    ) -> None:
        """Test behavior."""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(
            session_id=record.session_id, query='test content', top_k=5
        )

        assert result.results == []
        assert result.total_found == 0

    def test_search_top_k_zero_returns_empty_without_embedding(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        indexer, sid = seeded_indexer
        embedder = CountingEmbedder()
        retriever = HybridRetriever(session_indexer=indexer, embedder=embedder)

        result = retriever.search(session_id=sid, query="anything", top_k=0)

        assert result.results == []
        assert result.total_found == 0
        assert embedder.calls == 0

    def test_search_top_k_negative_returns_empty_without_embedding(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        indexer, sid = seeded_indexer
        embedder = CountingEmbedder()
        retriever = HybridRetriever(session_indexer=indexer, embedder=embedder)

        result = retriever.search(session_id=sid, query="anything", top_k=-1)

        assert result.results == []
        assert result.total_found == 0
        assert embedder.calls == 0

    def test_top_k_larger_than_corpus_returns_all_available(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """Test behavior."""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query='test content', top_k=100)

        assert len(result.results) <= 3  # Test note.
        assert result.total_found == len(result.results)
