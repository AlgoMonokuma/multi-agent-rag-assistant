"""Test behavior."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.rag.reranker import CrossEncoderReranker, RerankerException
from core.rag.retriever import HybridSearchResult, HybridRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    page_content: str,
    rank: int = 1,
    merged_score: float = 0.5,
    metadata: dict | None = None,
) -> RetrievedChunk:
    """Test behavior."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_content=page_content,
        metadata=metadata or {"source": "test.pdf"},
        vector_score=0.5,
        keyword_score=0.5,
        merged_score=merged_score,
        rank=rank,
        rerank_score=None,
    )


def _make_mock_cross_encoder(scores: list[float]) -> MagicMock:
    """Test behavior."""
    mock = MagicMock()
    mock.predict.return_value = scores
    return mock


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerInit:
    """Test behavior."""

    def test_init_with_injected_cross_encoder(self):
        """Test behavior."""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=mock_ce,
        )
        assert reranker is not None

    def test_init_stores_model_name(self):
        """Test behavior."""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=mock_ce,
        )
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_init_default_model_name(self):
        """Test behavior."""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)
        assert "MiniLM" in reranker._model_name or "ms-marco" in reranker._model_name

    def test_lazy_load_raises_when_sentence_transformers_missing(self):
        """Test behavior."""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=None,
        )
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RerankerException, match="sentence.transformers"):
                reranker._load_model()


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerEmptyInput:
    """Test behavior."""

    def test_rerank_empty_chunks_returns_empty_list(self):
        """Test behavior."""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=[], top_n=3)

        assert result == []
        mock_ce.predict.assert_not_called()

    def test_reranker_top_n_negative_returns_empty_without_model(self):
        """top_n < 0 follows the same explicit empty-result rule."""
        chunks = [_make_chunk("chunk-1", "content", rank=1)]
        mock_ce = _make_mock_cross_encoder([0.8])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="query", chunks=chunks, top_n=-1)

        assert result == []
        mock_ce.predict.assert_not_called()

    def test_rerank_empty_chunks_does_not_call_model(self):
        """Test behavior."""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        reranker.rerank(query='test content', chunks=[], top_n=5)
        mock_ce.predict.assert_not_called()


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerSorting:
    """Test behavior."""

    def test_rerank_sorts_by_score_descending(self):
        """Test behavior."""
        chunks = [
            _make_chunk("chunk-1", 'test content', rank=1),
            _make_chunk("chunk-2", 'test content', rank=2),
            _make_chunk("chunk-3", 'test content', rank=3),
        ]
        # Test note.
        scores = [0.1, 0.9, 0.5]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=chunks, top_n=3)

        assert len(result) == 3
        # Test note.
        assert result[0].chunk_id == "chunk-2"
        assert result[1].chunk_id == "chunk-3"
        assert result[2].chunk_id == "chunk-1"

    def test_rerank_truncates_to_top_n(self):
        """Test behavior."""
        chunks = [_make_chunk(f"chunk-{i}", f"text {i}", rank=i) for i in range(5)]
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=chunks, top_n=3)

        assert len(result) == 3

    def test_rerank_top_n_larger_than_chunks_returns_all(self):
        """Test behavior."""
        chunks = [_make_chunk(f"chunk-{i}", f"text {i}", rank=i) for i in range(2)]
        scores = [0.8, 0.3]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=chunks, top_n=10)

        assert len(result) == 2

    def test_reranker_top_n_boundary_is_explicit(self):
        """top_n <= 0 returns empty without invoking the model."""
        chunks = [_make_chunk("chunk-1", "content", rank=1)]
        mock_ce = _make_mock_cross_encoder([0.8])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="query", chunks=chunks, top_n=0)

        assert result == []
        mock_ce.predict.assert_not_called()

    def test_reranked_results_preserve_optional_citation_metadata(self):
        metadata = {
            "source": "doc.pdf",
            "chunk_id": "chunk-1",
            "session_id": "sid-1",
            "page": 5,
            "title": "Title",
            "parent_source": "parent.pdf",
            "chunk_index": 2,
        }
        chunks = [_make_chunk("chunk-1", "content", metadata=metadata)]
        mock_ce = _make_mock_cross_encoder([0.8])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="query", chunks=chunks, top_n=1)

        assert result[0].metadata == metadata


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerScoreUpdate:
    """Test behavior."""

    def test_rerank_fills_rerank_score(self):
        """Test behavior."""
        chunks = [
            _make_chunk("chunk-1", 'test content', rank=1),
            _make_chunk("chunk-2", 'test content', rank=2),
        ]
        scores = [0.75, 0.25]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=chunks, top_n=2)

        # Test note.
        assert result[0].chunk_id == "chunk-1"
        assert result[0].rerank_score == pytest.approx(0.75)
        assert result[1].chunk_id == "chunk-2"
        assert result[1].rerank_score == pytest.approx(0.25)

    def test_rerank_updates_rank_to_new_order(self):
        """Test behavior."""
        chunks = [
            _make_chunk("chunk-1", 'test content', rank=1),
            _make_chunk("chunk-2", 'test content', rank=2),
        ]
        scores = [0.1, 0.9]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query='test content', chunks=chunks, top_n=2)

        assert result[0].chunk_id == "chunk-2"
        assert result[0].rank == 1
        assert result[1].chunk_id == "chunk-1"
        assert result[1].rank == 2

    def test_rerank_passes_correct_pairs_to_model(self):
        """Test behavior."""
        query = 'test content'
        chunks = [
            _make_chunk("chunk-1", 'test content', rank=1),
            _make_chunk("chunk-2", 'test content', rank=2),
        ]
        mock_ce = _make_mock_cross_encoder([0.5, 0.8])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        reranker.rerank(query=query, chunks=chunks, top_n=2)

        mock_ce.predict.assert_called_once_with(
            [[query, 'test content'], [query, 'test content']]
        )


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


    def test_reranker_reuses_lazy_loaded_model_instance(self):
        class LazyTestReranker(CrossEncoderReranker):
            def __init__(self):
                super().__init__(cross_encoder=None)
                self.load_calls = 0
                self.model = _make_mock_cross_encoder([0.5])

            def _load_model(self):
                self.load_calls += 1
                return self.model

        reranker = LazyTestReranker()
        chunks = [_make_chunk("chunk-1", "content", rank=1)]

        reranker.rerank(query="query", chunks=chunks, top_n=1)
        reranker.rerank(query="query", chunks=chunks, top_n=1)

        assert reranker.load_calls == 1


class TestCrossEncoderRerankerErrorHandling:
    """Test behavior."""

    def test_rerank_wraps_model_exception_as_reranker_exception(self):
        """Test behavior."""
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError('test content')
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        chunks = [_make_chunk("chunk-1", 'test content', rank=1)]

        with pytest.raises(RerankerException) as exc_info:
            reranker.rerank(query='test content', chunks=chunks, top_n=1)

        message = str(exc_info.value)
        assert "top_n=1" in message
        assert "chunk_count=1" in message
        assert reranker._model_name in message

    def test_reranker_exception_inherits_from_exception(self):
        """Test behavior."""
        assert issubclass(RerankerException, Exception)


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestHybridRetrieverWithReranker:
    """Test behavior."""

    def _make_retriever_with_reranker(
        self, reranker: CrossEncoderReranker, top_k: int = 10, final_top_n: int = 3
    ) -> HybridRetriever:
        """Test behavior."""
        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        return HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=final_top_n,
        )

    def test_search_with_reranker_returns_reranked_results(self):
        """Test behavior."""
        import numpy as np

        # Test note.
        # Test note.
        # Test note.
        mock_ce = MagicMock()

        def _predict_side_effect(pairs):
            """Test behavior."""
            scores = []
            for pair in pairs:
                content = pair[1]
                if 'test content' in content:
                    scores.append(0.9)
                elif 'test content' in content:
                    scores.append(0.5)
                else:
                    scores.append(0.1)
            return scores

        mock_ce.predict.side_effect = _predict_side_effect
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()

        session_id = "test-session"
        mock_record = MagicMock()
        mock_record.vector_chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        mock_record.chunk_map = {
            "chunk-1": MagicMock(page_content='test content', metadata={}),
            "chunk-2": MagicMock(page_content='test content', metadata={}),
            "chunk-3": MagicMock(page_content='test content', metadata={}),
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record

        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0, 1, 2]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: [
            "chunk-1", "chunk-2", "chunk-3"
        ][ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=3,
        )

        result = retriever.search(session_id=session_id, query='test content', top_k=3)

        assert isinstance(result, HybridSearchResult)
        assert len(result.results) <= 3
        # Test note.
        assert result.results[0].rerank_score is not None
        assert result.results[0].rerank_score >= result.results[-1].rerank_score
        # Test note.
        assert 'test content' in result.results[0].page_content

    def test_search_with_reranker_truncates_to_final_top_n(self):
        """Test behavior."""
        import numpy as np

        mock_ce = _make_mock_cross_encoder([0.9, 0.8, 0.7, 0.6, 0.5])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        session_id = "test-session-2"

        chunk_ids = [f"chunk-{i}" for i in range(1, 6)]
        mock_record = MagicMock()
        mock_record.vector_chunk_ids = chunk_ids
        mock_record.chunk_map = {
            cid: MagicMock(page_content=f"text {i}", metadata={})
            for i, cid in enumerate(chunk_ids)
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record
        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: chunk_ids[ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=2,
        )

        result = retriever.search(session_id=session_id, query='test content', top_k=5)
        assert len(result.results) <= 2


# ---------------------------------------------------------------------------
# Test note.
# ---------------------------------------------------------------------------


class TestHybridRetrieverBackwardCompatibility:
    """Test behavior."""

    def test_search_without_reranker_returns_hybrid_results(self):
        """Test behavior."""
        import numpy as np

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        session_id = "compat-session"

        mock_record = MagicMock()
        mock_record.vector_chunk_ids = ["chunk-1", "chunk-2"]
        mock_record.chunk_map = {
            "chunk-1": MagicMock(page_content='test content', metadata={}),
            "chunk-2": MagicMock(page_content='test content', metadata={}),
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record
        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: [
            "chunk-1", "chunk-2"
        ][ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=None,  # Test note.
        )

        result = retriever.search(session_id=session_id, query='test content', top_k=2)

        assert isinstance(result, HybridSearchResult)
        # Test note.
        for chunk in result.results:
            assert chunk.rerank_score is None

    def test_default_reranker_is_none(self):
        """Test behavior."""
        mock_indexer = MagicMock()
        mock_embedder = MagicMock()

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
        )
        assert retriever._reranker is None
