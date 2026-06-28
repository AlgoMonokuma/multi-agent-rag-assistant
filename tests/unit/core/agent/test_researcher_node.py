"""Unit tests for researcher_node in core/agent/nodes.py.

Tests drive Story 3.3 AC:
  AC 1, 2, 3  — success path: chunks retrieved and stored, iteration_count == 1
  AC 4        — fail-open on RetrieverException / RerankerException
  AC 5        — empty results handled gracefully (no error)
  AC 6        — HybridRetriever injected via _retriever kwarg (DI-friendly)
  AC 7        — all 5 test scenarios covered here
"""

from unittest.mock import MagicMock

from core.agent import nodes as nodes_module
from core.agent.nodes import researcher_node
from core.agent.state import AgentState
from core.rag.reranker import RerankerException
from core.rag.retriever import HybridSearchResult, RetrievedChunk, RetrieverException


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str = "chunk-1", source: str = "doc.pdf") -> RetrievedChunk:
    """Return a minimal RetrievedChunk for use in test fixtures."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_content="Some relevant content.",
        metadata={"source": source, "chunk_id": chunk_id, "session_id": "s1"},
        vector_score=0.9,
        keyword_score=0.5,
        merged_score=0.8,
        rank=1,
    )


def _make_search_result(chunks: list) -> HybridSearchResult:
    """Return a HybridSearchResult wrapping the given chunks."""
    return HybridSearchResult(
        query="What is RAG?",
        session_id="s1",
        results=chunks,
        total_found=len(chunks),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_researcher_node_success():
    """AC 1, 2, 3: Mock retriever returns chunks; state updated correctly."""
    chunk = _make_chunk()
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = _make_search_result([chunk])

    state: AgentState = {"query": "What is RAG?", "session_id": "s1"}
    result = researcher_node(state, _retriever=mock_retriever)

    assert result["retrieved_chunks"] == [chunk]
    assert result["iteration_count"] == 1
    mock_retriever.search.assert_called_once_with(
        session_id="s1", query="What is RAG?", top_k=10
    )


def test_researcher_node_empty_session():
    """AC 5: Empty results from retriever are passed through gracefully."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = _make_search_result([])

    state: AgentState = {"query": "What is RAG?", "session_id": "s1"}
    result = researcher_node(state, _retriever=mock_retriever)

    assert result["retrieved_chunks"] == []
    assert result["iteration_count"] == 1


def test_researcher_node_retriever_failure():
    """AC 4: RetrieverException is caught; node fails open with empty chunks."""
    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = RetrieverException("Session not found: s1")

    state: AgentState = {"query": "What is RAG?", "session_id": "s1"}
    result = researcher_node(state, _retriever=mock_retriever)

    assert result["retrieved_chunks"] == []
    assert result["iteration_count"] == 1


def test_researcher_node_reranker_failure():
    """AC 4: RerankerException (raised inside retriever.search) is caught; fail open."""
    mock_retriever = MagicMock()
    mock_retriever.search.side_effect = RerankerException("model failed")

    state: AgentState = {"query": "What is RAG?", "session_id": "s1"}
    result = researcher_node(state, _retriever=mock_retriever)

    assert result["retrieved_chunks"] == []
    assert result["iteration_count"] == 1


def test_researcher_node_increments_iteration_count():
    """AC 3: iteration_count is always 1 in the returned dict (drives operator.add reducer)."""
    mock_retriever = MagicMock()
    mock_retriever.search.return_value = _make_search_result([])

    state: AgentState = {"query": "q", "session_id": "s2"}
    result = researcher_node(state, _retriever=mock_retriever)

    assert "iteration_count" in result
    assert result["iteration_count"] == 1


def test_get_default_retriever_wires_shared_indexer_and_reranker(monkeypatch):
    """AC 6: default retriever construction should use the shared indexer and reranker."""
    monkeypatch.setattr(nodes_module, "_default_retriever", None)

    embedder = object()
    session_indexer = object()
    reranker = object()
    built_retriever = object()

    embedder_cls = MagicMock(return_value=embedder)
    indexer_getter = MagicMock(return_value=session_indexer)
    reranker_cls = MagicMock(return_value=reranker)
    retriever_cls = MagicMock(return_value=built_retriever)

    monkeypatch.setattr(
        "core.rag.embeddings.SentenceTransformerEmbedder",
        embedder_cls,
    )
    monkeypatch.setattr(
        "core.rag.indexer.get_default_session_indexer",
        indexer_getter,
    )
    monkeypatch.setattr(
        "core.rag.reranker.CrossEncoderReranker",
        reranker_cls,
    )
    monkeypatch.setattr(
        "core.rag.retriever.HybridRetriever",
        retriever_cls,
    )

    result = nodes_module._get_default_retriever()

    assert result is built_retriever
    embedder_cls.assert_called_once_with()
    indexer_getter.assert_called_once_with()
    reranker_cls.assert_called_once_with()
    retriever_cls.assert_called_once_with(
        session_indexer=session_indexer,
        embedder=embedder,
        reranker=reranker,
        final_top_n=5,
    )

    cached = nodes_module._get_default_retriever()
    assert cached is built_retriever
    retriever_cls.assert_called_once_with(
        session_indexer=session_indexer,
        embedder=embedder,
        reranker=reranker,
        final_top_n=5,
    )
