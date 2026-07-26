# tests/unit/core/agent/test_web_search_node.py
"""Unit tests for web_search_node in core/agent/nodes.py.

Tests drive Story 3.4 AC:
  AC 1, 2 — success path: client returns results; web_search_results contains normalized dicts
  AC 6    — empty results and network failure handled gracefully (fail-open)
  AC 8    — WebSearchException from client fails open (missing API key scenario)
"""

from unittest.mock import MagicMock

from core.agent.exceptions import WebSearchException
from core.agent.nodes import web_search_node
from core.agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(results):
    """Return a mock Tavily client that returns the given results list."""
    mock = MagicMock()
    mock.results.return_value = results
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_web_search_node_success():
    """AC 1, 2: Client returns results; web_search_results contains normalized dicts."""
    raw = [{"url": "https://example.com", "content": "RAG explained.", "score": 0.9}]
    mock_client = _make_mock_client(raw)

    state: AgentState = {"query": "What is RAG?"}
    result = web_search_node(state, _client=mock_client)

    assert len(result["web_search_results"]) == 1
    assert result["web_search_results"][0]["url"] == "https://example.com"
    assert result["web_search_results"][0]["content"] == "RAG explained."
    assert result["web_search_results"][0]["score"] == 0.9
    mock_client.results.assert_called_once_with(query="What is RAG?", max_results=5)


def test_web_search_node_empty_results():
    """AC 6: Empty results from client pass through gracefully."""
    mock_client = _make_mock_client([])
    state: AgentState = {"query": "obscure query"}
    result = web_search_node(state, _client=mock_client)
    assert result["web_search_results"] == []


def test_web_search_node_missing_api_key():
    """AC 8: WebSearchException from client; node fails open."""
    mock_client = MagicMock()
    mock_client.results.side_effect = WebSearchException("TAVILY_API_KEY is not set.")
    state: AgentState = {"query": "What is RAG?"}
    result = web_search_node(state, _client=mock_client)
    assert result["web_search_results"] == []


def test_web_search_node_network_failure():
    """AC 6: Generic network exception; node fails open."""
    mock_client = _make_mock_client(None)
    mock_client.results.side_effect = ConnectionError("network timeout")
    state: AgentState = {"query": "test"}
    result = web_search_node(state, _client=mock_client)
    assert result["web_search_results"] == []
