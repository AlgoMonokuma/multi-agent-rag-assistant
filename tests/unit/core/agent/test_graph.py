"""Unit tests for Story 3.1: LangGraph State Graph Foundation.

Tests verify:
  AC1  - graph imports and compiles without error
  AC2  - AgentState has expected fields with correct defaults
  AC3  - compiled graph nodes include researcher, reporter, reviewer
  AC4  - graph.invoke() completes synchronously without exception
  AC5  - iteration_count == 1 after one full cycle (stub nodes)
  AC6  - graph routes back to researcher when review_passed=False and below max
  AC7  - graph routes to END when review_passed=True
  AC8  - graph routes to END when iteration_count >= max_iterations
  AC9  - `from core.agent.graph import build_graph` resolves cleanly
"""

import importlib
import operator
from typing import get_type_hints

import pytest

from core.agent.graph import build_graph, _route_after_review
from core.agent.state import AgentState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_state(**overrides) -> dict:
    """Return a minimal valid state dict for graph invocation."""
    base = {
        "query": "What is RAG?",
        "session_id": "test-session-01",
        "max_iterations": 3,
        "iteration_count": 0,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# AC9: module import resolves
# ---------------------------------------------------------------------------

def test_module_import_resolves():
    """AC9: from core.agent.graph import build_graph should not raise."""
    mod = importlib.import_module("core.agent.graph")
    assert hasattr(mod, "build_graph"), "build_graph not found in core.agent.graph"


# ---------------------------------------------------------------------------
# AC1: graph compiles
# ---------------------------------------------------------------------------

def test_build_graph_returns_compiled_graph():
    """AC1: build_graph() returns a compiled graph without raising ImportError."""
    from langgraph.graph.state import CompiledStateGraph

    graph = build_graph()
    assert graph is not None
    assert isinstance(graph, CompiledStateGraph)


def test_build_graph_singleton():
    """build_graph() returns the same object on repeated calls (lazy singleton)."""
    g1 = build_graph()
    g2 = build_graph()
    assert g1 is g2, "build_graph should return the same compiled instance"


# ---------------------------------------------------------------------------
# AC2: AgentState schema
# ---------------------------------------------------------------------------

def test_agent_state_is_typed_dict():
    """AC2: AgentState is a TypedDict and all expected keys are declared."""
    expected_keys = {
        "query",
        "session_id",
        "retrieved_chunks",
        "draft_answer",
        "citations",
        "review_passed",
        "review_feedback",
        "iteration_count",
        "max_iterations",
        "final_answer",
    }
    declared = set(AgentState.__annotations__.keys())
    assert expected_keys == declared, (
        f"AgentState missing keys: {expected_keys - declared}"
    )


def test_agent_state_iteration_count_uses_add_reducer():
    """AC2: iteration_count annotation uses operator.add for LangGraph reducer."""
    annotation = get_type_hints(AgentState, include_extras=True)["iteration_count"]
    # Annotated[int, operator.add] — check metadata contains operator.add
    metadata = getattr(annotation, "__metadata__", ())
    assert operator.add in metadata, (
        "iteration_count must be Annotated[int, operator.add] to enable auto-accumulation"
    )


# ---------------------------------------------------------------------------
# AC3: graph nodes
# ---------------------------------------------------------------------------

def test_graph_has_expected_nodes():
    """AC3: compiled graph contains exactly researcher, reporter, reviewer nodes."""
    graph = build_graph()
    nodes = graph.get_graph().nodes
    for required in ("researcher", "reporter", "reviewer"):
        assert required in nodes, f"Node '{required}' missing from compiled graph"


# ---------------------------------------------------------------------------
# AC4: synchronous invocation
# ---------------------------------------------------------------------------

def test_graph_invoke_synchronous_success():
    """AC4: graph.invoke() with minimal state completes without exception."""
    graph = build_graph()
    result = graph.invoke(_minimal_state())
    assert result is not None
    assert isinstance(result, dict)


def test_graph_invoke_returns_final_answer_key():
    """AC4 extension: invoked graph state includes final_answer key."""
    graph = build_graph()
    result = graph.invoke(_minimal_state())
    assert "final_answer" in result


# ---------------------------------------------------------------------------
# AC5: iteration_count increments per cycle
# ---------------------------------------------------------------------------

def test_iteration_count_increments_per_cycle():
    """AC5: iteration_count == 1 after one full researcher/reporter/reviewer cycle."""
    graph = build_graph()
    result = graph.invoke(_minimal_state(iteration_count=0))
    # Stub reviewer always approves, so exactly 1 cycle runs
    assert result.get("iteration_count") == 1, (
        f"Expected iteration_count=1 after one cycle, got {result.get('iteration_count')}"
    )


# ---------------------------------------------------------------------------
# AC6: routing loops back when review fails and below max
# ---------------------------------------------------------------------------

def test_routing_loops_back_when_review_fails_and_below_max():
    """AC6: _route_after_review returns 'researcher' when review_passed=False and below max."""
    state: AgentState = {
        "review_passed": False,
        "iteration_count": 1,
        "max_iterations": 3,
    }
    result = _route_after_review(state)
    assert result == "researcher", (
        f"Expected 'researcher' route for failed review below max, got {result!r}"
    )


# ---------------------------------------------------------------------------
# AC7: routing ends when review passes
# ---------------------------------------------------------------------------

def test_routing_ends_when_review_passes():
    """AC7: _route_after_review returns END when review_passed=True."""
    from langgraph.graph import END

    state: AgentState = {
        "review_passed": True,
        "iteration_count": 1,
        "max_iterations": 3,
    }
    result = _route_after_review(state)
    assert result == END, (
        f"Expected END when review_passed=True, got {result!r}"
    )


# ---------------------------------------------------------------------------
# AC8: routing ends at max iterations
# ---------------------------------------------------------------------------

def test_routing_ends_at_max_iterations():
    """AC8: _route_after_review returns END when iteration_count >= max_iterations."""
    from langgraph.graph import END

    state: AgentState = {
        "review_passed": False,
        "iteration_count": 3,
        "max_iterations": 3,
    }
    result = _route_after_review(state)
    assert result == END, (
        f"Expected END at max_iterations, got {result!r}"
    )


def test_routing_ends_when_iteration_exceeds_max():
    """AC8 extension: also ends when iteration_count > max_iterations (overflow guard)."""
    from langgraph.graph import END

    state: AgentState = {
        "review_passed": False,
        "iteration_count": 10,
        "max_iterations": 3,
    }
    result = _route_after_review(state)
    assert result == END


def test_routing_default_max_when_missing():
    """AC8 extension: missing max_iterations uses default of 3 — loops below 3."""
    state: AgentState = {
        "review_passed": False,
        "iteration_count": 1,
        # max_iterations deliberately omitted → default 3 applies
    }
    result = _route_after_review(state)
    assert result == "researcher", (
        f"Expected 'researcher' loop when iteration_count=1 and default max=3, got {result!r}"
    )
