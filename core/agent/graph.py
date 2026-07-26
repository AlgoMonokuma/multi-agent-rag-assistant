"""LangGraph state graph factory for the Epic 3 agent workflow.

Builds and compiles the full agent StateGraph with researcher, reporter,
and reviewer nodes. The graph is returned as a lazy-compiled singleton
via build_graph() to mirror the reuse pattern in core/rag/reranker.py.

Graph topology (Story 3.4 updated):
    START -> researcher -> (conditional: needs_web_search?)
        -> web_search -> reporter -> reviewer -> (conditional)
        -> reporter -> reviewer -> (conditional)
    Conditional edges from reviewer:
        review_passed=True              -> END
        review_passed=False + below max -> researcher  (retry loop)
        iteration_count >= max          -> END          (safety ceiling)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from core.agent.nodes import reporter_node, researcher_node, reviewer_node, web_search_node
from core.agent.state import AgentState
from core.log import get_logger

logger = get_logger(__name__)

_DEFAULT_MAX_ITERATIONS = 3

# Module-level singleton — mirroring the lazy-load pattern from CrossEncoderReranker.
_compiled_graph: CompiledStateGraph | None = None


def _route_after_researcher(state: AgentState) -> str:
    """Route to web_search_node or reporter after researcher runs.

    Args:
        state: Current workflow state after researcher_node has run.

    Returns:
        "web_search" if needs_web_search is True, else "reporter".
    """
    if state.get("needs_web_search", False):
        logger.debug("route_after_researcher: needs_web_search=True -> web_search")
        return "web_search"
    logger.debug("route_after_researcher: needs_web_search=False -> reporter")
    return "reporter"


def _route_after_review(state: AgentState) -> str:
    """Routing function called after reviewer_node to decide the next node.

    Args:
        state: Current workflow state after reviewer_node has run.

    Returns:
        "researcher" to retry, or END (a sentinel string) to finish.
    """
    if state.get("review_passed"):
        logger.debug("route_after_review: review passed -> END")
        return END

    current = state.get("iteration_count", 0)
    ceiling = state.get("max_iterations", _DEFAULT_MAX_ITERATIONS)

    if current >= ceiling:
        logger.debug(
            "route_after_review: iteration_count=%d >= max_iterations=%d -> END",
            current,
            ceiling,
        )
        return END

    logger.debug(
        "route_after_review: review failed (iteration=%d < max=%d) -> researcher",
        current,
        ceiling,
    )
    return "researcher"


def _create_graph() -> CompiledStateGraph:
    """Assemble and compile the agent StateGraph.

    Returns:
        A compiled LangGraph graph ready for .invoke() / .stream() calls.
    """
    graph = StateGraph(AgentState)

    graph.add_node("researcher", researcher_node)
    graph.add_node("web_search", web_search_node)  # NEW — Story 3.4
    graph.add_node("reporter", reporter_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("researcher")  # equivalent to add_edge(START, "researcher")

    # NEW — Story 3.4: replace direct researcher->reporter edge with conditional routing
    graph.add_conditional_edges(
        "researcher",
        _route_after_researcher,
        {
            "web_search": "web_search",
            "reporter": "reporter",
        },
    )
    graph.add_edge("web_search", "reporter")  # NEW — Story 3.4

    graph.add_edge("reporter", "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        _route_after_review,
        {
            "researcher": "researcher",
            END: END,
        },
    )

    compiled = graph.compile()
    logger.debug("_create_graph: LangGraph agent graph compiled successfully")
    return compiled


def build_graph() -> CompiledStateGraph:
    """Return the compiled LangGraph agent graph (lazy singleton).

    Thread-safety note: this function is not thread-safe during the first
    call. For the current dev-server use case (sequential requests) this
    is acceptable. Multi-threaded deployment can add a threading.Lock if
    needed in a future story.

    Returns:
        The shared compiled CompiledStateGraph instance.
    """
    global _compiled_graph
    if _compiled_graph is None:
        logger.debug("build_graph: compiling agent graph (first call)")
        _compiled_graph = _create_graph()
    return _compiled_graph
