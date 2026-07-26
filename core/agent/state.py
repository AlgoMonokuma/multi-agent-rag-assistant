"""AgentState TypedDict for LangGraph workflow state management.

Defines the shared state schema passed between all agent nodes in the
Epic 3 orchestration graph. All fields use TypedDict to satisfy
LangGraph's duck-typed dict access requirements.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class AgentState(TypedDict, total=False):
    """Shared state schema for the LangGraph agent workflow.

    Fields are populated progressively as the graph advances through nodes:
    - researcher populates: retrieved_chunks, iteration_count
    - reporter populates: draft_answer, citations
    - reviewer populates: review_passed, review_feedback, final_answer

    Notes:
        iteration_count uses the operator.add reducer so LangGraph
        automatically accumulates incremental updates (e.g., returning
        {"iteration_count": 1} from a node adds 1 to the current total).
        This avoids the need for nodes to read-modify-write the full state.
    """

    query: str
    """The user's question submitted to the agent workflow."""

    session_id: str
    """Session identifier used for RAG lookup in the SessionIndexer."""

    retrieved_chunks: list
    """Chunks returned by HybridRetriever; populated by researcher_node."""

    draft_answer: str
    """LLM-generated answer draft; populated by reporter_node."""

    citations: list
    """Cited chunk references extracted by reporter_node."""

    review_passed: bool
    """Whether the reviewer agent approved the draft answer."""

    review_feedback: str
    """Reviewer's explanation when review_passed is False."""

    iteration_count: Annotated[int, operator.add]
    """Cycle counter. Uses operator.add reducer for automatic accumulation."""

    max_iterations: int
    """Safety ceiling for retry loops. Defaults to 3 when not provided."""

    final_answer: str
    """Confirmed answer after review passes; set by reviewer_node."""

    needs_web_search: bool
    """Set by researcher_node when retrieved chunks < WEB_SEARCH_THRESHOLD."""

    web_search_results: list
    """Web search snippets from web_search_node; consumed by reporter_node."""
