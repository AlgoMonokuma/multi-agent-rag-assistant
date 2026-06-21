"""Stub node functions for the LangGraph agent workflow.

Each node accepts an AgentState dict and returns a partial state update.
These are stubs that will be replaced with real implementations in later stories:
  - researcher_node -> Story 3.3 (HybridRetriever integration)
  - reporter_node   -> Story 3.2 (LLMGenerator / Groq integration)
  - reviewer_node   -> Story 3.6 (Reviewer LLM and quality gate)

Rules followed by all stub nodes:
  1. Accept state: AgentState, return dict of state updates.
  2. Use the shared logger from core.log (not print or logging directly).
  3. Do not raise exceptions for valid state input.
  4. Log a DEBUG message indicating which node ran and key state fields.
"""

from core.agent.state import AgentState
from core.log import get_logger

logger = get_logger(__name__)


def researcher_node(state: AgentState) -> dict:
    """Retrieve relevant chunks for the user query.

    Stub: will call HybridRetriever with session_id in Story 3.3.

    Args:
        state: Current workflow state containing query and session_id.

    Returns:
        Partial state update with retrieved_chunks (empty stub list)
        and an iteration_count increment of 1.
    """
    logger.debug(
        "researcher_node: query=%r session_id=%r",
        state.get("query"),
        state.get("session_id"),
    )
    return {
        "retrieved_chunks": [],
        "iteration_count": 1,  # Triggers operator.add reducer: total += 1
    }


def reporter_node(state: AgentState) -> dict:
    """Generate a draft answer from retrieved chunks.

    Stub: will call LLMGenerator (Groq) in Story 3.2.

    Args:
        state: Current workflow state containing retrieved_chunks.

    Returns:
        Partial state update with draft_answer and citations.
    """
    chunk_count = len(state.get("retrieved_chunks") or [])
    logger.debug("reporter_node: incoming chunk count=%d", chunk_count)
    return {
        "draft_answer": "[stub] No answer generated yet.",
        "citations": [],
    }


def reviewer_node(state: AgentState) -> dict:
    """Review and approve (or reject) the draft answer.

    Stub: will call a reviewer LLM in Story 3.6.
    The stub always approves to keep the graph operational during development.

    Args:
        state: Current workflow state containing draft_answer.

    Returns:
        Partial state update with review_passed, review_feedback, and final_answer.
    """
    draft = (state.get("draft_answer") or "")[:80]
    logger.debug("reviewer_node: reviewing draft=%r", draft)
    return {
        "review_passed": True,
        "review_feedback": "",
        "final_answer": state.get("draft_answer", ""),
    }
