"""Node functions for the LangGraph agent workflow.

Each node accepts an AgentState dict and returns a partial state update.

Current implementation status:
  - researcher_node -> stub (HybridRetriever integration deferred to Story 3.3)
  - reporter_node   -> real (LLMGenerator / Groq — implemented in Story 3.2)
  - reviewer_node   -> stub (Reviewer LLM / quality gate deferred to Story 3.6)

Rules followed by all nodes:
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
    """Generate a draft answer from retrieved chunks using LLMGenerator.

    Calls LLMGenerator (Groq) to produce a grounded answer from the
    retrieved and re-ranked chunks stored in state.

    Args:
        state: Current workflow state containing retrieved_chunks, query,
               and session_id.

    Returns:
        Partial state update with draft_answer (str) and citations (list[dict]).
        Citations are stored as plain dicts for JSON-compatibility with
        downstream nodes.
    """
    from core.rag.generator import GeneratorException, LLMGenerator  # lazy import

    query = state.get("query", "")
    chunks = state.get("retrieved_chunks") or []
    session_id = state.get("session_id", "")

    logger.debug("reporter_node: chunk_count=%d session_id=%r", len(chunks), session_id)

    generator = LLMGenerator()
    try:
        result = generator.generate(query=query, chunks=chunks, session_id=session_id)
    except GeneratorException as exc:
        logger.error("reporter_node: generation failed: %s", exc)
        return {
            "draft_answer": f"[error] Answer generation failed: {exc}",
            "citations": [],
        }

    return {
        "draft_answer": result.answer,
        "citations": [{"source": c.source, "chunk_id": c.chunk_id} for c in result.citations],
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
