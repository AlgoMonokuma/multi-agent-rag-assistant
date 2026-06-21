# Story 3.1: LangGraph State Graph Foundation

Status: Complete

## User Story

As a developer, I want a compiled LangGraph state graph foundation so that future researcher, reporter, and reviewer agents can be integrated without redesigning workflow state, routing, or retry behavior.

## Scope

This story creates the first Epic 3 agent workflow skeleton. It defines the shared `AgentState`, adds stub researcher/reporter/reviewer nodes, compiles a LangGraph `StateGraph`, and verifies the retry/end routing behavior before real retrieval and LLM calls are added.

## Acceptance Criteria

1. Given the agent workflow module, when `from core.agent.graph import build_graph` runs, then the import resolves without error.
2. Given `build_graph()` is called, when the graph is first requested, then a LangGraph `CompiledStateGraph` is assembled and returned.
3. Given repeated `build_graph()` calls in one process, when the graph is already compiled, then the same compiled singleton instance is reused.
4. Given the shared agent state schema, when inspected, then it declares `query`, `session_id`, `retrieved_chunks`, `draft_answer`, `citations`, `review_passed`, `review_feedback`, `iteration_count`, `max_iterations`, and `final_answer`.
5. Given the workflow runs with minimal valid state, when `graph.invoke()` executes, then the stub researcher, reporter, and reviewer cycle completes synchronously without exception.
6. Given a completed stub cycle, when the reviewer approves the draft, then the graph routes to `END`.
7. Given review failure below the iteration ceiling, when routing runs after review, then the graph routes back to `researcher`.
8. Given review failure at or above `max_iterations`, when routing runs after review, then the graph routes to `END` as a safety ceiling.
9. Given the researcher stub runs, when it returns an iteration update, then `iteration_count` accumulates through the LangGraph reducer.

## Implementation Notes

- `core/agent/state.py` defines `AgentState` as a `TypedDict` with `iteration_count: Annotated[int, operator.add]` for LangGraph accumulation.
- `core/agent/nodes.py` contains stub node implementations:
  - `researcher_node` returns an empty `retrieved_chunks` list and increments `iteration_count`.
  - `reporter_node` returns a placeholder draft answer and empty citations.
  - `reviewer_node` approves the placeholder draft and copies it to `final_answer`.
- `core/agent/graph.py` builds the graph topology:

```text
START -> researcher -> reporter -> reviewer -> conditional route

review_passed=True              -> END
review_passed=False + below max -> researcher
iteration_count >= max          -> END
```

- `build_graph()` returns a lazy-compiled singleton to mirror the model reuse pattern used elsewhere in the RAG runtime.
- Stub nodes intentionally avoid external model calls so the graph foundation can be tested deterministically.

## Out of Scope

- Real `HybridRetriever` integration in the researcher node
- Groq or other LLM answer generation in the reporter node
- LLM-based review or quality gate logic
- Streaming graph events to the UI
- Persistent graph checkpointing
- LangSmith, Phoenix, or production tracing integration

## Definition of Done

- `core/agent/state.py` defines the shared state contract.
- `core/agent/nodes.py` provides deterministic stub node functions.
- `core/agent/graph.py` compiles and exposes the LangGraph workflow through `build_graph()`.
- `core/agent/__init__.py` exports `build_graph`.
- `tests/unit/core/agent/test_graph.py` covers import, compilation, singleton reuse, state schema, synchronous invocation, iteration accumulation, and conditional routing.
- Public docs and roadmap list Story 3.1 as complete while leaving downstream real agent behavior in future stories.

## Completion Notes

- Added the Epic 3 LangGraph workflow package under `core/agent`.
- Added a typed state contract for future researcher, reporter, and reviewer nodes.
- Added deterministic stub nodes so the graph can compile and run before external integrations are introduced.
- Added conditional review routing with retry behavior and a max-iteration safety ceiling.
- Added focused unit coverage for the graph foundation and routing logic.
