# Story 3.3: Agent Researcher Implementation

Status: Complete

## User Story

As a user, I want the agent to retrieve relevant document chunks when I ask a question so that the reporter node can generate a grounded answer using real content rather than an empty context.

## Scope

This story replaces the stub `researcher_node` with a real implementation that calls `HybridRetriever` (with built-in `CrossEncoderReranker`) to fetch and re-rank document chunks from the user's session. It wires the retrieval pipeline into the LangGraph agent workflow so that the reporter node receives meaningful context on every query.

## Acceptance Criteria

1. `researcher_node` in `core/agent/nodes.py` calls `HybridRetriever.search()` using `session_id` and `query` from `AgentState`.
2. Retrieved chunks are re-ranked by `CrossEncoderReranker` (injected into `HybridRetriever` at construction time) before being stored in state.
3. The node stores the final chunks in `state["retrieved_chunks"]` and increments `iteration_count` by 1.
4. `RetrieverException` or `RerankerException` is caught at the node boundary; the node logs the error and returns `retrieved_chunks: []` (fail-open so the graph continues).
5. If the session has no indexed documents, the node returns `retrieved_chunks: []` gracefully.
6. `HybridRetriever` is constructed lazily via a module-level sentinel so it is easy to replace in tests.
7. Unit tests in `tests/unit/core/agent/test_researcher_node.py` cover success, empty-session, retriever-failure, reranker-failure, and iteration-count paths.
8. All pre-existing tests continue to pass (no regressions).

## Design Decisions

- **Fail-open pattern**: on any retrieval or re-ranking error, the node returns `retrieved_chunks: []` so the reporter node's empty-context fallback (Story 3.2) can produce a graceful answer.
- **DI via keyword-only arg**: `researcher_node(state, *, _retriever=None)` — LangGraph only passes `state` positionally, so the `_retriever` kwarg is safe in production and convenient in tests.
- **Re-ranking is internal to `HybridRetriever`**: `CrossEncoderReranker` is injected into the retriever at construction; the node does not call the reranker separately.
- **Shared `SessionIndexer` singleton**: the default retriever factory calls `get_default_session_indexer()` so the agent node and the ingestion pipeline share the same in-memory session registry.

## Implementation Notes

- `_get_default_retriever()` is a lazy module-level factory (sentinel pattern matching `_compiled_graph` in `graph.py`).
- All lazy imports inside functions are annotated with `# noqa: PLC0415` (established convention).
- `iteration_count` uses an `operator.add` reducer — the node always returns `1` (not read-modify-write).
- `core/rag/indexer.py` gained `get_default_session_indexer()` singleton factory and re-exported it from `core/rag/__init__.py`.

## Files Changed

| File | Action |
|---|---|
| `core/agent/nodes.py` | MODIFIED — replaced `researcher_node` stub; added `_default_retriever` sentinel and `_get_default_retriever()` factory |
| `core/rag/indexer.py` | MODIFIED — added `get_default_session_indexer()` singleton factory |
| `core/rag/__init__.py` | MODIFIED — re-exported `get_default_session_indexer` |
| `tests/unit/core/agent/test_researcher_node.py` | CREATED — 6 unit tests for `researcher_node` |

## Out of Scope

- MCP web search tool integration (Story 3.4)
- Reviewer LLM quality gate (Story 3.5/3.6)
- Streaming Groq responses
- FastAPI upload endpoint (Epic 4)
- Streamlit UI (Epic 4)
- Session persistence to disk (Epic 5)
- Thread-safe `SessionIndexer` construction

## Definition of Done

- `researcher_node` in `nodes.py` calls `HybridRetriever.search()` (not the empty stub).
- All 6 new unit tests in `tests/unit/core/agent/test_researcher_node.py` pass.
- Full test suite passes with no regressions (**148/148** tests).
- Story status updated to `review`.

## Completion Notes

- Replaced `researcher_node` stub with real `HybridRetriever` integration wired with `CrossEncoderReranker`.
- Added `_get_default_retriever()` lazy factory using module-level sentinel pattern.
- Added shared `SessionIndexer` singleton so the agent node and ingestion pipeline operate on the same in-memory registry.
- Catches both `RetrieverException` and `RerankerException` at the node boundary — consistent fail-open pattern with `reporter_node`.
- 6 unit tests: success, empty-session, retriever-failure, reranker-failure, iteration-count increment, and default retriever wiring.
- Full regression: **148/148 passed** (0 regressions from 142 pre-existing tests).
