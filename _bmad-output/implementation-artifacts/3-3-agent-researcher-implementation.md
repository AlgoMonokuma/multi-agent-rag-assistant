# Story 3.3: Agent Researcher Implementation

Status: review

## Story

As a user,
I want the agent to retrieve relevant document chunks when I ask a question,
so that the reporter node can generate a grounded answer using real content rather than an empty context.

## Acceptance Criteria

1. `researcher_node` in `core/agent/nodes.py` calls `HybridRetriever.search()` using `session_id` and `query` from `AgentState`.
2. The retrieved `RetrievedChunk` list is passed through `CrossEncoderReranker.rerank()` before being stored in state.
3. The node stores the final re-ranked chunks in `state["retrieved_chunks"]` and increments `iteration_count` by 1.
4. A `RetrieverException` or `RerankerException` is caught at the node boundary; the node logs the error and returns `retrieved_chunks: []` (fail-open so the graph continues).
5. If the session has no indexed documents, the node returns `retrieved_chunks: []` gracefully (HybridRetriever already handles this natively).
6. `HybridRetriever` and `CrossEncoderReranker` are constructed lazily / via DI so they are easy to replace in tests.
7. Unit tests in `tests/unit/core/agent/test_researcher_node.py` mock both `HybridRetriever` and `CrossEncoderReranker` and cover success, empty-session, retriever-failure, and reranker-failure paths.
8. All 129 existing tests continue to pass (no regressions).

---

## Tasks / Subtasks

- [x] Task 1: Replace `researcher_node` stub in `core/agent/nodes.py` (AC: 1, 2, 3, 4, 5, 6)
  - [x] 1.1 Import `HybridSearchResult`, `HybridRetriever`, `RetrieverException` via lazy import inside node function
  - [x] 1.2 Import `CrossEncoderReranker`, `RerankerException` via lazy import inside node function
  - [x] 1.3 Construct `SentenceTransformerEmbedder` and `SessionIndexer` singletons or accept them as defaults via a factory function for DI
  - [x] 1.4 Call `retriever.search(session_id=..., query=..., top_k=10)` and extract `results`
  - [x] 1.5 Call `reranker.rerank(query=..., chunks=results, top_n=5)` on successful retrieval
  - [x] 1.6 Catch `RetrieverException` or `RerankerException` — log error, return `retrieved_chunks: []`
  - [x] 1.7 Return `{"retrieved_chunks": reranked_chunks, "iteration_count": 1}`
- [x] Task 2: Create `tests/unit/core/agent/test_researcher_node.py` (AC: 7)
  - [x] 2.1 `test_researcher_node_success` — mock retriever + reranker return chunks; assert state updated
  - [x] 2.2 `test_researcher_node_empty_session` — mock retriever returns `[]`; assert `retrieved_chunks == []`
  - [x] 2.3 `test_researcher_node_retriever_failure` — mock retriever raises `RetrieverException`; assert `retrieved_chunks == []`
  - [x] 2.4 `test_researcher_node_reranker_failure` — mock reranker raises `RerankerException`; assert `retrieved_chunks == []`
  - [x] 2.5 `test_researcher_node_increments_iteration_count` — assert `iteration_count == 1` in return dict
- [x] Task 3: Run full test suite to confirm no regressions (AC: 8)
  - [x] `.\\.venv\\Scripts\\python.exe -m pytest tests/`

---

## Dev Notes

### Where This File Lives

```
core/
  agent/
    nodes.py      ← MODIFY: replace researcher_node stub with real implementation
tests/
  unit/
    core/
      agent/
        test_researcher_node.py   ← NEW: researcher node unit tests
```

> **Architecture rule**: Business logic belongs in `core/rag/`. The agent node is a thin orchestrator that calls into `core/rag/` — it does NOT own retrieval logic. This mirrors how `reporter_node` calls `LLMGenerator` without owning generation logic.

### Key Interfaces

**`HybridRetriever.search()` signature** (from `core/rag/retriever.py`):
```python
def search(
    self,
    session_id: str,
    query: str,
    top_k: int = DEFAULT_TOP_K,        # DEFAULT_TOP_K = 10
    vector_weight: float | None = None,
    keyword_weight: float | None = None,
) -> HybridSearchResult:
```

**`HybridSearchResult`** (from `core/rag/retriever.py`):
```python
@dataclass(slots=True)
class HybridSearchResult:
    query: str
    session_id: str
    results: List[RetrievedChunk]   # ← use this for reranking
    total_found: int
```

**`RetrievedChunk`** (from `core/rag/retriever.py`):
```python
@dataclass
class RetrievedChunk:
    chunk_id: str
    page_content: str
    metadata: Dict[str, Any]          # contains "source", "chunk_id", "session_id"
    vector_score: float
    keyword_score: float
    merged_score: float
    rank: int
    rerank_score: Optional[float] = None
```

**`CrossEncoderReranker.rerank()` signature** (from `core/rag/reranker.py`):
```python
def rerank(
    self,
    query: str,
    chunks: Sequence[object],
    top_n: int = 3,
) -> list[object]:                    # returns list[RetrievedChunk] at runtime
```

**`HybridRetriever.__init__()` required arguments**:
```python
def __init__(
    self,
    session_indexer: Any,   # SessionIndexer instance
    embedder: Any,          # SentenceTransformerEmbedder instance
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,    # 0.7
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,  # 0.3
    reranker: Optional["CrossEncoderReranker"] = None,
    final_top_n: int = 3,
) -> None:
```

### Recommended Implementation Pattern

Use a **factory function** to bundle the heavy dependency construction. This keeps the node signature clean and makes test injection straightforward:

```python
# core/agent/nodes.py

def _build_retriever():
    """Construct the default HybridRetriever stack. Lazy-loaded on first call."""
    from core.rag.embeddings import SentenceTransformerEmbedder
    from core.rag.indexer import SessionIndexer
    from core.rag.reranker import CrossEncoderReranker
    from core.rag.retriever import HybridRetriever

    embedder = SentenceTransformerEmbedder()
    indexer = SessionIndexer()
    reranker = CrossEncoderReranker()
    return HybridRetriever(
        session_indexer=indexer,
        embedder=embedder,
        reranker=reranker,
        final_top_n=5,
    )
```

> **CAUTION**: Do **NOT** make `_build_retriever()` a module-level singleton call. If called at import time, it will load `sentence-transformers` and `faiss` eagerly — this will slow down all tests and break the lazy-load guarantee. Call the factory inside the node function, or use a module-level `None` sentinel with lazy initialization (same pattern as `_compiled_graph` in `graph.py`).

### Preferred DI Pattern for `researcher_node`

The cleanest approach for test injection is a **default-argument factory**. This is more Pythonic than a module-level sentinel for a function that doesn't need to be a singleton:

```python
def researcher_node(state: AgentState, *, _retriever=None) -> dict:
    """Retrieve relevant chunks for the user query.

    Calls HybridRetriever (with CrossEncoderReranker) using session_id and
    query from state. Fails open (returns retrieved_chunks=[]) on any error
    so the graph continues to the reporter node.

    Args:
        state: Current workflow state containing query and session_id.
        _retriever: Optional pre-built HybridRetriever for test injection.
                    When None, the default retrieval stack is constructed.

    Returns:
        Partial state update with retrieved_chunks (list[RetrievedChunk])
        and an iteration_count increment of 1.
    """
```

> **ISSUE**: LangGraph invokes node functions with a single positional arg `(state,)`, so keyword-only injection via `_retriever=None` will work correctly with `graph.invoke()` — LangGraph never passes extra kwargs to nodes.

### Full researcher_node Implementation

```python
def researcher_node(state: AgentState, *, _retriever=None) -> dict:
    """Retrieve relevant chunks for the user query."""
    from core.rag.reranker import RerankerException  # noqa: PLC0415
    from core.rag.retriever import RetrieverException  # noqa: PLC0415

    query = state.get("query", "")
    session_id = state.get("session_id", "")

    logger.debug(
        "researcher_node: query=%r session_id=%r",
        query,
        session_id,
    )

    retriever = _retriever or _get_default_retriever()

    try:
        search_result = retriever.search(session_id=session_id, query=query, top_k=10)
        chunks = search_result.results
    except RetrieverException as exc:
        logger.error("researcher_node: retrieval failed: %s", exc)
        return {"retrieved_chunks": [], "iteration_count": 1}

    logger.debug(
        "researcher_node: retrieved %d chunks for session %r",
        len(chunks),
        session_id,
    )

    return {"retrieved_chunks": list(chunks), "iteration_count": 1}
```

> **Note on re-ranking placement**: The `HybridRetriever` already accepts an optional `reranker` in its constructor and calls `reranker.rerank()` internally (see `retriever.py` lines 167–181). This means **you do NOT need to call the reranker separately** in `researcher_node` — it is handled inside `HybridRetriever.search()` when a reranker is injected at construction time. The `_build_retriever()` factory should pass `reranker=CrossEncoderReranker()` to the `HybridRetriever` constructor; doing so wires re-ranking automatically.

### Module-Level Lazy Default Pattern

```python
# Module-level sentinel (after logger line)
_default_retriever = None


def _get_default_retriever():
    """Return the module-level default retriever, constructing it lazily."""
    global _default_retriever
    if _default_retriever is None:
        from core.rag.embeddings import SentenceTransformerEmbedder  # noqa: PLC0415
        from core.rag.indexer import SessionIndexer  # noqa: PLC0415
        from core.rag.reranker import CrossEncoderReranker  # noqa: PLC0415
        from core.rag.retriever import HybridRetriever  # noqa: PLC0415

        embedder = SentenceTransformerEmbedder()
        indexer = SessionIndexer()
        reranker = CrossEncoderReranker()
        _default_retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=embedder,
            reranker=reranker,
            final_top_n=5,
        )
    return _default_retriever
```

> **IMPORTANT**: Tests must reset `_default_retriever = None` (or bypass it entirely using the `_retriever=` kwarg) so each test has a clean state. The `_retriever=` kwarg approach completely avoids touching the module-level sentinel and is the cleanest option for unit tests.

### Exception Handling Rules

Follow the established pattern from `reporter_node`:

```python
except RetrieverException as exc:
    logger.error("researcher_node: retrieval failed: %s", exc)
    return {"retrieved_chunks": [], "iteration_count": 1}
```

- **Fail open**: return `retrieved_chunks: []` so the graph continues to `reporter_node`. The reporter has its own empty-context fallback (from Story 3.2) that will gracefully produce a "no information found" answer.
- **Always return `iteration_count: 1`**: Even on failure, the cycle counter must increment to prevent infinite loops.
- **Log the exception, do not re-raise**: Consistent with `reporter_node`'s error handling pattern.

### SessionIndexer Lifetime Note

`SessionIndexer` is a stateful in-memory store — each new instance starts with zero sessions. In production, the `SessionIndexer` singleton is shared across the whole app (session state would be lost otherwise). For Story 3.3's unit tests, this does **not matter** because we mock `HybridRetriever` directly; the indexer is never touched.

In a later integration story, the `SessionIndexer` singleton will need to be injected at the application boundary (e.g., FastAPI startup event). For now, the `_get_default_retriever()` factory creates a **new** `SessionIndexer` on first call — this is acceptable for the current dev-mode graph tests because `test_graph.py` does not use real chunks and `researcher_node` now returns `[]` from an empty index.

> Specifically: calling `retriever.search()` with an empty `SessionIndexer` (no sessions) will raise `RetrieverException("Session not found: ...")`, which the fail-open handler catches and converts to `retrieved_chunks: []`. This keeps the graph operational.

---

## Testing Requirements

File: `tests/unit/core/agent/test_researcher_node.py`

| Test name | AC | What it proves |
|---|---|---|
| `test_researcher_node_success` | 1, 2, 3 | Mock retriever returns chunks; assert `retrieved_chunks` matches and `iteration_count == 1` |
| `test_researcher_node_empty_session` | 5 | Mock retriever returns empty `results`; assert `retrieved_chunks == []` |
| `test_researcher_node_retriever_failure` | 4 | Mock raises `RetrieverException`; assert `retrieved_chunks == []` and `iteration_count == 1` |
| `test_researcher_node_reranker_failure` | 4 | Mock raises `RerankerException` from inside retriever; assert fail-open |
| `test_researcher_node_increments_iteration_count` | 3 | Assert return dict always contains `iteration_count == 1` |

### Test Pattern

```python
# tests/unit/core/agent/test_researcher_node.py
from unittest.mock import MagicMock

import pytest

from core.agent.nodes import researcher_node
from core.agent.state import AgentState
from core.rag.retriever import HybridSearchResult, RetrievedChunk, RetrieverException
from core.rag.reranker import RerankerException


def _make_chunk(chunk_id: str = "chunk-1", source: str = "doc.pdf") -> RetrievedChunk:
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
    return HybridSearchResult(
        query="What is RAG?",
        session_id="s1",
        results=chunks,
        total_found=len(chunks),
    )


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

    # RerankerException not currently caught — test drives proper exception handling
    # The implementation must catch RerankerException too (or wrap inside retriever)
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
```

> **RerankerException catch note**: `HybridRetriever.search()` already catches `RerankerException` internally (see `retriever.py:176`) and logs then continues without re-raising. However, failing tests with `RerankerException` as a `side_effect` on the mocked retriever's `.search()` call verifies that the node's own `except` clause catches it at the node boundary too. The node's `except` block must include `RerankerException` alongside `RetrieverException`.

### Exception Catch Block (exact implementation target)

```python
try:
    search_result = retriever.search(session_id=session_id, query=query, top_k=10)
    chunks = search_result.results
except (RetrieverException, RerankerException) as exc:
    logger.error("researcher_node: retrieval failed: %s", exc)
    return {"retrieved_chunks": [], "iteration_count": 1}
```

---

## Previous Story Intelligence

From **Story 3.2 (done — review status)**:
- `reporter_node` in `nodes.py` uses a lazy import pattern: `from core.rag.generator import GeneratorException, LLMGenerator  # lazy import`. Use the **identical** lazy import pattern for retriever/reranker imports in `researcher_node`.
- `reporter_node` uses `state.get("retrieved_chunks") or []` — note that `retrieved_chunks` will now be populated by the real `researcher_node`. After Story 3.3, the full pipeline becomes: researcher fills chunks → reporter reads chunks → LLMGenerator produces answer.
- `reporter_node` stores citations as plain dicts for JSON-compatibility. `researcher_node` stores `RetrievedChunk` dataclasses — this is correct; `reporter_node` consumes them and converts to dicts.
- The `LLMGenerator` constructor uses `client=None` for DI. Use `_retriever=None` keyword-only arg for the same effect.
- All 129 tests passed after Story 3.2. Do not break them.

From **Story 3.1 (done)**:
- `researcher_node` stub currently returns `{"retrieved_chunks": [], "iteration_count": 1}`. This exact dict structure must be preserved in the real implementation.
- `iteration_count` uses `operator.add` reducer — returning `1` from the node adds 1 to the total (do NOT read-modify-write the full count).
- `test_graph.py` test `test_iteration_count_increments_per_cycle` asserts exactly `1` after one cycle — this test will still pass if:
  - `researcher_node` returns `iteration_count: 1` (returning empty chunks is fine)
  - `reporter_node` reads empty chunks and returns the fallback answer
  - `reviewer_node` stub approves (returns `review_passed: True`)
- The `test_graph_invoke_synchronous_success` test calls `graph.invoke()` which will now trigger the real `researcher_node`. Because `_get_default_retriever()` creates a new `SessionIndexer` with no sessions, `retriever.search()` will raise `RetrieverException`("Session not found"), which the fail-open handler catches. The test asserts `result is not None` and `isinstance(result, dict)` — both will still pass.

From **Story 2.5.1 (done)**:
- Domain exceptions must log useful context before failing open.
- `HybridRetriever._validate_session()` raises `RetrieverException("Session not found: {session_id}")` when the session does not exist in `SessionIndexer`. **This is the expected behavior** for the empty-index case in tests.
- `HybridRetriever` handles the empty-index case within a session gracefully (returns empty results without error) — see `retriever.py:144-151`.

---

## Git Intelligence

Recent commit context:
- `feat(rag): implement Story 3.1.2 plain-text parser support` — parser now supports `.txt`; `parser.py` exports `TextFileParser`
- `docs: finalize story 3.1.2 status in roadmap and index` — docs only, no code impact
- `docs: mark Story 3.2 complete` — `generator.py` fully implemented with 129 passing tests
- Branch naming convention: `feature/story-3.3-researcher-node` (following `feature/story-3.2-llm-generator` pattern)
- Existing files NOT to modify:
  - `core/agent/graph.py` — wiring is complete
  - `core/agent/state.py` — schema is sufficient
  - `core/rag/retriever.py` — `HybridRetriever` used as-is
  - `core/rag/reranker.py` — `CrossEncoderReranker` used as-is
  - Any existing test files (only add new ones)

---

## Project Context Reference

- **Logger**: `from core.log import get_logger; logger = get_logger(__name__)` — already initialized in `nodes.py`
- **Lazy import `# noqa` tag**: Use `# noqa: PLC0415` on all lazy imports inside functions (established convention in `generator.py` and `retriever.py`)
- **No `print()` calls**: Use `logger.debug/info/error` exclusively
- **All docstrings in English** (Story 2.5.2 requirement)
- **Completion note format**: Write brief notes in English in the "Dev Agent Record" section at story completion

---

## Files To Create / Modify

| File | Action |
|---|---|
| `core/agent/nodes.py` | **MODIFY** — replace `researcher_node` stub with real `HybridRetriever` call; add `_get_default_retriever()` factory |
| `tests/unit/core/agent/test_researcher_node.py` | **CREATE** — 5 unit tests with mocked HybridRetriever |

## Files NOT To Touch

- `core/agent/graph.py` — graph wiring is done; no changes needed
- `core/agent/state.py` — schema is sufficient; `retrieved_chunks: list` already exists
- `core/rag/retriever.py` — `HybridRetriever` is used, not modified
- `core/rag/reranker.py` — `CrossEncoderReranker` is used, not modified
- `core/rag/embeddings.py` — `SentenceTransformerEmbedder` is used, not modified
- `core/rag/indexer.py` — `SessionIndexer` is used, not modified
- `pyproject.toml` — all dependencies already declared (groq, sentence-transformers, faiss-cpu, langgraph, jieba)
- `tests/unit/core/agent/test_graph.py` — must keep passing; verify after implementation

---

## Definition of Done

- All 5 new unit tests in `tests/unit/core/agent/test_researcher_node.py` pass.
- `researcher_node` in `nodes.py` calls `HybridRetriever.search()` (not the empty stub).
- Running `.\\.venv\\Scripts\\python.exe -m pytest tests/` returns green (no regressions from the 129 existing tests).
- Story status updated to `review`.

---

## Out of Scope

- MCP web search tool integration (Story 3.4)
- Reviewer LLM scoring (Story 3.5/3.6)
- Streaming Groq responses
- FastAPI upload endpoint (Epic 4)
- Streamlit UI (Epic 4)
- Session persistence to disk (Epic 5)
- Thread-safe `SessionIndexer` construction (deferred — noted in architecture for future hardening)

---

## Change Log

- 2026-06-28: Story file created. Status set to ready-for-dev.
- 2026-06-28: Implementation complete. All 6 new unit tests pass. Full suite: 148/148 passed. Status set to review.
- 2026-06-28: Patch — default retriever now uses shared `get_default_session_indexer()` singleton instead of private `SessionIndexer()`; added wiring test. Final test count 142 (base) + 6 (new) = 148 total. Files touched: `core/rag/indexer.py`, `core/rag/__init__.py`, `core/agent/nodes.py`, `tests/unit/core/agent/test_researcher_node.py`.

---

## Dev Agent Record

### Agent Model Used

Claude Sonnet 4.6 (Thinking)

### Debug Log References

_None_

### Completion Notes List

- Replaced `researcher_node` stub with real `HybridRetriever` integration.
- Added `_get_default_retriever()` lazy factory using module-level sentinel pattern (mirrors `_compiled_graph` in `graph.py`).
- `researcher_node` accepts keyword-only `_retriever=None` arg for DI; LangGraph passes only `state` positionally so this is safe in production.
- Re-ranking is handled inside `HybridRetriever.search()` (injected via `reranker=CrossEncoderReranker()` at construction); no separate reranker call needed in the node.
- Catches both `RetrieverException` and `RerankerException` at node boundary — fail-open pattern mirrors `reporter_node`.
- 6 new unit tests in `tests/unit/core/agent/test_researcher_node.py`: success, empty-session, retriever-failure, reranker-failure, iteration-count, and factory wiring.
- Full regression: **148/148 passed** (0 regressions from 142 pre-existing tests).

**Patch (post-review):**
- `_get_default_retriever()` now uses `get_default_session_indexer()` shared singleton (from `core/rag/indexer.py`) instead of constructing a private `SessionIndexer()`. This ensures the default graph path and ingestion entry points operate on the same in-memory session registry.
- Added `get_default_session_indexer()` factory in `core/rag/indexer.py` with module-level sentinel `_default_session_indexer`.
- Re-exported `get_default_session_indexer` from `core/rag/__init__.py` for app/ingestion boundary wiring.
- Added 1 wiring test (`test_get_default_retriever_wires_shared_indexer_and_reranker`) verifying that `HybridRetriever` is constructed with the shared indexer + reranker, and that the sentinel caches the instance on second call.
- Test count 5→6; full regression **148/148 passed**.

### File List

Initial implementation:
- `core/agent/nodes.py` — MODIFIED (replaced researcher_node stub; added _default_retriever sentinel and _get_default_retriever factory)
- `tests/unit/core/agent/test_researcher_node.py` — CREATED (5 unit tests for researcher_node)

Post-review patch:
- `core/agent/nodes.py` — MODIFIED (factory now calls `get_default_session_indexer()` instead of `SessionIndexer()`)
- `core/rag/indexer.py` — MODIFIED (added `get_default_session_indexer()` singleton factory + `_default_session_indexer` sentinel)
- `core/rag/__init__.py` — MODIFIED (re-exported `get_default_session_indexer` in import + `__all__`)
- `tests/unit/core/agent/test_researcher_node.py` — MODIFIED (added `test_get_default_retriever_wires_shared_indexer_and_reranker` wiring test; total 6 tests)
