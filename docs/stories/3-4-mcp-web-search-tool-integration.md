# Story 3.4: MCP Web Search Tool Integration

Status: Complete

## User Story

As a user, I want the researcher agent to fall back to live web search when local documents do not contain enough relevant context, so that I can get a grounded answer even when the uploaded documents are insufficient.

## Scope

This story adds a `web_search_node` to the LangGraph agent workflow that performs live web search via Tavily when the researcher node retrieves too few relevant chunks. The graph is extended with conditional routing and the reporter node merges web results alongside local chunks in the LLM prompt.

## Acceptance Criteria

1. A `web_search_node` is added to `core/agent/nodes.py` that calls a Tavily web search client using the `query` from `AgentState`.
2. The node stores results in `state["web_search_results"]` as a list of plain dicts with at least `url`, `content`, and `score` keys.
3. `researcher_node` detects low-context conditions (fewer than `WEB_SEARCH_THRESHOLD` chunks, default `2`) and sets `state["needs_web_search"] = True`.
4. The LangGraph graph routes through `web_search_node` when `needs_web_search` is `True`, before proceeding to `reporter_node`.
5. `reporter_node` merges `web_search_results` alongside `retrieved_chunks` when assembling the LLM prompt context.
6. `web_search_node` fails open on any `WebSearchException` or network error — logs and returns `web_search_results: []`.
7. All Tavily client construction is lazy — the client is not loaded at import time.
8. `TAVILY_API_KEY` is loaded via `core/config.py` settings; missing key raises `WebSearchException` with a clear message.
9. Unit tests cover: success, empty results, missing API key, network failure, threshold detection, and routing logic.
10. All 148+ existing tests continue to pass (no regressions).

## Design Decisions

- **`langchain-tavily` direct client** — NOT a subprocess MCP server. Chosen for simplicity (sync, one dependency, easy to mock). Full MCP can be adopted in a future story.
- **Fail-open pattern**: on any web search error, the node returns an empty results list so the graph continues normally through the reporter node.
- **DI via keyword-only arg**: `web_search_node(state, *, _client=None)` — mirrors `researcher_node(state, *, _retriever=None)`. LangGraph only passes `state` positionally.
- **Lazy singleton**: `_get_default_web_search_client()` constructs `TavilySearchAPIWrapper` on first call using a module-level sentinel.
- **Backward-compatible generator API**: `LLMGenerator.generate()` accepts optional `web_context: list[dict] | None = None` — all existing callers remain unchanged.
- **Conditional graph routing**: `_route_after_researcher(state)` replaces the old direct `researcher → reporter` edge with a conditional that routes through `web_search_node` when chunks are insufficient.

## Implementation Notes

- `WEB_SEARCH_THRESHOLD = 2` is a module-level constant in `nodes.py`.
- `researcher_node` now returns `needs_web_search` on every code path (including the error path).
- Web search results are appended as a `## Web Search Results` section in the LLM prompt, capped at 5 items.
- The list comprehension in `web_search_node` is inside the try block and includes an `isinstance(r, dict)` guard.
- `_route_after_researcher` uses explicit default `state.get("needs_web_search", False)`.
- All lazy imports use `# noqa: PLC0415`. All docstrings in English.

## Files Changed

| File | Action |
|---|---|
| `core/agent/exceptions.py` | CREATED — `WebSearchException` domain exception |
| `core/agent/state.py` | MODIFIED — added `needs_web_search`, `web_search_results` fields |
| `core/agent/nodes.py` | MODIFIED — added `web_search_node`, `WEB_SEARCH_THRESHOLD`, lazy client factory; updated `researcher_node` and `reporter_node` |
| `core/agent/graph.py` | MODIFIED — added `_route_after_researcher`, `web_search` node, conditional routing |
| `core/config.py` | MODIFIED — added `TAVILY_API_KEY` setting |
| `core/rag/generator.py` | MODIFIED — added `web_context` param to `generate()` and `_build_prompt()` |
| `pyproject.toml` | MODIFIED — added `langchain-tavily>=0.1.0` |
| `.env.example` | MODIFIED — added `TAVILY_API_KEY=` |
| `tests/unit/core/agent/test_web_search_node.py` | CREATED — 4 unit tests |
| `tests/unit/core/agent/test_researcher_node.py` | MODIFIED — 2 tests for `needs_web_search` flag |
| `tests/unit/core/agent/test_graph.py` | MODIFIED — 2 routing tests for `_route_after_researcher` |

## Out of Scope

- Full MCP subprocess / `langchain-mcp-adapters` wire protocol (future story)
- Streaming Groq responses (Epic 4)
- Reviewer LLM scoring (Story 3.5)
- FastAPI upload endpoint (Epic 4)
- Streamlit UI (Epic 4)
- Session persistence to disk (Epic 5)
- Rate limiting or caching for web search calls

## Definition of Done

- `web_search_node` callable with injected mock client.
- `researcher_node` returns `needs_web_search` on all code paths.
- `_route_after_researcher` routes correctly based on `needs_web_search`.
- `reporter_node` reads and passes `web_search_results` to `LLMGenerator`.
- `LLMGenerator.generate()` accepts `web_context` optional param (backward-compatible).
- All 8 new tests pass (4 web search + 2 researcher + 2 routing).
- Full test suite: **156/156 passed** (0 regressions).

## Completion Notes

- Implemented `web_search_node` with `TavilySearchAPIWrapper`, lazy client construction, and fail-open error handling.
- Updated `researcher_node` with `WEB_SEARCH_THRESHOLD = 2` threshold detection and `needs_web_search` flag on all return paths.
- Replaced direct `researcher → reporter` edge with conditional routing via `_route_after_researcher`.
- Updated `reporter_node` and `LLMGenerator` to merge web search results into the prompt context.
- 3 defensive fixes from code review: list comprehension inside try block, explicit default in router, `isinstance` guard in prompt builder.
- Dependency `langchain-tavily==0.2.18` added via `uv add`.
- CHANGELOG updated with Story 3.3 and 3.4 entries.
