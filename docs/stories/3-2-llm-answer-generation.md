# Story 3.2: LLM Answer Generation

Status: Complete

## User Story

As a user, I want the system to generate grounded answers from retrieved document chunks so that I can receive a usable response with citation references instead of a placeholder draft.

## Scope

This story adds the first end-to-end answer generation capability on top of the existing RAG runtime and LangGraph agent workflow foundation. It replaces the placeholder reporter output with a Groq-backed generator that assembles retrieved context, enforces a prompt budget, returns citation metadata, and handles recoverable generation failures consistently.

## Acceptance Criteria

1. Given a query, retrieved chunks, and a session ID, when answer generation runs, then `LLMGenerator` accepts those inputs through a clear public API.
2. Given retrieved chunks, when the prompt is assembled, then included context contains source and chunk ID metadata within the configured token budget.
3. Given generation is enabled, when the Groq client is loaded, then it uses the configured `GROQ_API_KEY`.
4. Given a successful Groq response, when generation completes, then the result includes answer text and citation references.
5. Given API, configuration, import, or malformed response failures, when generation fails, then `GeneratorException` is raised from the generator boundary.
6. Given no relevant context is available, when generation runs, then a graceful fallback answer is returned without making an API call.
7. Given the generator unit tests run, then success, fallback, prompt budgeting, citation accuracy, and failure paths are covered with mocked Groq clients.

## Implementation Notes

- `core/rag/generator.py` defines `LLMGenerator`, `GeneratorException`, `CitationRef`, and `GeneratorResponse`.
- The generator lazily loads the Groq client and reads the API key through `settings.require_groq_api_key()`.
- Prompt assembly includes `[Source: ... | Chunk ID: ...]` headers for admitted chunks.
- Citation references are derived only from chunks that fit within the prompt budget.
- Empty context and oversized first-chunk scenarios return the same no-context fallback without calling the API.
- `core/agent/nodes.py` wires `reporter_node` to call `LLMGenerator` and store citations as JSON-compatible dictionaries.

## Out of Scope

- Real researcher-node retrieval integration
- Reviewer LLM or quality gate behavior
- Streaming answer tokens to the UI
- Web search fallback when local context is insufficient
- Production observability or tracing

## Definition of Done

- `core/rag/generator.py` implements the Groq-backed generator boundary.
- `core/agent/nodes.py` calls the generator from `reporter_node`.
- `pyproject.toml` includes the Groq dependency.
- `tests/unit/core/rag/test_generator.py` covers success, fallback, prompt budgeting, citation behavior, API/config failures, and malformed responses.
- Public documentation and story indexes mark Story 3.2 as complete.

## Completion Notes

- Added Groq-backed LLM generation for the reporter node.
- Added citation-aware prompt assembly and token-budget handling.
- Added graceful fallback behavior for empty or unusable context.
- Added generator-specific exception handling for API, config, import, and malformed response failures.
- Added mocked unit coverage for the generator behavior and failure paths.
