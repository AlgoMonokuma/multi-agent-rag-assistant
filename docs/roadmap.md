# Project Roadmap & Release Strategy

## Versioning Plan

| Version | Theme | Release When |
| --- | --- | --- |
| `v0.1.0` | Project foundation | Setup, config, logging, initial API/UI bootstrap are stable. |
| `v0.2.0` | RAG runtime foundation | Parser, indexing, chunking, hybrid retrieval, profiles, and re-ranking are documented and tested. |
| `v0.2.1` | RAG runtime hardening | Story 2.5.1 guardrails are complete and tests pass. |
| `v0.2.2` | Codebase language normalization | Engineering Task 2.5.2 is complete; developer-facing comments, docstrings, logs, exceptions, and tests now use consistent English. |
| `v0.3.0` | Agent workflow prototype | LangGraph state graph and first agent workflow are usable. |
| `v0.4.0` | Streaming user experience | FastAPI SSE and Streamlit trace UI are usable. |
| `v0.5.0` | Deployable demo | Docker, CI, and hosting path are ready. |
| `v1.0.0` | Interview demo release | End-to-end demo is stable, documented, and easy to run. |

## Future Epic Direction

### Epic 3: Agent Workflow Prototype

Goal: Build the first multi-agent reasoning workflow on top of the hardened RAG runtime using LangGraph.

Planned direction:

- LangGraph state graph foundation for cyclic reasoning and state persistence.
- Observability and tracing setup (LangSmith or Arize Phoenix) at the start of agent development to monitor reasoning paths.
- Researcher agent that can query retrieved context and identify information gaps.
- Reporter agent that can produce grounded answers with explicit citation mapping.
- Web search tool integration through MCP when local context is insufficient.
- Error recovery and state rollback for multi-agent workflow failures.
- Quality gate agent for answer validation against user intent.

### Epic 4: Streaming User Experience

Goal: Make the system usable as an interactive demo with transparent reasoning and citation-aware answers.

Planned direction:

- Streamlit visual theme and application shell.
- FastAPI server-sent events for streaming progress.
- Reasoning trace UI for agent steps.
- Markdown answer rendering with citations.
- User-friendly error and fallback messaging for recoverable runtime failures.

### Epic 5: Deployable Demo

Goal: Prepare the project for interview review and hosted demonstration.

Planned direction:

- Dockerized runtime suitable for Hugging Face Spaces or similar hosting.
- RAG evaluation benchmark using RAGAS or equivalent tooling to quantify Hit Rate, Faithfulness, and Answer Relevance.
- GitHub Actions quality checks.
- Release preparation, documentation cleanup, and demo readiness.
- Versioned GitHub Releases for meaningful milestones.

## Story Status

| Story | Title | Public Status |
| --- | --- | --- |
| [1.1](stories/1-1-project-foundation-initialization.md) | Project foundation initialization | Complete |
| [1.2](stories/1-2-configure-tdd-test-environment.md) | TDD test environment | Complete |
| [1.3](stories/1-3-environment-variable-and-secure-config-loading.md) | Secure config loading | Complete |
| [1.4](stories/1-4-logging-and-engineering-documentation.md) | Shared logging and engineering docs | Complete |
| [2.1](stories/2-1-document-ingestion-and-parser-pipeline.md) | Document ingestion and parser pipeline | Complete |
| [2.2](stories/2-2-session-isolated-indexing-foundation.md) | Session-isolated indexing foundation | Complete |
| [2.3](stories/2-3-text-chunking-and-embedding-pipeline.md) | Text chunking and embedding pipeline | Complete |
| [2.4](stories/2-4-hybrid-search-implementation.md) | Hybrid search implementation | Complete |
| [2.4.5](stories/2-4-5-document-type-chunking-profile.md) | Document-type chunking profile | Complete |
| [2.5](stories/2-5-re-ranking-mechanism.md) | Re-ranking mechanism | Complete |
| [2.5.1](stories/2-5-1-rag-runtime-hardening.md) | Runtime hardening and harness guardrails | Complete |
| [2.5.2](stories/2-5-2-codebase-language-normalization.md) | Engineering task: codebase language normalization | Complete |

## Known Scope Gaps & Backlog

- [ ] [Issue #1](https://github.com/AlgoMonokuma/multi-agent-rag-assistant/issues/1): PDF image and table support (scope gap identified after Story 2.1, targeting v0.3.0 or later)
