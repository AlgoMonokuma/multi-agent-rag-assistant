# Project Roadmap & Release Strategy

## Versioning Plan

| Version | Theme | Release When |
| --- | --- | --- |
| `v0.1.0` | Project foundation | Setup, config, logging, initial API/UI bootstrap are stable. |
| `v0.2.0` | RAG runtime foundation | Parser, indexing, chunking, hybrid retrieval, profiles, and re-ranking are documented and tested. |
| `v0.2.1` | RAG runtime hardening | Story 2.5.1 guardrails are complete and tests pass. |
| `v0.2.2` | Codebase language normalization | Engineering Task 2.5.2 is complete; all comments, docstrings, and tests use consistent English. |
| `v0.3.0` | Complete RAG Agent prototype | Stories 3.1 (LangGraph foundation), 3.1.1 (multilingual + CJK), 3.1.2 (TXT parser), 3.2 (LLM generation), and 3.3 (Researcher Agent) are complete. The system can complete a full retrieval → research → answer loop end-to-end. |
| `v0.4.0` | Advanced workflow + streaming UI | Story 3.5 (Reviewer quality gate), Stories 4.1–4.4 (Streamlit UI, SSE streaming, citation rendering) are usable. |
| `v0.5.0` | External search + deployable demo | Story 3.4 (MCP web search), Docker, CI, session persistence (Story 5.1), and hosting path are ready. |
| `v1.0.0` | Interview demo release | End-to-end demo is stable, documented, and easy to run. |

## Future Epic Direction

### Epic 3: Agent Workflow Prototype

Goal: Build a complete multi-agent reasoning workflow on top of the hardened RAG runtime using LangGraph. Resolves foundational multilingual gaps and delivers a full retrieval → research → answer loop.

**v0.3.0 scope (complete RAG Agent loop):**

- **Story 3.1** ✅: LangGraph state graph foundation — graph defines researcher, reporter, and reviewer stub nodes with conditional retry routing and a max-iteration safety ceiling.
- **Story 3.1.1** ✅: Multilingual embedding model migration and CJK-aware BM25 tokenization. Default embedder uses `paraphrase-multilingual-MiniLM-L12-v2`; keyword retrieval uses `jieba` for Han text.
- **Story 3.1.2**: Plain-text (`.txt`) parser — allows users to upload plain documents in addition to PDF and Markdown.
- **Story 3.2** ✅: LLM answer generation using Groq API — prompt assembly, context budget management, citation mapping, and graceful API failure handling.
- **Story 3.3**: Researcher Agent — queries the RAG retriever, identifies information gaps, and passes grounded context to the reporter node. This completes the first real end-to-end agent loop.

**Post v0.3.0 (planned for v0.4.0 / v0.5.0):**

- **Story 3.4**: MCP web search tool integration — when local context is insufficient, the researcher can invoke an external search tool.
- **Story 3.5**: Reviewer quality gate — a dedicated agent validates the reporter's answer against user intent before final output.

### Epic 4: Streaming User Experience

Goal: Make the system usable as an interactive demo with transparent reasoning and citation-aware answers.

Planned direction:

- Streamlit visual theme and application shell.
- **Story 4.2**: File upload API endpoint (`POST /upload`) with extension whitelist, magic-byte validation, per-file 100 MB size limit, batch limit of 10 files, temp-file cleanup, and FAISS write serialization.
- FastAPI server-sent events for streaming progress.
- Reasoning trace UI for agent steps.
- Markdown answer rendering with citations.
- User-friendly error and fallback messaging for recoverable runtime failures.

### Epic 5: Deployable Demo

Goal: Prepare the project for interview review and hosted demonstration. Includes session persistence so users can resume their work without re-uploading documents.

Planned direction:

- **Story 5.1**: Session persistence — disk-backed FAISS index, browser `localStorage` session ID, and server-side session reload on restart. No login system required.
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
| [3.1](stories/3-1-langgraph-state-graph-foundation.md) | LangGraph state graph foundation | Complete |
| [3.1.1](stories/3-1-1-multilingual-embedding-and-cjk-tokenization.md) | Multilingual embedding and CJK tokenization | Complete |
| [3.1.2](stories/3-1-2-plain-text-parser-support.md) | Plain-text parser support | Complete |
| [3.2](stories/3-2-llm-answer-generation.md) | LLM answer generation | Complete |

## Known Scope Gaps & Backlog

- [ ] [Issue #1](https://github.com/AlgoMonokuma/multi-agent-rag-assistant/issues/1): PDF image and table support (scope gap identified after Story 2.1, targeting v0.3.0 or later)
- [x] LangGraph state graph foundation with stub researcher/reporter/reviewer nodes (Story 3.1)
- [x] Embedding model migration: `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2` (Story 3.1.1)
- [x] CJK-aware BM25 tokenization using jieba (Story 3.1.1)
- [x] LLM answer generation end-to-end with Groq-backed grounded answers and citation mapping (Story 3.2)
- [ ] File upload API with security validation — magic-byte check, size limit, batch limit, temp-file cleanup (Story 4.2)
- [ ] FAISS write serialization — asyncio.Lock to prevent race condition under concurrent ingestion (Story 4.2)
- [ ] Session persistence — disk-backed index + browser localStorage session ID (Story 5.1)
- [x] Plain-text (.txt) parser (Story 3.1.2)
