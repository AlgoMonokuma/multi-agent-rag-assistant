# Project Roadmap & Release Strategy

## Versioning Plan

| Version | Theme | Release When |
| --- | --- | --- |
| `v0.1.0` | Project foundation | Setup, config, logging, initial API/UI bootstrap are stable. |
| `v0.2.0` | RAG runtime foundation | Parser, indexing, chunking, hybrid retrieval, profiles, and re-ranking are documented and tested. |
| `v0.2.1` | RAG runtime hardening | Story 2.5.1 guardrails are complete and tests pass. |
| `v0.2.2` | Codebase language normalization | Engineering Task 2.5.2 is complete; all comments, docstrings, and tests use consistent English. |
| `v0.3.0` | Agent workflow prototype + multilingual foundation | Stories 3.1, 3.1.1 (multilingual model + CJK tokenizer), 3.1.2 (TXT parser), and 3.2 (LLM answer generation) are usable. |
| `v0.4.0` | Streaming user experience + file upload API | Stories 4.1–4.5 (Streamlit UI, upload endpoint with validation, SSE, citation rendering) are usable. |
| `v0.5.0` | Deployable demo with session persistence | Docker, CI, session persistence (Story 5.1), and hosting path are ready. |
| `v1.0.0` | Interview demo release | End-to-end demo is stable, documented, and easy to run. |

## Future Epic Direction

### Epic 3: Agent Workflow Prototype

Goal: Build the first multi-agent reasoning workflow on top of the hardened RAG runtime using LangGraph. Also resolves foundational multilingual gaps and delivers the first end-to-end answer generation capability.

Planned direction:

- **Story 3.1.1**: Migrate embedding model to `paraphrase-multilingual-MiniLM-L12-v2` and add jieba-based CJK word segmentation to the BM25 tokenizer. Prerequisite for accurate non-English retrieval.
- **Story 3.1.2**: Add plain-text (`.txt`) parser so users can upload plain documents in addition to PDF and Markdown.
- **Story 3.2**: LLM answer generation using Groq API — prompt assembly, context budget management, citation mapping, and graceful API failure handling. This is the first Story that produces a usable end-to-end answer.
- LangGraph state graph foundation for cyclic reasoning and state persistence.
- Observability and tracing setup (LangSmith or Arize Phoenix) at the start of agent development.
- Researcher agent that can query retrieved context and identify information gaps.
- Reporter agent that can produce grounded answers with explicit citation mapping.
- Web search tool integration through MCP when local context is insufficient.
- Error recovery and state rollback for multi-agent workflow failures.
- Quality gate agent for answer validation against user intent.

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

## Known Scope Gaps & Backlog

- [ ] [Issue #1](https://github.com/AlgoMonokuma/multi-agent-rag-assistant/issues/1): PDF image and table support (scope gap identified after Story 2.1, targeting v0.3.0 or later)
- [ ] Embedding model migration: `all-MiniLM-L6-v2` → `paraphrase-multilingual-MiniLM-L12-v2` (Story 3.1.1)
- [ ] CJK-aware BM25 tokenization using jieba (Story 3.1.1)
- [ ] LLM answer generation end-to-end (Story 3.2 — currently 0% implemented despite Groq key being configured)
- [ ] File upload API with security validation — magic-byte check, size limit, batch limit, temp-file cleanup (Story 4.2)
- [ ] FAISS write serialization — asyncio.Lock to prevent race condition under concurrent ingestion (Story 4.2)
- [ ] Session persistence — disk-backed index + browser localStorage session ID (Story 5.1)
- [ ] Plain-text (.txt) parser (Story 3.1.2)
