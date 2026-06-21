# Changelog

All notable public-facing project changes are tracked here.

The project uses semantic versioning while it is pre-1.0:

- `v0.x.0` for meaningful project milestones.
- `v0.x.y` for documentation, bug fix, or hardening updates.
- `v1.0.0` for the first demo-ready release.

## Unreleased

### Added
- Story 3.1 LangGraph workflow foundation with typed agent state,
  researcher/reporter/reviewer stub nodes, lazy graph compilation, and
  conditional retry routing.

### Planned
- PDF parser image and table support — see [Issue #1](https://github.com/AlgoMonokuma/multi-agent-rag-assistant/issues/1).
- Epic 3: Agent Workflow prototype implementation.

## v0.2.2 - 2026-06-11

### Changed
- Standardized all docstrings, comments, log messages, and exception
  messages to English across Python source and tests (Story 2.5.2).

## v0.2.1 - 2026-06-11

### Added
- Runtime guardrails: failed ingestion no longer persists partial state.
- Citation metadata validation for source, chunk_id, and session_id.
- Top-K and Top-N boundary behavior made explicit and tested.
- Lazy-loaded model instances reused after first load.
- Failure paths raise domain exceptions with structured logging.

### Docs
- Public documentation set under docs/.
- Clean README and contributing guide.
- Public roadmap and release strategy.
- Public Story specifications.

## v0.2.0 - 2026-05-30

### Added
- PDF and Markdown parser pipeline.
- Session-isolated FAISS indexing foundation.
- Text chunking and embedding pipeline.
- Hybrid vector and keyword retrieval.
- Document-type chunking profiles.
- Cross-encoder re-ranking mechanism.

### Testing
- Unit coverage for parser, chunker, indexer, retrieval,
  document-type profiles, and re-ranking behavior.

## v0.1.0 - 2026-04-10

### Added
- Python project foundation with uv.
- Secure environment configuration.
- Shared logging.
- Initial FastAPI and Streamlit bootstraps.
- Baseline test structure.