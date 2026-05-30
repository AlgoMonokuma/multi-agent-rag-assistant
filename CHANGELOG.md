# Changelog

All notable public-facing project changes are tracked here.

The project uses semantic versioning while it is pre-1.0:

- `v0.x.0` for meaningful project milestones.
- `v0.x.y` for documentation, bug fix, or hardening updates.
- `v1.0.0` for the first demo-ready release.

## Unreleased

### Added

- Public documentation set under `docs/`.
- Clean README and contributing guide for GitHub review.
- Public roadmap and release strategy.
- Public Story 1 specifications.
- Engineering standards for code, tests, documentation, and releases.

### Planned

- Story 2.5.1: RAG runtime hardening and harness guardrails.
- Story 2.5.2: Codebase language normalization.
- Runtime metadata validation and Top-K / Top-N boundary rules.
- GitHub release tagging after the next stable milestone.

## v0.2.0 - RAG Runtime Foundation

### Added

- PDF and Markdown parser pipeline.
- Session-isolated FAISS indexing foundation.
- Text chunking and embedding pipeline.
- Hybrid vector and keyword retrieval.
- Document-type chunking profiles.
- Cross-encoder re-ranking mechanism.

### Testing

- Unit coverage for parser, chunker, indexer, retrieval, document-type profiles, and re-ranking behavior.

## v0.1.0 - Project Foundation

### Added

- Python project foundation with `uv`.
- Secure environment configuration.
- Shared logging.
- Initial FastAPI and Streamlit bootstraps.
- Baseline test structure.
