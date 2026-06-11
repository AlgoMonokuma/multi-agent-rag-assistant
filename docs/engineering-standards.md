# Engineering Standards

## Purpose

This document defines the engineering standards for AI Knowledge Work Assistant. Use it as the highest-level guide when writing code, tests, documentation, and story specifications.

## Language Policy

- Public documentation must be written in English.
- Internal story specifications should be written in English.
- Code comments and docstrings should be written in English.
- User-facing UI text can be localized later, but the implementation baseline should stay English until localization is an explicit feature.
- Avoid mixing Chinese and English in the same documentation set unless the document is intentionally bilingual.

Rationale: English is the most common language for GitHub review, technical interviews, open-source conventions, package APIs, and AI-assisted implementation.

## Python Style

- Follow PEP 8 naming conventions.
- Use `snake_case` for variables, functions, modules, and file names.
- Use `PascalCase` for classes and exceptions.
- Use `UPPER_SNAKE_CASE` for constants.
- Prefer type hints for public functions and core runtime contracts.
- Prefer small pure functions when behavior is easy to isolate.
- Avoid broad `except Exception` unless the code re-raises a domain-specific exception with useful context.

## Architecture Principles

- High cohesion: each module should own one clear responsibility.
- Low coupling: modules should depend on stable interfaces rather than unrelated implementation details.
- Separation of concerns: API, UI, and core RAG logic must stay in separate layers.
- Dependency injection: external models, indexes, and clients should be injectable for tests.
- Explicit boundaries: a story should state which files are likely to change and which areas are out of scope.
- Backward compatibility: do not rename public result fields unless the story explicitly requires it.

## RAG Runtime Rules

- Session data must remain isolated by `session_id`.
- FAISS indexes and metadata maps must not be shared across sessions.
- Retrieval results must preserve required citation metadata: `source`, `chunk_id`, and `session_id`.
- Optional citation metadata should be preserved when available: `page`, `title`, `parent_source`, and `chunk_index`.
- Heavy model resources may be lazy-loaded and reused if they are read-only runtime resources.
- Failure paths must raise domain-specific exceptions and emit useful logs.

## Testing Standards

- Every story that changes behavior should include tests.
- Unit tests should cover happy paths, boundary cases, and failure paths.
- Tests should avoid real network calls and real secrets.
- Use dependency injection or lightweight fakes for models and indexes.
- Prefer explicit test names such as `test_search_top_k_zero_returns_empty_result`.

## Documentation Standards

- Every public story spec should use the same section order:
  - User Story
  - Scope
  - Acceptance Criteria
  - Implementation Notes
  - Out of Scope
  - Definition of Done
- Public docs must not mention private workflow tools, execution logs, generated diffs, or model/tool transcripts.
- Public docs must not include real secrets, local absolute paths, or local-only process notes.
- `README.md` should explain the product, setup, architecture summary, and documentation links.
- `CHANGELOG.md` should summarize milestone-level changes.

## Git and Release Standards

- Use focused commits.
- Use conventional commit style when practical, such as `feat(rag): add hybrid retrieval`.
- Push normal story work as commits.
- Use GitHub Releases for meaningful milestones, not for every story.
- Use pre-1.0 semantic versions until the project is demo-ready.

## Current Codebase Note

Some existing code comments and runtime messages were written before this English-only standard. Do not mix comment cleanup into unrelated feature stories. Use Story 2.5.2 as the dedicated non-behavioral refactor for comment, docstring, log, exception, and test-message migration, then run the full test suite.
