# Story 1.4: Shared Logging and Engineering Documentation

Status: Complete

## User Story

As a developer, I want shared logging and engineering documentation so that debugging, testing, and collaboration remain consistent.

## Scope

This story introduces a shared logger and contributor-facing development guidance.

## Acceptance Criteria

1. Given runtime code, then it uses the shared logger instead of `print()`.
2. Given an error path, then logs include useful context.
3. Given a new contributor, then setup and test instructions are documented.
4. Given public documentation, then private workflow artifacts are not required to understand the project.

## Implementation Notes

- Logger lives in `core/log.py`.
- Contributor guidance lives in `CONTRIBUTING.md`.
- Public-facing documentation lives in `README.md`, `CHANGELOG.md`, and `docs/`.

## Out of Scope

- Centralized production logging
- Metrics backend
- Distributed tracing

## Definition of Done

- Shared logger exists.
- Contributor guide exists.
- README explains local setup.
- Logging tests pass.
