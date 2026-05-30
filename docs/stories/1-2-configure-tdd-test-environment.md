# Story 1.2: Configure TDD Test Environment

Status: Complete

## User Story

As a developer, I want a working pytest setup so that future stories can be implemented with focused automated tests.

## Scope

This story establishes the test runner, test folders, and baseline sanity checks.

## Acceptance Criteria

1. Given the project root, when `pytest` runs, then test discovery works.
2. Given unit tests, then they live under `tests/unit/`.
3. Given integration tests, then they live under `tests/integration/`.
4. Given baseline project setup, then at least one sanity test verifies the test runner.

## Implementation Notes

- Test configuration lives in `pyproject.toml`.
- Unit tests should be small and deterministic.
- Integration tests should avoid external network dependencies unless explicitly required.

## Out of Scope

- Feature-specific RAG tests
- CI pipeline configuration
- Deployment tests

## Definition of Done

- Pytest runs from the project root.
- Unit and integration folders exist.
- Baseline sanity tests pass.
