# Contributing Guide

This project uses a small, test-first workflow. The goal is to keep each change easy to review, easy to run locally, and clearly connected to a user-facing or engineering outcome.

## Local Setup

Install dependencies:

```powershell
uv sync --group dev
```

Create a local `.env` from the example file:

```powershell
Copy-Item .env.example .env
```

Never commit `.env`, credentials, local-only tool output, or generated temporary files.

## Development Workflow

1. Keep each change focused on one story, bug fix, or documentation improvement.
2. Add or update tests for behavior changes.
3. Run the smallest relevant test suite before committing.
4. Use clear conventional commit messages, such as `feat(rag): add hybrid retriever`.
5. Keep public documentation aligned when behavior, setup, or project scope changes.

## Branch and Commit Conventions

Use short-lived branches for focused work. Prefer names that describe the change type and scope:

- `feature/<story-or-feature-name>`
- `fix/<bug-or-component-name>`
- `docs/<documentation-topic>`

Use conventional commit messages for commits and merge titles:

- `feat(<scope>): implement <feature-or-story>`
- `fix(<scope>): handle <bug-or-failure-case>`
- `docs: update <documentation-topic>`
- `test(<scope>): cover <behavior-or-failure-path>`
- `build: add <dependency-or-build-change>`

Keep commit messages concise and outcome-focused. Use the commit body only when the change needs extra context, verification notes, or migration details.

## Pull Request Format

Use this structure for branch and pull request descriptions:

### Background

Explain why the branch exists and which story, bug, or project need it addresses.

### Changes Made

Summarize the main implementation, test, documentation, or dependency changes.

### Verification

List the test commands or checks run, plus the result when available.

### Resolves

List the issue, story, acceptance criteria, or review findings resolved by the branch.

## Testing

Run the full test suite:

```powershell
uv run pytest
```

Run only RAG unit tests:

```powershell
uv run pytest tests/unit/core/rag
```

## Code Style

- Follow the project-wide [Engineering Standards](docs/engineering-standards.md).
- Use `snake_case` for functions and variables.
- Use `PascalCase` for classes.
- Keep core logic inside `core/`.
- Keep API concerns inside `api/`.
- Keep UI concerns inside `app/`.
- Prefer explicit exceptions over silent failure.
- Use the shared logger instead of `print()`.

## Public Documentation

The public documentation lives in `docs/`, `README.md`, and `CHANGELOG.md`.

Internal planning files, local tool configuration, generated diffs, and temporary review notes should stay out of GitHub. Keep machine-specific exclusions in local Git excludes when they are not useful to collaborators.

## Release Notes

Update `CHANGELOG.md` when a set of related stories reaches a meaningful milestone. Use GitHub Releases for milestones that an interviewer or external reviewer can understand and try, not for every individual story.
