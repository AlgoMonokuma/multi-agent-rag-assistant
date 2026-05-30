# Story 2.1: Document Ingestion and Parser Pipeline

Status: Complete

## User Story

As a user, I want to upload PDF or Markdown files so that the system can extract clean text while preserving source metadata such as page numbers and titles.

## Acceptance Criteria

1. Given a PDF file, when it is parsed, then the system extracts page text and records `source` and `page` metadata.
2. Given a Markdown file, when it is parsed, then the system extracts document text and records `source` and `title` metadata.
3. Given different supported document types, when parsing completes, then the output uses a consistent internal document model.
4. Given an invalid path or parser failure, when parsing is attempted, then the system raises a domain-specific parser exception.
5. Given the story is complete, when parser tests run, then PDF, Markdown, and error-path behavior are covered.

## Implementation Notes

- Parser logic lives in `core/rag/parser.py`.
- The normalized output model is `ParsedDocument`.
- Parser errors are surfaced through `ParserException`.
- Tests live in `tests/unit/core/rag/test_parser.py`.

## Public Evidence

- `core/rag/parser.py`
- `tests/unit/core/rag/test_parser.py`
