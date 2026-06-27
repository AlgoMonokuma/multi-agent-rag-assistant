# Story 3.1.2: Plain-Text Parser Support

Status: Ready for Dev

## User Story

As a user, I want to upload `.txt` files so that plain-text documents can be indexed alongside PDF and Markdown files in the RAG pipeline.

## Scope

This story extends the ingestion pipeline by adding support for plain text files. It involves implementing a dedicated parser that handles UTF-8 text extraction, preserves source metadata, and integrates into the existing `BaseParser` framework.

## Acceptance Criteria

1. Given a `.txt` file path, when `TextFileParser.parse()` runs, then the full text content is extracted.
2. Given a parsed text document, then metadata includes the `source` filename.
3. Given a text file, when parsing occurs, then UTF-8 encoding is expected by default.
4. Given a file with invalid encoding (non-UTF8), when parsing fails, then `ParserException` is raised with a clear error message.
5. Given a non-existent file path, when parsing is attempted, then `ParserException` is raised.
6. Given the parser unit tests run, then success, encoding failure, and file-not-found paths are covered.

## Implementation Notes

- `core/rag/parser.py` will include the `TextFileParser` class inheriting from `BaseParser`.
- The parser will use a standard `with open(file_path, "r", encoding="utf-8")` pattern.
- Metadata is limited to `source` as plain text files do not have page or heading structure by default.
- Failures are wrapped in the domain-specific `ParserException` to maintain consistency with PDF and Markdown parsers.

## Out of Scope

- Support for legacy encodings (e.g., Latin-1, GBK).
- Automatic encoding detection (chardet).
- Parsing of structured text like CSV or JSON (handled by separate logic if needed).

## Definition of Done

- `TextFileParser` is implemented and follows the project's parser interface.
- Unit tests in `tests/unit/core/rag/test_parser.py` cover all acceptance criteria.
- Existing RAG ingestion tests pass without regression.
- Story status is updated in the internal sprint tracker and roadmap.
