# Story 2.4.5: Document-Type Chunking Profile

Status: Complete

## User Story

As a user, I want to select a document type when uploading files so that the system applies chunking and retrieval weights that fit my content type.

## Scope

This story adds configurable chunking profiles and retrieval-weight defaults for semantic, precise, and code-oriented documents.

## Acceptance Criteria

1. Given a `semantic`, `precise`, or `code` document type, when ingestion runs, then the matching `ChunkingProfile` is applied.
2. Given no document type, when ingestion runs, then the `semantic` profile is used by default.
3. Given a chunking profile, when `TextChunker` is created, then profile chunk size and overlap override the default values.
4. Given a profile with vector and keyword weights, when ingestion completes, then the active session stores those retrieval weights.
5. Given hybrid retrieval runs after ingestion, when no explicit weights are passed, then the retriever can use the session's profile weights.
6. Given the story is complete, when profile tests run, then profile configuration, defaults, and invalid document type behavior are covered.

## Implementation Notes

- `DocumentType`, `ChunkingProfile`, and `CHUNKING_PROFILES` live in `core/rag/chunker.py`.
- `ingest_documents()` accepts an optional `document_type`.
- `SessionIndexRecord` stores profile weights for later retrieval.
- Tests live in `tests/unit/core/rag/test_chunking_profile.py`.

Profile matrix:

| Profile | Use Case | Chunk Size | Overlap | Vector Weight | Keyword Weight |
| --- | --- | ---: | ---: | ---: | ---: |
| `semantic` | General prose and conceptual documents | 1000 | 200 | 0.7 | 0.3 |
| `precise` | FAQ, policy, or exact-match documents | 400 | 100 | 0.4 | 0.6 |
| `code` | Code-heavy or notebook-like content | 600 | 50 | 0.6 | 0.4 |

## Out of Scope

- New parser types
- User interface controls
- Model-specific embedding changes
- Persistent profile preferences

## Definition of Done

- `core/rag/chunker.py`
- `core/rag/pipeline.py`
- `core/rag/indexer.py`
- `core/rag/retriever.py`
- `tests/unit/core/rag/test_chunking_profile.py`
