# Story 2.3: Text Chunking and Embedding Pipeline

Status: Complete

## User Story

As a developer, I want parsed documents to be chunked, embedded, and stored in the active session index so that document content becomes searchable through vector retrieval.

## Scope

This story connects parsed documents to chunking, embedding generation, and session index ingestion.

## Acceptance Criteria

1. Given parsed documents, when the ingestion pipeline runs, then text is split into chunks while preserving source metadata.
2. Given chunked documents, when embeddings are generated, then the system produces `384`-dimension `float32` vectors.
3. Given embeddings and chunk metadata, when indexing completes, then FAISS vector order maps deterministically back to chunk ids.
4. Given multiple batches in the same session, when ingestion runs again, then vectors and metadata append without corrupting existing mappings.
5. Given invalid sessions or embedding/indexing failures, when ingestion runs, then the system raises explicit domain exceptions and avoids corrupting session state.

## Implementation Notes

- Chunking lives in `core/rag/chunker.py`.
- Embedding logic lives in `core/rag/embeddings.py`.
- Pipeline orchestration lives in `core/rag/pipeline.py`.
- Session storage and FAISS ordinal mapping live in `core/rag/indexer.py`.
- Tests live in `tests/unit/core/rag/test_chunker.py` and `tests/unit/core/rag/test_indexer.py`.

## Out of Scope

- Hybrid keyword retrieval
- Re-ranking
- Answer generation
- UI upload workflow

## Definition of Done

- `core/rag/chunker.py`
- `core/rag/embeddings.py`
- `core/rag/pipeline.py`
- `core/rag/indexer.py`
- `tests/unit/core/rag/test_chunker.py`
- `tests/unit/core/rag/test_indexer.py`
