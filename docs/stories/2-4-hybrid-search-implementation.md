# Story 2.4: Hybrid Search Implementation

Status: Complete

## User Story

As a user, I want the system to combine semantic vector search and keyword matching so that retrieval works for both broad conceptual queries and precise terms.

## Acceptance Criteria

1. Given an indexed session and a query, when search runs, then the system performs vector retrieval against the session FAISS index.
2. Given the same query, when keyword scoring runs, then the system scores relevant chunks with lightweight BM25-style matching.
3. Given vector and keyword hits, when results are merged, then duplicates are removed and deterministic ordering is applied.
4. Given retrieval results, when they are returned, then each result includes score fields and citation metadata.
5. Given empty sessions or invalid parameters, when search runs, then behavior is explicit and covered by tests.

## Implementation Notes

- Retrieval logic lives in `core/rag/retriever.py`.
- Result objects include vector score, keyword score, merged score, rank, and metadata.
- The retriever preserves metadata needed for future citation display.
- Tests live in `tests/unit/core/rag/test_retriever.py`.

## Public Evidence

- `core/rag/retriever.py`
- `tests/unit/core/rag/test_retriever.py`
