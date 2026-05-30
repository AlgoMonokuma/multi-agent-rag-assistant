# Story 2.2: Session-Isolated Indexing Foundation

Status: Complete

## User Story

As a developer, I want each user session to own an isolated FAISS index and metadata registry so that one session cannot read or mutate another session's retrieval state.

## Acceptance Criteria

1. Given a new session request, when a session is created, then the system creates a unique session id and an independent FAISS index.
2. Given multiple sessions, when documents are stored, then chunk metadata remains isolated per session.
3. Given a session id, when metadata or chunks are requested, then only that session's records are returned.
4. Given an unknown session or invalid index operation, when the indexer is used, then a domain-specific indexer exception is raised.
5. Given the story is complete, when indexer tests run, then session creation, isolation, cleanup, and error paths are covered.

## Implementation Notes

- Index management lives in `core/rag/indexer.py`.
- `SessionIndexRecord` stores the FAISS index, metadata maps, and vector ordinal mapping.
- `SessionIndexer` creates sessions, stores chunk metadata, ingests embeddings, and cleans up sessions.
- Tests live in `tests/unit/core/rag/test_indexer.py`.

## Public Evidence

- `core/rag/indexer.py`
- `tests/unit/core/rag/test_indexer.py`
