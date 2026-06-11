# Story 2.5.1: RAG Runtime Hardening and Harness Guardrails

Status: Complete

## User Story

As a developer, I want lightweight runtime guardrails around ingestion, retrieval, and re-ranking so that the RAG pipeline remains observable, recoverable, and consistent under normal and failure scenarios.

## Scope

This story hardens existing RAG runtime behavior around state consistency, citation metadata, boundary handling, model reuse, and failure observability.

## Acceptance Criteria

1. Given ingestion updates document-type retrieval weights, when chunking, embedding, or indexing fails, then the system must not leave partially updated session state.
2. Given hybrid retrieval results, when `HybridSearchResult` is assembled, then required citation metadata must include at least `source`, `chunk_id`, and `session_id`.
3. Given optional citation metadata such as `page`, `title`, `parent_source`, or `chunk_index`, when merge or re-ranking runs, then those fields must not be lost.
4. Given `top_k` and `top_n` inputs, when values are zero, negative, or inconsistent, then behavior must be explicit and covered by tests.
5. Given lazy-loaded models, when repeated queries run in one process, then model instances should be reused rather than re-initialized per query.
6. Given failure paths, when runtime errors occur, then domain exceptions and logs must make the failure observable.

## Implementation Notes

- Partial update protection should live near ingestion orchestration in `core/rag/pipeline.py`.
- Citation metadata checks should stay close to retrieval result assembly in `core/rag/retriever.py`.
- Re-ranker model reuse should be handled without making tests load real model weights.
- Tests should cover partial update protection, required citation metadata, Top-K and Top-N boundaries, re-ranker model reuse, and failure-path logging.

## Out of Scope

- New retrieval algorithms
- UI changes
- Production monitoring backend
- Persistent model serving infrastructure

## Definition of Done

- `core/rag/pipeline.py`
- `core/rag/retriever.py`
- `core/rag/reranker.py`
- `tests/unit/core/rag/test_retriever.py`
- `tests/unit/core/rag/test_reranker.py`
- Additional focused hardening tests as needed.
