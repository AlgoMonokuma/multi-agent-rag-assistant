# Story 2.5: Re-Ranking Mechanism

Status: Complete

## User Story

As a user, I want retrieved chunks to be re-ranked before final answer generation so that the most relevant passages are prioritized.

## Scope

This story adds optional cross-encoder re-ranking for hybrid search candidates while preserving baseline retrieval behavior when no re-ranker is configured.

## Acceptance Criteria

1. Given top candidate chunks from hybrid search, when re-ranking runs, then a cross-encoder scores query-passage pairs.
2. Given scored chunks, when final results are returned, then chunks are ordered by re-rank score and limited to `top_n`.
3. Given an empty candidate list, when re-ranking runs, then the system returns an empty list without loading the model.
4. Given model loading or prediction failure, when re-ranking runs, then a domain-specific re-ranker exception is raised.
5. Given hybrid retrieval has no configured re-ranker, when search runs, then baseline hybrid search behavior remains unchanged.
6. Given the story is complete, when re-ranker tests run, then model injection, scoring, ranking, empty input, and fallback behavior are covered.

## Implementation Notes

- Re-ranking logic lives in `core/rag/reranker.py`.
- `CrossEncoderReranker` supports dependency injection for tests.
- The default model is `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- `RetrievedChunk` includes optional `rerank_score`.
- Tests live in `tests/unit/core/rag/test_reranker.py`.

## Out of Scope

- Answer generation
- Citation rendering
- Model fine-tuning
- Persistent model cache management

## Definition of Done

- `core/rag/reranker.py`
- `core/rag/retriever.py`
- `tests/unit/core/rag/test_reranker.py`
