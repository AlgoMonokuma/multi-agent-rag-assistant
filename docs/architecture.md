# Architecture Overview

## Summary

The system is organized around a session-isolated RAG runtime. Each user session owns its own FAISS index and metadata registry. Parsed documents move through chunking, embedding, indexing, hybrid retrieval, and optional cross-encoder re-ranking before final context is assembled for answer generation.

```text
PDF / Markdown
      |
      v
Parser -> ParsedDocument[]
      |
      v
TextChunker -> chunk metadata
      |
      v
SentenceTransformerEmbedder -> float32 vectors
      |
      v
SessionIndexer -> per-session FAISS index + metadata maps
      |
      v
HybridRetriever -> vector hits + keyword hits
      |
      v
CrossEncoderReranker -> final ordered chunks
```

## Runtime Components

| Component | File | Responsibility |
| --- | --- | --- |
| Parser | `core/rag/parser.py` | Convert PDF and Markdown files into normalized parsed documents. |
| Chunker | `core/rag/chunker.py` | Split parsed text and apply document-type chunking profiles. |
| Embedder | `core/rag/embeddings.py` | Generate `float32` sentence-transformer vectors. |
| Indexer | `core/rag/indexer.py` | Manage session-scoped FAISS indexes and metadata maps. |
| Pipeline | `core/rag/pipeline.py` | Orchestrate chunking, embedding, indexing, and profile weight updates. |
| Retriever | `core/rag/retriever.py` | Merge vector and keyword retrieval results with deterministic ordering. |
| Reranker | `core/rag/reranker.py` | Re-rank candidate chunks with a cross-encoder model. |
| API | `api/main.py` | Provide the FastAPI application boundary. |
| UI | `app/main.py` | Provide the Streamlit application boundary. |

## Session Isolation

The current architecture uses in-memory FAISS indexes per session. A `SessionIndexRecord` stores:

- `session_id`
- FAISS index instance
- chunk metadata
- parsed chunk documents
- vector ordinal to `chunk_id` mapping
- optional retrieval weights for the active document profile

This design intentionally avoids cross-session sharing of indexes and metadata. Model instances may be reused, but user documents and retrieval metadata stay session-scoped.

## Retrieval Strategy

Hybrid retrieval combines two signals:

- Vector similarity for semantic matching.
- Keyword scoring for exact or precise matches.

The document-type profile can tune `vector_weight` and `keyword_weight`:

| Profile | Use Case | Chunk Size | Overlap | Vector Weight | Keyword Weight |
| --- | --- | ---: | ---: | ---: | ---: |
| `semantic` | General prose and conceptual documents | 1000 | 200 | 0.7 | 0.3 |
| `precise` | FAQ, policy, or exact-match documents | 400 | 100 | 0.4 | 0.6 |
| `code` | Code-heavy or notebook-like content | 600 | 50 | 0.6 | 0.4 |

## Re-Ranking Strategy

The cross-encoder re-ranker receives the top candidates from hybrid retrieval and scores each `[query, passage]` pair. The highest-scoring chunks become the final ordered context.

The default model is:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The re-ranker supports dependency injection for tests and lazy loading for runtime efficiency.

## Testing Strategy

The current unit suite focuses on:

- Parser success and error paths.
- Chunk boundaries and metadata preservation.
- Embedding dimensions and dtype conversion.
- Session isolation and FAISS ordinal mapping.
- Hybrid search scoring and deterministic ordering.
- Document-type profile behavior.
- Re-ranking behavior and fallback paths.

## Known Technical Debt

- Public docs have been cleaned, but some internal artifacts and legacy comments still contain encoding noise.
- API and UI are currently bootstrap-level and do not yet expose the full RAG workflow.
- Runtime hardening around partial updates and metadata validation is planned as the next story.
