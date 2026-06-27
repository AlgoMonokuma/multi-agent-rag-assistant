# Architecture Overview

## Summary

The system is organized around a session-isolated RAG runtime. Each user session owns its own FAISS index and metadata registry. Parsed documents move through chunking, embedding, indexing, hybrid retrieval, and optional cross-encoder re-ranking before final context is assembled for answer generation.

```text
File Upload (PDF / Markdown / TXT)
      |
      v
FileValidator -> extension whitelist + magic-byte check
      |
      v
Parser -> ParsedDocument[]
      |
      v
TextChunker -> chunk metadata (CJK-aware splitting)
      |
      v
MultilingualEmbedder -> float32 vectors
      |
      v
SessionIndexer -> per-session FAISS index + metadata maps (write-serialized)
      |
      v
HybridRetriever -> vector hits + keyword hits (jieba CJK tokenizer)
      |
      v
CrossEncoderReranker -> final ordered chunks
      |
      v
LLM Answer Generator -> grounded answer with citations
```

## Runtime Components

| Component | File | Responsibility |
| --- | --- | --- |
| FileValidator | `core/rag/validator.py` (planned) | Reject disallowed file types using extension whitelist and magic-byte check before parsing. |
| Parser | `core/rag/parser.py` | Convert PDF, Markdown, and TXT files into normalized parsed documents. |
| Chunker | `core/rag/chunker.py` | Split parsed text and apply document-type chunking profiles. |
| Embedder | `core/rag/embeddings.py` | Generate `float32` multilingual sentence-transformer vectors. |
| Indexer | `core/rag/indexer.py` | Manage session-scoped FAISS indexes and metadata maps with write serialization. |
| Pipeline | `core/rag/pipeline.py` | Orchestrate chunking, embedding, indexing, and profile weight updates. |
| Retriever | `core/rag/retriever.py` | Merge vector and keyword retrieval results using CJK-aware tokenization. |
| Reranker | `core/rag/reranker.py` | Re-rank candidate chunks with a cross-encoder model. |
| LLM Generator | `core/rag/generator.py` | Call the configured LLM with retrieved context and return a grounded answer with citations. |
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

### Concurrency Safety

FAISS `IndexFlatL2.add()` is not thread-safe. Because FastAPI handles concurrent requests on the same event loop, all FAISS write operations (ingestion) must be serialized using an `asyncio.Lock` or equivalent per-session lock. Read operations (search) do not mutate index state and are safe to run concurrently.

### Session Persistence Direction (Planned — Epic 5)

The current in-memory store is intentional for the development phase but unsuitable for production. The planned direction:

- Store FAISS indexes and chunk metadata to disk or a lightweight vector store (e.g., ChromaDB, Qdrant, or SQLite + FAISS on-disk).
- Persist the `session_id` in the user's browser `localStorage` so they can resume their session after a page reload or server restart without requiring a login system.
- On session resume, load the saved index from disk rather than re-ingesting all documents.

## Embedding Model

The default embedding model is `paraphrase-multilingual-MiniLM-L12-v2`:

- Dimension: 384-dimension `float32` vectors (compatible with existing FAISS `IndexFlatL2(384)`).
- Language coverage: 50+ languages including English, Traditional Chinese, Simplified Chinese, Japanese, Korean, German, and French.
- Rationale: The previously planned `all-MiniLM-L6-v2` model is English-primary and unsuitable for the multilingual user base.

## Retrieval Strategy

Hybrid retrieval combines two signals:

- Vector similarity for semantic matching (multilingual, model-driven).
- Keyword scoring for exact or precise matches.

### CJK Tokenization

The current `_tokenize` implementation is CJK-aware. It uses `jieba` for Han character blocks, lowercases ASCII tokens, and preserves Japanese kana and Korean Hangul blocks for keyword matching:

```
Han:      "人工智慧" -> jieba word segmentation
ASCII:    "Hello AI" -> ["hello", "ai"]
Kana:     "AIテスト" -> ["ai", "テスト"]
Hangul:   "한국어 테스트" -> ["한국어", "테스트"]
```

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

## Answer Generation

`LLMGenerator` turns retrieved and re-ranked chunks into grounded answers through the configured Groq client. Prompt assembly includes source and chunk ID metadata for admitted chunks, applies a context token budget, and returns citation references only for chunks included in the prompt.

When no usable context is available, the generator returns a graceful fallback answer without calling the LLM. API, configuration, import, and malformed response failures are normalized through `GeneratorException`.

## Testing Strategy

The current unit suite focuses on:

- Parser success and error paths.
- Chunk boundaries and metadata preservation.
- Embedding dimensions and dtype conversion.
- Session isolation and FAISS ordinal mapping.
- Hybrid search scoring and deterministic ordering.
- Document-type profile behavior.
- Re-ranking behavior and fallback paths.
- LLM answer generation, prompt budgeting, citation mapping, and failure handling.

## Evaluation Strategy (Heterogeneous Validation)

The system employs heterogeneous validation across two levels to ensure robustness:

1.  **Agent Diversity**: Multi-agent workflows (Epic 3) use different model families (e.g., Llama and Qwen) to reduce correlated reasoning failures.
2.  **RAG Metrics**: Retrieval and generation quality are measured using a combination of automated metrics (RAGAS), LLM-as-judge, and metadata integrity checks.

## Known Technical Debt

- API and UI are currently bootstrap-level and do not yet expose the full RAG workflow.
- FAISS write serialization (asyncio.Lock) is planned before any concurrent API endpoint exposes ingestion.
- Session persistence (disk-backed index) is planned for Epic 5.
- File upload validation (magic-byte check) is planned for Epic 4 API layer.
- Researcher-node retrieval integration is still planned after the generator foundation.
- Heterogeneous Evaluation Framework (Story 5.5) is required to quantitatively measure RAG quality.
