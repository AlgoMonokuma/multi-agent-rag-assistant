# Story 3.1.1: Multilingual Embedding and CJK Tokenization

Status: Complete

## User Story

As a multilingual user, I want vector retrieval and keyword retrieval to work for English and CJK documents so that uploaded content in Chinese, Japanese, Korean, and Latin-script languages can be searched with better relevance.

## Scope

This story improves the retrieval foundation by switching the default sentence-transformer embedding model to a multilingual model and making keyword tokenization CJK-aware. It preserves the existing FAISS vector dimension and hybrid retrieval contract while improving non-English retrieval behavior.

## Acceptance Criteria

1. Given the default embedder configuration, when embeddings are generated, then the model name is `paraphrase-multilingual-MiniLM-L12-v2`.
2. Given the embedding model migration, when vectors are produced, then the existing 384-dimensional FAISS index contract remains valid.
3. Given Chinese Han text in a query or document chunk, when keyword tokenization runs, then the tokenizer uses `jieba` word segmentation instead of single-character matching.
4. Given ASCII tokens mixed with CJK text, when keyword tokenization runs, then ASCII tokens remain lowercased and searchable.
5. Given Japanese kana or Korean Hangul text, when keyword tokenization runs, then contiguous kana and Hangul terms are preserved for keyword matching.
6. Given the retrieval test suite, when the focused retriever tests run, then tokenization behavior is covered for ASCII, Han text, Japanese kana, and Korean Hangul.

## Implementation Notes

- Default embedding configuration lives in `core/rag/embeddings.py`.
- Keyword tokenization lives in `HybridRetriever._tokenize` in `core/rag/retriever.py`.
- `jieba` is a runtime dependency because keyword retrieval uses it directly.
- The tokenizer applies `jieba` only to Han character blocks and preserves other supported token groups directly.
- The change is foundational for future agent workflows because downstream answer generation depends on retrieval quality.

## Out of Scope

- LLM answer generation
- Agent orchestration behavior
- Plain-text (`.txt`) parser support
- OCR or image extraction from PDFs
- Persistent vector storage
- Retrieval evaluation benchmarks

## Definition of Done

- `core/rag/embeddings.py` uses `paraphrase-multilingual-MiniLM-L12-v2` as the default embedding model.
- `pyproject.toml` includes `jieba`.
- `core/rag/retriever.py` tokenizes Han text with `jieba` while preserving ASCII, Japanese kana, and Korean Hangul token paths.
- `tests/unit/core/rag/test_retriever.py` covers the multilingual tokenizer behavior.
- Public docs and roadmap no longer list Story 3.1.1 as planned.

## Completion Notes

- Migrated the default embedding model to `paraphrase-multilingual-MiniLM-L12-v2`.
- Kept the embedding dimension at 384 for compatibility with the current FAISS index setup.
- Added CJK-aware keyword tokenization with `jieba` for Han text.
- Preserved ASCII keyword matching and added explicit Japanese kana and Korean Hangul token coverage.
- Updated public documentation to mark Story 3.1.1 complete.
