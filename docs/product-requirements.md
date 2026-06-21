# Product Requirements

## Product Vision

AI Knowledge Work Assistant helps users ask questions over uploaded documents and receive structured, citation-aware answers. The product emphasizes retrieval reliability, source traceability, and a clean engineering foundation suitable for future agent workflows.

## Target Users

- Knowledge workers who need to inspect or summarize PDF and Markdown documents.
- Students and enterprise users who may upload large volumes of documents for research or business Q&A.
- Technical reviewers evaluating a production-oriented RAG architecture.
- Developers who want a clear example of session-isolated retrieval infrastructure.

## Language and Locale Requirements

The system is expected to serve users uploading documents in multiple languages including English, Traditional Chinese, Simplified Chinese, Japanese, and other languages. The following constraints apply:

- The embedding model must be a genuine multilingual model capable of producing meaningful vectors for non-English text.
- The keyword search tokenizer must handle CJK (Chinese, Japanese, Korean) text using word-segmentation rather than single-character splitting.
- The default embedding model is `paraphrase-multilingual-MiniLM-L12-v2` (384-dimension, supports 50+ languages) rather than `all-MiniLM-L6-v2` (English-primary).

## Problem

Document QA systems often fail in three ways:

- They retrieve irrelevant chunks because vector search alone is too broad.
- They lose citation metadata while merging or transforming results.
- They mix state across users or sessions when indexes are shared too loosely.

This project addresses those risks by building a session-scoped RAG runtime with hybrid search, re-ranking, and metadata preservation.

## Goals

- Parse PDF and Markdown content into a consistent internal document model.
- Chunk content with metadata that can be traced back to the source file.
- Store each session in an isolated FAISS index.
- Combine vector and keyword retrieval for better recall and precision.
- Re-rank candidate chunks before passing final context to an LLM.
- Prepare the foundation for future agent orchestration and streaming UI.

## Non-Goals

- Multi-user authentication (login system) is not required. Sessions are identified by a browser-persisted session ID.
- Long-term persistent vector storage beyond the active development milestone is deferred.
- Complex multi-agent arbitration is not part of the current baseline.
- Production observability infrastructure is deferred until deployment hardening.
- Support for binary executable formats (.exe, .zip, .js) is explicitly excluded and will be rejected at upload.

## Supported File Formats

The system applies a whitelist policy. Only the following formats are accepted for upload:

| Format | Extensions | Priority |
| --- | --- | --- |
| PDF | `.pdf` | v0.3.0 — required |
| Markdown | `.md`, `.markdown` | v0.2.0 — complete |
| Plain text | `.txt` | v0.3.0 — required |
| Word document | `.docx` | v0.4.0 — planned |
| PowerPoint | `.pptx` | v0.4.0 — planned |

All other file types must be rejected before any parsing is attempted. Validation uses both extension checking and magic-byte (file header) verification to prevent disguised uploads.

## File Upload Constraints

- **Per-file size limit**: 100 MB per single file. Larger files must be split by the user before upload.
- **Per-batch limit**: Maximum 10 files per upload request to prevent server overload.
- **Total session volume**: No hard cap — users may upload as many files as they need across multiple requests.
- **Rationale**: A per-file limit prevents extremely long embedding jobs from timing out. The batch limit prevents concurrent memory exhaustion. Total session volume is left open to maximize usability.

## Functional Requirements

| ID | Requirement | Status |
| --- | --- | --- |
| FR1 | Parse PDF and Markdown files while preserving metadata. | Complete |
| FR2 | Create session-isolated FAISS indexes. | Complete |
| FR3 | Convert parsed documents into chunks and embeddings using a multilingual model. | Complete (model migration planned) |
| FR4 | Retrieve relevant chunks using hybrid vector and keyword search with CJK-aware tokenization. | Complete (CJK tokenizer planned) |
| FR5 | Support document-type profiles for semantic, precise, and code content. | Complete |
| FR6 | Re-rank retrieved chunks with a cross-encoder. | Complete |
| FR7 | Add runtime hardening around metadata, partial updates, and parameter boundaries. | Complete |
| FR8 | Validate uploaded files by extension whitelist and magic-byte verification before parsing. | Planned — Epic 4 |
| FR9 | Generate LLM answers from re-ranked context chunks with citation mapping. | Planned — Epic 3 |
| FR10 | Add agent workflow orchestration. | Foundation complete — Epic 3 |
| FR11 | Add streaming answer experience with reasoning trace. | Planned — Epic 4 |
| FR12 | Persist session state to disk so users can resume after browser reload or server restart. | Planned — Epic 5 |
| FR13 | Support plain-text (.txt) file upload in addition to PDF and Markdown. | Planned — Epic 3 |
| FR14 | Support fallback OCR and image description for PDFs with embedded images or scanned pages. | Planned — Epic 3 (Issue #1) |

## Quality Requirements

- Retrieval results must preserve `source`, `chunk_id`, and `session_id`.
- Session data must not leak across FAISS indexes.
- The embedding model must produce meaningful results for English and CJK languages.
- Keyword search tokenization must use word-level segmentation for CJK text.
- FAISS write operations must be serialized to prevent race conditions under concurrent requests.
- Files must be validated (extension + magic bytes) before any parsing is attempted.
- Tests must cover success paths, boundary behavior, and failure paths.
- Heavy model instances should be loaded lazily and reused where possible.
- Public documentation must explain the architecture without exposing local-only process artifacts.

## Success Criteria

- A reviewer can understand the system from the README and docs within a few minutes.
- The RAG unit suite can verify parser, chunking, indexing, retrieval, and re-ranking behavior.
- New RAG features can be mapped to a story, test file, and changelog entry.
- Future UI and agent work can build on the existing RAG runtime without redesigning retrieval contracts.
