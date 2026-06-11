# Product Requirements

## Product Vision

AI Knowledge Work Assistant helps users ask questions over uploaded documents and receive structured, citation-aware answers. The product emphasizes retrieval reliability, source traceability, and a clean engineering foundation suitable for future agent workflows.

## Target Users

- Knowledge workers who need to inspect or summarize PDF and Markdown documents.
- Technical reviewers evaluating a production-oriented RAG architecture.
- Developers who want a clear example of session-isolated retrieval infrastructure.

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

- Multi-user authentication is not part of the current milestone.
- Long-term persistent vector storage is not part of the current milestone.
- Complex multi-agent arbitration is not part of the current baseline.
- Production observability infrastructure is deferred until deployment hardening.

## Functional Requirements

| ID | Requirement | Status |
| --- | --- | --- |
| FR1 | Parse PDF and Markdown files while preserving metadata. | Complete |
| FR2 | Create session-isolated FAISS indexes. | Complete |
| FR3 | Convert parsed documents into chunks and embeddings. | Complete |
| FR4 | Retrieve relevant chunks using hybrid vector and keyword search. | Complete |
| FR5 | Support document-type profiles for semantic, precise, and code content. | Complete |
| FR6 | Re-rank retrieved chunks with a cross-encoder. | Complete |
| FR7 | Add runtime hardening around metadata, partial updates, and parameter boundaries. | Planned |
| FR8 | Add agent workflow orchestration. | Planned |
| FR9 | Add streaming answer experience. | Planned |

## Quality Requirements

- Retrieval results must preserve `source`, `chunk_id`, and `session_id`.
- Session data must not leak across FAISS indexes.
- Tests must cover success paths, boundary behavior, and failure paths.
- Heavy model instances should be loaded lazily and reused where possible.
- Public documentation must explain the architecture without exposing local-only process artifacts.

## Success Criteria

- A reviewer can understand the system from the README and docs within a few minutes.
- The RAG unit suite can verify parser, chunking, indexing, retrieval, and re-ranking behavior.
- New RAG features can be mapped to a story, test file, and changelog entry.
- Future UI and agent work can build on the existing RAG runtime without redesigning retrieval contracts.
