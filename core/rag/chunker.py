"""Text chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Protocol, Sequence

from core.log import logger
from core.rag.parser import ParsedDocument


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


class ChunkingException(Exception):
    """Text chunking error."""


class DocumentType(str, Enum):
    """Document category used to select chunking and retrieval defaults."""

    SEMANTIC = "semantic"
    PRECISE = "precise"
    CODE = "code"


@dataclass(frozen=True)
class ChunkingProfile:
    """Immutable chunking and hybrid retrieval profile."""

    chunk_size: int
    chunk_overlap: int
    vector_weight: float
    keyword_weight: float


CHUNKING_PROFILES: dict[DocumentType, ChunkingProfile] = {
    DocumentType.SEMANTIC: ChunkingProfile(
        chunk_size=1000,
        chunk_overlap=200,
        vector_weight=0.7,
        keyword_weight=0.3,
    ),
    DocumentType.PRECISE: ChunkingProfile(
        chunk_size=400,
        chunk_overlap=100,
        vector_weight=0.4,
        keyword_weight=0.6,
    ),
    DocumentType.CODE: ChunkingProfile(
        chunk_size=600,
        chunk_overlap=50,
        vector_weight=0.6,
        keyword_weight=0.4,
    ),
}


class TextSplitter(Protocol):
    """Injectable text splitter interface."""

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""


class TextChunker:
    """Convert ParsedDocument objects into embeddable chunk documents."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        splitter: TextSplitter | None = None,
        profile: ChunkingProfile | None = None,
    ) -> None:
        """Initialize the text chunker."""
        if profile is not None:
            self._chunk_size = profile.chunk_size
            self._chunk_overlap = profile.chunk_overlap
        else:
            self._chunk_size = chunk_size
            self._chunk_overlap = chunk_overlap
        self._splitter = splitter

    def chunk_documents(
        self,
        documents: Sequence[ParsedDocument],
        session_id: str,
    ) -> List[ParsedDocument]:
        """Convert documents into chunk documents with metadata."""
        splitter = self._splitter or self._create_default_splitter()
        chunked_documents: List[ParsedDocument] = []

        for document_index, document in enumerate(documents):
            text_chunks = splitter.split_text(document.page_content)
            base_metadata = dict(document.metadata)
            parent_source = str(base_metadata.get("source", "unknown"))

            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk_metadata = dict(base_metadata)
                chunk_metadata["chunk_index"] = chunk_index
                chunk_metadata["document_index"] = document_index
                chunk_metadata["parent_source"] = parent_source
                chunk_metadata["session_id"] = session_id

                chunked_documents.append(
                    ParsedDocument(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

        logger.info(
            "Session %s completed chunking with %s chunks.",
            session_id,
            len(chunked_documents),
        )
        return chunked_documents

    def _create_default_splitter(self) -> TextSplitter:
        """Create the default RecursiveCharacterTextSplitter."""
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except ImportError as error:
            raise ChunkingException(
                "langchain-text-splitters is not installed; text chunking cannot run."
            ) from error

        return RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
