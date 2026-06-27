"""PDF and Markdown document parsers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from core.log import logger

try:
    import pypdf
except ImportError:
    pypdf = None


class ParserException(Exception):
    """Document parsing error."""


class ParsedDocument(BaseModel):
    """Parsed document content and metadata."""

    page_content: str = Field(description="Parsed text content.")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Document source, page, title, and other metadata.",
    )


class BaseParser(ABC):
    """Shared parser interface."""

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedDocument]:
        """Parse a file into a list of ParsedDocument objects."""


class PdfParser(BaseParser):
    """Parse PDF files."""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """Parse a PDF file and emit one document per page with text."""
        if not os.path.exists(file_path):
            logger.error("PDF file not found: %s", file_path)
            raise ParserException(f"File not found: {file_path}")

        if pypdf is None:
            logger.error("pypdf is not installed.")
            raise ParserException("Install pypdf before parsing PDF files.")

        logger.info("Starting PDF parse: %s", file_path)
        documents: List[ParsedDocument] = []

        try:
            with open(file_path, "rb") as file:
                reader = pypdf.PdfReader(file)
                source_name = os.path.basename(file_path)

                for index, page in enumerate(reader.pages):
                    text = page.extract_text()
                    if text:
                        documents.append(
                            ParsedDocument(
                                page_content=text,
                                metadata={"source": source_name, "page": index + 1},
                            )
                        )
        except Exception as error:
            logger.error("PDF parsing failed: %s", error)
            raise ParserException(f"PDF parsing failed: {error}") from error

        logger.info("PDF parsing completed with %s text pages.", len(documents))
        return documents


class MarkdownParser(BaseParser):
    """Parse Markdown files."""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """Parse Markdown while preserving full text and the first H1 title."""
        if not os.path.exists(file_path):
            logger.error("Markdown file not found: %s", file_path)
            raise ParserException(f"File not found: {file_path}")

        logger.info("Starting Markdown parse: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                source_name = os.path.basename(file_path)
                title = source_name

                for line in content.splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break

                document = ParsedDocument(
                    page_content=content,
                    metadata={"source": source_name, "title": title},
                )
        except Exception as error:
            logger.error("Markdown parsing failed: %s", error)
            raise ParserException(f"Markdown parsing failed: {error}") from error

        logger.info("Markdown parsing completed.")
        return [document]


class TextFileParser(BaseParser):
    """Parse plain text files."""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """Parse a .txt file while preserving full text."""
        if not os.path.exists(file_path):
            logger.error("Text file not found: %s", file_path)
            raise ParserException(f"File not found: {file_path}")

        logger.info("Starting text file parse: %s", file_path)

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                source_name = os.path.basename(file_path)

                document = ParsedDocument(
                    page_content=content,
                    metadata={"source": source_name},
                )
        except UnicodeDecodeError as error:
            logger.error("Text file decoding failed (expected UTF-8): %s", error)
            raise ParserException(
                f"Text file decoding failed for {file_path}. Only UTF-8 is supported."
            ) from error
        except Exception as error:
            logger.error("Text file parsing failed: %s", error)
            raise ParserException(f"Text file parsing failed: {error}") from error

        logger.info("Text file parsing completed.")
        return [document]
