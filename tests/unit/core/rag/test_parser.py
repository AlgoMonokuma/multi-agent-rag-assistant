"""Test behavior."""

from __future__ import annotations

import os
from unittest.mock import mock_open

import pytest

from core.rag.parser import MarkdownParser, ParsedDocument, ParserException, PdfParser


@pytest.fixture
def temp_pdf_file() -> str:
    """Test behavior."""
    return "mocked_pdf.pdf"


def test_markdown_parser_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = MarkdownParser()
    file_path = "test_doc.md"
    markdown_content = "# Test Document Title\n\nThis is Markdown test content."

    monkeypatch.setattr(os.path, "exists", lambda path: path == file_path)
    monkeypatch.setattr(
        "builtins.open",
        mock_open(read_data=markdown_content),
    )

    documents = parser.parse(file_path)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, ParsedDocument)
    assert document.metadata["source"] == "test_doc.md"
    assert document.metadata["title"] == "Test Document Title"
    assert "This is Markdown test content" in document.page_content


def test_markdown_parser_file_not_found() -> None:
    """Test behavior."""
    parser = MarkdownParser()

    with pytest.raises(ParserException, match="File not found"):
        parser.parse("non_existent_file.md")


def test_pdf_parser_success(temp_pdf_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = PdfParser()

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    class MockPage:
        def extract_text(self) -> str:
            return "This is PDF test content"

    class MockReader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.pages = [MockPage()]

    import core.rag.parser as parser_module

    class MockPypdf:
        PdfReader = MockReader

    monkeypatch.setattr(parser_module, "pypdf", MockPypdf)
    monkeypatch.setattr("builtins.open", mock_open(read_data=b"PDF contents"))

    documents = parser.parse(temp_pdf_file)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, ParsedDocument)
    assert document.metadata["source"] == temp_pdf_file
    assert document.metadata["page"] == 1
    assert "This is PDF test content" in document.page_content


def test_pdf_parser_file_not_found() -> None:
    """Test behavior."""
    parser = PdfParser()

    with pytest.raises(ParserException, match="File not found"):
        parser.parse("non_existent_file.pdf")
