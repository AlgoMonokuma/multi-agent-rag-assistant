"""Test behavior."""

from __future__ import annotations

import os
from unittest.mock import mock_open

import pytest

from core.rag.parser import (
    MarkdownParser,
    ParsedDocument,
    ParserException,
    PdfParser,
    TextFileParser,
)


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


def test_pdf_parser_empty_text_raises(
    temp_pdf_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PDFs with no extractable text fail explicitly instead of returning []."""
    parser = PdfParser()

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    class MockPage:
        def __init__(self, text: str | None) -> None:
            self._text = text

        def extract_text(self) -> str | None:
            return self._text

    class MockReader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.pages = [MockPage(None), MockPage("")]

    import core.rag.parser as parser_module

    class MockPypdf:
        PdfReader = MockReader

    monkeypatch.setattr(parser_module, "pypdf", MockPypdf)
    monkeypatch.setattr("builtins.open", mock_open(read_data=b"PDF contents"))

    with pytest.raises(ParserException, match="no extractable text"):
        parser.parse(temp_pdf_file)


def test_text_parser_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = TextFileParser()
    file_path = "test_doc.txt"
    text_content = "This is plain text content."

    monkeypatch.setattr(os.path, "exists", lambda path: path == file_path)
    monkeypatch.setattr(os.path, "getsize", lambda path: 100)
    monkeypatch.setattr(
        "builtins.open",
        mock_open(read_data=text_content),
    )

    documents = parser.parse(file_path)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, ParsedDocument)
    assert document.metadata["source"] == "test_doc.txt"
    assert "This is plain text content" in document.page_content


def test_text_parser_encoding_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = TextFileParser()
    file_path = "binary.txt"

    monkeypatch.setattr(os.path, "exists", lambda path: path == file_path)
    monkeypatch.setattr(os.path, "getsize", lambda path: 100)

    # Mock open to raise UnicodeDecodeError
    def mock_open_encoding_error(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr("builtins.open", mock_open_encoding_error)

    with pytest.raises(ParserException, match="Only UTF-8 is supported"):
        parser.parse(file_path)


def test_text_parser_file_not_found() -> None:
    """Test behavior."""
    parser = TextFileParser()

    with pytest.raises(ParserException, match="File not found"):
        parser.parse("non_existent_file.txt")


def test_text_parser_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = TextFileParser()
    file_path = "huge_file.txt"

    # Mock file size to be 21MB
    monkeypatch.setattr(os.path, "getsize", lambda path: 21 * 1024 * 1024)

    with pytest.raises(ParserException, match="File too large"):
        parser.parse(file_path)


def test_text_parser_empty_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = TextFileParser()
    file_path = "empty.txt"

    monkeypatch.setattr(os.path, "getsize", lambda path: 0)
    monkeypatch.setattr("builtins.open", mock_open(read_data=""))

    documents = parser.parse(file_path)

    assert documents == []


def test_text_parser_permission_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test behavior."""
    parser = TextFileParser()
    file_path = "secret.txt"

    monkeypatch.setattr(os.path, "getsize", lambda path: 100)

    def mock_open_permission_error(*args, **kwargs):
        raise PermissionError("Access denied")

    monkeypatch.setattr("builtins.open", mock_open_permission_error)

    with pytest.raises(ParserException, match="Permission denied"):
        parser.parse(file_path)
