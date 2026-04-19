"""文件解析器測試。"""

from __future__ import annotations

import os
from unittest.mock import mock_open

import pytest

from core.rag.parser import MarkdownParser, ParsedDocument, ParserException, PdfParser


@pytest.fixture
def temp_pdf_file() -> str:
    """提供假 PDF 路徑。"""
    return "mocked_pdf.pdf"


def test_markdown_parser_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markdown 解析成功時應保留標題與內容。"""
    parser = MarkdownParser()
    file_path = "test_doc.md"
    markdown_content = "# 測試文件標題\n\n這是一段 Markdown 測試內容。"

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
    assert document.metadata["title"] == "測試文件標題"
    assert "這是一段 Markdown 測試內容" in document.page_content


def test_markdown_parser_file_not_found() -> None:
    """檔案不存在時應回傳明確錯誤。"""
    parser = MarkdownParser()

    with pytest.raises(ParserException, match="找不到檔案"):
        parser.parse("non_existent_file.md")


def test_pdf_parser_success(temp_pdf_file: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """PDF 解析成功時應輸出頁碼與內容。"""
    parser = PdfParser()

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    class MockPage:
        def extract_text(self) -> str:
            return "這是 PDF 測試內容"

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
    assert "這是 PDF 測試內容" in document.page_content


def test_pdf_parser_file_not_found() -> None:
    """PDF 檔案不存在時應回傳明確錯誤。"""
    parser = PdfParser()

    with pytest.raises(ParserException, match="找不到檔案"):
        parser.parse("non_existent_file.pdf")
