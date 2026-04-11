"""測試文件解析器模組。"""

import os
from unittest.mock import mock_open

import pytest

from core.rag.parser import MarkdownParser, ParsedDocument, ParserException, PdfParser


@pytest.fixture
def temp_markdown_file(tmp_path):
    """建立暫存的 Markdown 檔案供測試使用。"""
    file_path = tmp_path / "test_doc.md"
    content = "# 測試文件標題\n\n這是一份用於測試 Markdown 解析器的暫存文件。\n包含了一些段落與內容。"
    file_path.write_text(content, encoding="utf-8")
    return str(file_path)


@pytest.fixture
def temp_pdf_file():
    """提供模擬的 PDF 檔案路徑。"""
    return "mocked_pdf.pdf"


def test_markdown_parser_success(temp_markdown_file):
    """Markdown 解析器應正確提取標題與內容。"""
    parser = MarkdownParser()

    documents = parser.parse(temp_markdown_file)

    assert len(documents) == 1
    document = documents[0]
    assert isinstance(document, ParsedDocument)
    assert document.metadata["source"] == "test_doc.md"
    assert document.metadata["title"] == "測試文件標題"
    assert "這是一份用於測試 Markdown 解析器的暫存文件" in document.page_content


def test_markdown_parser_file_not_found():
    """找不到 Markdown 檔案時應拋出例外。"""
    parser = MarkdownParser()

    with pytest.raises(ParserException, match="找不到檔案"):
        parser.parse("non_existent_file.md")


def test_pdf_parser_success(temp_pdf_file, monkeypatch):
    """PDF 解析器應正確提取頁面內容與頁碼。"""
    parser = PdfParser()

    monkeypatch.setattr(os.path, "exists", lambda path: True)

    class MockPage:
        def extract_text(self):
            return "這是第一頁的測試內容"

    class MockReader:
        def __init__(self, *args, **kwargs):
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
    assert "這是第一頁的測試內容" in document.page_content


def test_pdf_parser_file_not_found():
    """找不到 PDF 檔案時應拋出例外。"""
    parser = PdfParser()

    with pytest.raises(ParserException, match="找不到檔案"):
        parser.parse("non_existent_file.pdf")
