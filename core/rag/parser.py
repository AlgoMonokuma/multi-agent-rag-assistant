"""提供 PDF 與 Markdown 文件解析能力。"""

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
    """文件解析相關錯誤。"""


class ParsedDocument(BaseModel):
    """保存解析後的文件內容與 metadata。"""

    page_content: str = Field(description="解析出的文字內容。")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="文件來源、頁碼、標題等中繼資料。",
    )


class BaseParser(ABC):
    """定義解析器共用介面。"""

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedDocument]:
        """將檔案解析成 ParsedDocument 清單。"""


class PdfParser(BaseParser):
    """解析 PDF 檔案。"""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """解析 PDF，逐頁輸出文件內容。"""
        if not os.path.exists(file_path):
            logger.error("找不到 PDF 檔案: %s", file_path)
            raise ParserException(f"找不到檔案: {file_path}")

        if pypdf is None:
            logger.error("尚未安裝 pypdf。")
            raise ParserException("解析 PDF 前請先安裝 pypdf。")

        logger.info("開始解析 PDF: %s", file_path)
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
            logger.error("PDF 解析失敗: %s", error)
            raise ParserException(f"PDF 解析失敗: {error}") from error

        logger.info("PDF 解析完成，共 %s 頁有文字內容。", len(documents))
        return documents


class MarkdownParser(BaseParser):
    """解析 Markdown 檔案。"""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """解析 Markdown，保留完整文字與標題。"""
        if not os.path.exists(file_path):
            logger.error("找不到 Markdown 檔案: %s", file_path)
            raise ParserException(f"找不到檔案: {file_path}")

        logger.info("開始解析 Markdown: %s", file_path)

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
            logger.error("Markdown 解析失敗: %s", error)
            raise ParserException(f"Markdown 解析失敗: {error}") from error

        logger.info("Markdown 解析完成。")
        return [document]
