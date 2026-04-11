"""文件解析器模組，支援 PDF 與 Markdown 格式。"""

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
    """解析器專用的自訂例外事件。"""


class ParsedDocument(BaseModel):
    """解析後標準化的文件輸出格式。"""

    page_content: str = Field(description="解析後的純文字內容")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="來源檔案的屬性與中繼資料",
    )


class BaseParser(ABC):
    """文件解析器的基礎介面。"""

    @abstractmethod
    def parse(self, file_path: str) -> List[ParsedDocument]:
        """解析給定檔案並回傳標準化的文件列表。"""


class PdfParser(BaseParser):
    """實作 PDF 檔案的解析器。"""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """解析 PDF 檔案，提取每一頁的文字與中繼資料。"""
        if not os.path.exists(file_path):
            logger.error(f"找不到檔案：{file_path}")
            raise ParserException(f"找不到檔案：{file_path}")

        if pypdf is None:
            logger.error("尚未安裝 pypdf 套件。")
            raise ParserException("解析 PDF 需要安裝 pypdf。")

        logger.info(f"開始解析 PDF 檔案：{file_path}")
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
            logger.error(f"解析 PDF 檔案時發生錯誤：{error}")
            raise ParserException(
                f"解析 PDF 時發生非預期錯誤：{error}"
            ) from error

        logger.info(f"PDF 檔案解析完成，共提取 {len(documents)} 頁內容。")
        return documents


class MarkdownParser(BaseParser):
    """實作 Markdown 檔案的解析器。"""

    def parse(self, file_path: str) -> List[ParsedDocument]:
        """解析 Markdown 檔案，將整份文件視為單一區塊提取。"""
        if not os.path.exists(file_path):
            logger.error(f"找不到檔案：{file_path}")
            raise ParserException(f"找不到檔案：{file_path}")

        logger.info(f"開始解析 Markdown 檔案：{file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                source_name = os.path.basename(file_path)

                # 優先使用第一個 H1 作為標題，否則回退為檔名。
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
            logger.error(f"解析 Markdown 檔案時發生錯誤：{error}")
            raise ParserException(
                f"解析 Markdown 時發生非預期錯誤：{error}"
            ) from error

        logger.info("Markdown 檔案解析完成。")
        return [document]
