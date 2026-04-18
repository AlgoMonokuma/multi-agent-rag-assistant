"""提供文字嵌入功能。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from core.log import logger
from core.rag.parser import ParsedDocument


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384


class EmbeddingException(Exception):
    """文字嵌入相關錯誤。"""


class SentenceTransformerEmbedder:
    """封裝 sentence-transformers 的文字嵌入流程。"""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        expected_dimension: int = DEFAULT_EMBEDDING_DIMENSION,
        model: Any | None = None,
    ) -> None:
        """初始化嵌入器。"""
        self._model_name = model_name
        self._expected_dimension = expected_dimension
        self._model = model

    def embed_documents(self, documents: Sequence[ParsedDocument]) -> np.ndarray:
        """將 Chunk 文件轉為 float32 向量矩陣。"""
        texts = [document.page_content for document in documents]
        return self.embed_texts(texts)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """將多段文字轉為 float32 向量矩陣。"""
        model = self._model or self._load_model()

        try:
            vectors = model.encode(list(texts))
        except Exception as error:
            logger.error("產生嵌入向量失敗: %s", error)
            raise EmbeddingException("產生嵌入向量失敗。") from error

        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2:
            raise EmbeddingException("嵌入向量必須為二維矩陣。")

        if matrix.shape[1] != self._expected_dimension:
            raise EmbeddingException(
                f"嵌入維度錯誤，預期 {self._expected_dimension}，實際為 {matrix.shape[1]}。"
            )

        logger.info("已產生 %s 筆、維度 %s 的嵌入向量。", matrix.shape[0], matrix.shape[1])
        return matrix

    def _load_model(self) -> Any:
        """Lazy-load 預設的 sentence-transformers 模型。"""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingException(
                "尚未安裝 sentence-transformers，無法產生嵌入向量。"
            ) from error

        self._model = SentenceTransformer(self._model_name)
        return self._model
