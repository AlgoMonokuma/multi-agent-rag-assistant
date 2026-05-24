"""提供 Cross-Encoder Re-Ranking 能力，對 Hybrid Search 初步結果進行二次精排。"""

from __future__ import annotations

from typing import List, Protocol, Sequence, runtime_checkable

from core.log import logger


class RerankerException(Exception):
    """Re-Ranking 相關錯誤。"""


@runtime_checkable
class CrossEncoderProtocol(Protocol):
    """定義 Cross-Encoder 所需最小介面，方便注入 mock 進行測試。"""

    def predict(self, pairs: List[List[str]]) -> List[float]:
        """接收 [query, passage] pair 列表，回傳對應的相關度分數列表。"""


# 預設 Cross-Encoder 模型名稱（HuggingFace 免費，適合 HF Spaces Demo）
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """使用 Cross-Encoder 模型對初步檢索結果進行語意重排序。

    設計原則：
    - 懶加載（Lazy Loading）：模型不在 __init__ 中強制載入，
      而是在 _load_model() 中按需載入，確保測試環境可跳過真實下載。
    - 依賴注入（DI）：透過 cross_encoder 參數注入 mock，
      讓單元測試無需網路或 GPU。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        cross_encoder: CrossEncoderProtocol | None = None,
    ) -> None:
        """初始化 Cross-Encoder Re-Ranker。

        Args:
            model_name: HuggingFace Cross-Encoder 模型名稱。
            cross_encoder: 可注入的自訂 Cross-Encoder 實例（None 時懶加載）。
        """
        self._model_name = model_name
        self._cross_encoder = cross_encoder

    def _load_model(self) -> CrossEncoderProtocol:
        """懶加載 Cross-Encoder 模型。

        Returns:
            已載入的 CrossEncoder 實例。

        Raises:
            RerankerException: 若 sentence_transformers 未安裝或模型載入失敗。
        """
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except (ImportError, TypeError) as error:
            raise RerankerException(
                "尚未安裝 sentence_transformers，無法載入 Cross-Encoder 模型。"
                " 請執行：uv add sentence-transformers"
            ) from error

        try:
            model = CrossEncoder(self._model_name)
        except Exception as error:
            logger.error("Cross-Encoder 模型載入失敗: %s", error)
            raise RerankerException(
                f"Cross-Encoder 模型 '{self._model_name}' 載入失敗。"
            ) from error

        return model  # type: ignore[return-value]

    def rerank(
        self,
        query: str,
        chunks: Sequence[object],
        top_n: int = 3,
    ) -> list[object]:
        """對初步檢索結果執行 Cross-Encoder Re-Ranking。

        Args:
            query: 使用者的查詢文字。
            chunks: 初步檢索出的 RetrievedChunk 列表（來自 HybridRetriever）。
            top_n: Re-Ranking 後回傳的最大結果數量。

        Returns:
            按 Cross-Encoder 分數降序排列，並截取前 top_n 的新 RetrievedChunk 列表。
            每個 chunk 的 rerank_score 與 rank 欄位均已更新。

        Raises:
            RerankerException: 模型推論過程中發生例外時。
        """
        # 空列表直接回傳，不執行推論（AC: 5）
        if not chunks:
            return []

        # 取得已載入（或注入）的 cross_encoder，載入後存入 instance 變數進行 Cache (Bug B Fix)
        if self._cross_encoder is None:
            self._cross_encoder = self._load_model()
        cross_encoder = self._cross_encoder

        # 組建 [query, passage] pair 列表
        pairs = [[query, chunk.page_content] for chunk in chunks]  # type: ignore[union-attr]

        # 批量推論
        try:
            raw_scores = cross_encoder.predict(pairs)
        except Exception as error:
            logger.error("Cross-Encoder 推論失敗: %s", error)
            raise RerankerException("Cross-Encoder 推論失敗。") from error

        # 將分數與 chunk 配對，依分數降序排列
        scored = sorted(
            zip(raw_scores, chunks),
            key=lambda pair: float(pair[0]),
            reverse=True,
        )

        # 截取 top_n，更新 rerank_score 與 rank
        reranked: list[object] = []
        from dataclasses import replace  # noqa: PLC0415
        for new_rank, (score, chunk) in enumerate(scored[:top_n], start=1):
            updated_chunk = replace(  # type: ignore[call-overload]
                chunk,
                rerank_score=float(score),
                rank=new_rank,
            )
            reranked.append(updated_chunk)

        logger.info(
            "Re-Ranking 完成：輸入 %d 筆，回傳 Top-%d 結果。",
            len(chunks),
            len(reranked),
        )
        return reranked
