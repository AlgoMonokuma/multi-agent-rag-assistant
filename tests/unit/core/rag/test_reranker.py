"""測試 CrossEncoderReranker 的功能與整合。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.rag.reranker import CrossEncoderReranker, RerankerException
from core.rag.retriever import HybridSearchResult, HybridRetriever, RetrievedChunk


# ---------------------------------------------------------------------------
# 輔助工廠函數
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    page_content: str,
    rank: int = 1,
    merged_score: float = 0.5,
) -> RetrievedChunk:
    """建立測試用的 RetrievedChunk。"""
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_content=page_content,
        metadata={"source": "test.pdf"},
        vector_score=0.5,
        keyword_score=0.5,
        merged_score=merged_score,
        rank=rank,
        rerank_score=None,
    )


def _make_mock_cross_encoder(scores: list[float]) -> MagicMock:
    """建立回傳固定分數的 mock cross_encoder。"""
    mock = MagicMock()
    mock.predict.return_value = scores
    return mock


# ---------------------------------------------------------------------------
# Task 4.1 - CrossEncoderReranker 初始化測試
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerInit:
    """測試 CrossEncoderReranker 初始化行為。"""

    def test_init_with_injected_cross_encoder(self):
        """注入 mock cross_encoder 時不應嘗試載入真實模型。"""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=mock_ce,
        )
        assert reranker is not None

    def test_init_stores_model_name(self):
        """初始化後應保存 model_name 供日後懶加載使用。"""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=mock_ce,
        )
        assert reranker._model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_init_default_model_name(self):
        """不傳 model_name 時應使用預設值。"""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)
        assert "MiniLM" in reranker._model_name or "ms-marco" in reranker._model_name

    def test_lazy_load_raises_when_sentence_transformers_missing(self):
        """sentence_transformers 未安裝時，懶加載應拋出 RerankerException。"""
        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            cross_encoder=None,
        )
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(RerankerException, match="sentence.transformers"):
                reranker._load_model()


# ---------------------------------------------------------------------------
# Task 4.2 - 空輸入行為
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerEmptyInput:
    """測試空輸入時的行為。"""

    def test_rerank_empty_chunks_returns_empty_list(self):
        """chunks 為空列表時，應直接回傳空列表，不執行推論。"""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="測試查詢", chunks=[], top_n=3)

        assert result == []
        mock_ce.predict.assert_not_called()

    def test_rerank_empty_chunks_does_not_call_model(self):
        """確認 chunks 為空時模型不被呼叫。"""
        mock_ce = _make_mock_cross_encoder([])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        reranker.rerank(query="任何查詢", chunks=[], top_n=5)
        mock_ce.predict.assert_not_called()


# ---------------------------------------------------------------------------
# Task 4.3 - 排序與 top_n 截取
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerSorting:
    """測試 rerank() 的排序與截取行為。"""

    def test_rerank_sorts_by_score_descending(self):
        """rerank 後結果應按 Cross-Encoder 分數降序排列。"""
        chunks = [
            _make_chunk("chunk-1", "低相關文字", rank=1),
            _make_chunk("chunk-2", "高相關文字", rank=2),
            _make_chunk("chunk-3", "中相關文字", rank=3),
        ]
        # 對應 chunk-1, chunk-2, chunk-3 的分數
        scores = [0.1, 0.9, 0.5]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="高相關查詢", chunks=chunks, top_n=3)

        assert len(result) == 3
        # 最高分 0.9 的 chunk-2 應在第一位
        assert result[0].chunk_id == "chunk-2"
        assert result[1].chunk_id == "chunk-3"
        assert result[2].chunk_id == "chunk-1"

    def test_rerank_truncates_to_top_n(self):
        """rerank 應截取前 top_n 個結果。"""
        chunks = [_make_chunk(f"chunk-{i}", f"文字 {i}", rank=i) for i in range(5)]
        scores = [0.1, 0.5, 0.9, 0.3, 0.7]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="查詢", chunks=chunks, top_n=3)

        assert len(result) == 3

    def test_rerank_top_n_larger_than_chunks_returns_all(self):
        """top_n 大於 chunks 數量時，應回傳全部 chunks。"""
        chunks = [_make_chunk(f"chunk-{i}", f"文字 {i}", rank=i) for i in range(2)]
        scores = [0.8, 0.3]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="查詢", chunks=chunks, top_n=10)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Task 4.4 - rerank_score 與 rank 更新
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerScoreUpdate:
    """測試 rerank_score 與 rank 欄位更新。"""

    def test_rerank_fills_rerank_score(self):
        """rerank 後每個 chunk 的 rerank_score 應被填入對應分數。"""
        chunks = [
            _make_chunk("chunk-1", "文字A", rank=1),
            _make_chunk("chunk-2", "文字B", rank=2),
        ]
        scores = [0.75, 0.25]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="查詢", chunks=chunks, top_n=2)

        # chunk-1 分數 0.75 → rank=1；chunk-2 分數 0.25 → rank=2
        assert result[0].chunk_id == "chunk-1"
        assert result[0].rerank_score == pytest.approx(0.75)
        assert result[1].chunk_id == "chunk-2"
        assert result[1].rerank_score == pytest.approx(0.25)

    def test_rerank_updates_rank_to_new_order(self):
        """rerank 後 rank 欄位應反映 re-ranking 後的新排名（從 1 開始）。"""
        chunks = [
            _make_chunk("chunk-1", "低相關", rank=1),
            _make_chunk("chunk-2", "高相關", rank=2),
        ]
        scores = [0.1, 0.9]
        mock_ce = _make_mock_cross_encoder(scores)
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        result = reranker.rerank(query="查詢", chunks=chunks, top_n=2)

        assert result[0].chunk_id == "chunk-2"
        assert result[0].rank == 1
        assert result[1].chunk_id == "chunk-1"
        assert result[1].rank == 2

    def test_rerank_passes_correct_pairs_to_model(self):
        """rerank 應以 [query, chunk.page_content] 為 pair 呼叫 predict。"""
        query = "搜尋問題"
        chunks = [
            _make_chunk("chunk-1", "第一段文字", rank=1),
            _make_chunk("chunk-2", "第二段文字", rank=2),
        ]
        mock_ce = _make_mock_cross_encoder([0.5, 0.8])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        reranker.rerank(query=query, chunks=chunks, top_n=2)

        mock_ce.predict.assert_called_once_with(
            [[query, "第一段文字"], [query, "第二段文字"]]
        )


# ---------------------------------------------------------------------------
# Task 4.5 - 錯誤處理
# ---------------------------------------------------------------------------


class TestCrossEncoderRerankerErrorHandling:
    """測試模型推論失敗時的錯誤處理。"""

    def test_rerank_wraps_model_exception_as_reranker_exception(self):
        """模型推論中拋出的例外應被包裹為 RerankerException。"""
        mock_ce = MagicMock()
        mock_ce.predict.side_effect = RuntimeError("模型推論失敗")
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        chunks = [_make_chunk("chunk-1", "文字", rank=1)]

        with pytest.raises(RerankerException):
            reranker.rerank(query="查詢", chunks=chunks, top_n=1)

    def test_reranker_exception_inherits_from_exception(self):
        """RerankerException 應繼承自 Exception。"""
        assert issubclass(RerankerException, Exception)


# ---------------------------------------------------------------------------
# Task 4.6 - HybridRetriever 整合（有 reranker）
# ---------------------------------------------------------------------------


class TestHybridRetrieverWithReranker:
    """測試 HybridRetriever 整合 CrossEncoderReranker 的行為。"""

    def _make_retriever_with_reranker(
        self, reranker: CrossEncoderReranker, top_k: int = 10, final_top_n: int = 3
    ) -> HybridRetriever:
        """建立注入 reranker 的 HybridRetriever（使用 mock indexer 與 embedder）。"""
        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        return HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=final_top_n,
        )

    def test_search_with_reranker_returns_reranked_results(self):
        """search() 有 reranker 時，results 應反映 Re-Ranking 後的順序（按 rerank_score 降序）。"""
        import numpy as np

        # 建立 mock cross_encoder：分數越高的 chunk 排越前面
        # predict 的分數是按照傳入 pair 的順序決定的
        # 我們讓 chunk_id 包含在 page_content 中，方便識別
        mock_ce = MagicMock()

        def _predict_side_effect(pairs):
            """根據 page_content 的關鍵字分配固定分數。"""
            scores = []
            for pair in pairs:
                content = pair[1]
                if "高相關" in content:
                    scores.append(0.9)
                elif "中相關" in content:
                    scores.append(0.5)
                else:
                    scores.append(0.1)
            return scores

        mock_ce.predict.side_effect = _predict_side_effect
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()

        session_id = "test-session"
        mock_record = MagicMock()
        mock_record.vector_chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]
        mock_record.chunk_map = {
            "chunk-1": MagicMock(page_content="低相關文字", metadata={}),
            "chunk-2": MagicMock(page_content="中相關文字", metadata={}),
            "chunk-3": MagicMock(page_content="高相關文字", metadata={}),
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record

        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3]]),
            np.array([[0, 1, 2]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: [
            "chunk-1", "chunk-2", "chunk-3"
        ][ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=3,
        )

        result = retriever.search(session_id=session_id, query="高相關查詢", top_k=3)

        assert isinstance(result, HybridSearchResult)
        assert len(result.results) <= 3
        # 驗證 rerank_score 已填入且最高分的 chunk 排第一
        assert result.results[0].rerank_score is not None
        assert result.results[0].rerank_score >= result.results[-1].rerank_score
        # 驗證最高相關度的 chunk（包含「高相關」）排第一
        assert "高相關" in result.results[0].page_content

    def test_search_with_reranker_truncates_to_final_top_n(self):
        """search() 有 reranker 時，最終結果應不超過 final_top_n。"""
        import numpy as np

        mock_ce = _make_mock_cross_encoder([0.9, 0.8, 0.7, 0.6, 0.5])
        reranker = CrossEncoderReranker(cross_encoder=mock_ce)

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        session_id = "test-session-2"

        chunk_ids = [f"chunk-{i}" for i in range(1, 6)]
        mock_record = MagicMock()
        mock_record.vector_chunk_ids = chunk_ids
        mock_record.chunk_map = {
            cid: MagicMock(page_content=f"文字{i}", metadata={})
            for i, cid in enumerate(chunk_ids)
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record
        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),
            np.array([[0, 1, 2, 3, 4]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: chunk_ids[ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=reranker,
            final_top_n=2,
        )

        result = retriever.search(session_id=session_id, query="查詢", top_k=5)
        assert len(result.results) <= 2


# ---------------------------------------------------------------------------
# Task 4.7 - 向下相容（無 reranker）
# ---------------------------------------------------------------------------


class TestHybridRetrieverBackwardCompatibility:
    """測試無 reranker 時 HybridRetriever 的向下相容行為。"""

    def test_search_without_reranker_returns_hybrid_results(self):
        """無 reranker 時，search() 應直接回傳 hybrid search 結果，不走 re-ranking。"""
        import numpy as np

        mock_indexer = MagicMock()
        mock_embedder = MagicMock()
        session_id = "compat-session"

        mock_record = MagicMock()
        mock_record.vector_chunk_ids = ["chunk-1", "chunk-2"]
        mock_record.chunk_map = {
            "chunk-1": MagicMock(page_content="文字A", metadata={}),
            "chunk-2": MagicMock(page_content="文字B", metadata={}),
        }
        mock_record.vector_weight = None
        mock_record.keyword_weight = None
        mock_indexer.get_session.return_value = mock_record
        mock_embedder.embed_texts.return_value = np.array([[0.1] * 384])
        mock_record.index.search.return_value = (
            np.array([[0.1, 0.2]]),
            np.array([[0, 1]]),
        )
        mock_indexer.get_chunk_id_by_ordinal.side_effect = lambda sid, ord_: [
            "chunk-1", "chunk-2"
        ][ord_]
        mock_indexer.get_chunk_document.side_effect = lambda sid, cid: mock_record.chunk_map[cid]

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
            reranker=None,  # 明確傳入 None
        )

        result = retriever.search(session_id=session_id, query="查詢", top_k=2)

        assert isinstance(result, HybridSearchResult)
        # 無 reranker 時 rerank_score 應為 None
        for chunk in result.results:
            assert chunk.rerank_score is None

    def test_default_reranker_is_none(self):
        """HybridRetriever 預設 reranker=None，確保向下相容。"""
        mock_indexer = MagicMock()
        mock_embedder = MagicMock()

        retriever = HybridRetriever(
            session_indexer=mock_indexer,
            embedder=mock_embedder,
        )
        assert retriever._reranker is None
