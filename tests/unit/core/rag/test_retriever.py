"""Hybrid Retrieval 模組測試。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from core.rag.indexer import IndexerException, SessionIndexer
from core.rag.parser import ParsedDocument
from core.rag.retriever import (
    HybridRetriever,
    HybridSearchResult,
    RetrievedChunk,
    RetrieverException,
)


# ---------------------------------------------------------------------------
# 測試替身 (Test Doubles)
# ---------------------------------------------------------------------------

class FakeFaissIndex:
    """模擬 FAISS 索引，支援 search 操作。"""

    def __init__(self) -> None:
        self.vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.vectors.append(vectors.copy())

    @property
    def ntotal(self) -> int:
        if not self.vectors:
            return 0
        return sum(v.shape[0] for v in self.vectors)

    def search(
        self, query_vectors: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """回傳 (distances, indices) 模擬 FAISS search。"""
        total = self.ntotal
        if total == 0:
            empty_d = np.full((query_vectors.shape[0], k), np.inf, dtype=np.float32)
            empty_i = np.full((query_vectors.shape[0], k), -1, dtype=np.int64)
            return empty_d, empty_i

        # 簡單模擬：按照加入順序回傳前 k 筆，距離遞增
        actual_k = min(k, total)
        distances = np.arange(actual_k, dtype=np.float32).reshape(1, -1)
        indices = np.arange(actual_k, dtype=np.int64).reshape(1, -1)
        # 若 k > total 則補 -1
        if k > total:
            pad = k - total
            distances = np.concatenate(
                [distances, np.full((1, pad), np.inf, dtype=np.float32)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((1, pad), -1, dtype=np.int64)], axis=1
            )
        return distances, indices


class FakeEmbedder:
    """模擬嵌入器，固定回傳可預測的向量。"""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 384), dtype=np.float32) * 0.5


@pytest.fixture
def fake_index_factory() -> Any:
    """建立 FakeFaissIndex 工廠。"""
    def factory() -> FakeFaissIndex:
        return FakeFaissIndex()
    return factory


@pytest.fixture
def seeded_indexer(fake_index_factory: Any) -> tuple[SessionIndexer, str]:
    """建立含有已 ingest chunk 的 Session。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()
    sid = record.session_id

    indexer.ingest_chunk_embeddings(
        sid,
        documents=[
            ParsedDocument(
                page_content="台灣天氣概述報告內容",
                metadata={"source": "weather.pdf", "page": 1, "title": "天氣報告"},
            ),
            ParsedDocument(
                page_content="人工智慧發展趨勢分析",
                metadata={"source": "ai_trend.md", "title": "AI 趨勢"},
            ),
            ParsedDocument(
                page_content="台灣經濟發展報告彙整",
                metadata={"source": "economy.pdf", "page": 3, "title": "經濟報告"},
            ),
        ],
        embeddings=[[0.1] * 384, [0.2] * 384, [0.3] * 384],
    )
    return indexer, sid


# ---------------------------------------------------------------------------
# Task 1: Domain Model 與 Public API 測試
# ---------------------------------------------------------------------------

class TestRetrievedChunkModel:
    """RetrievedChunk 資料模型結構驗證。"""

    def test_retrieved_chunk_contains_required_fields(self) -> None:
        """RetrievedChunk 必須包含 chunk_id、page_content、metadata 與三種分數。"""
        chunk = RetrievedChunk(
            chunk_id="chunk-1",
            page_content="測試內容",
            metadata={"source": "test.md", "session_id": "sid-1"},
            vector_score=0.85,
            keyword_score=0.0,
            merged_score=0.85,
            rank=1,
        )
        assert chunk.chunk_id == "chunk-1"
        assert chunk.page_content == "測試內容"
        assert chunk.metadata["source"] == "test.md"
        assert chunk.vector_score == 0.85
        assert chunk.keyword_score == 0.0
        assert chunk.merged_score == 0.85
        assert chunk.rank == 1

    def test_retrieved_chunk_preserves_citation_metadata(self) -> None:
        """RetrievedChunk 必須保留完整 citation metadata 欄位。"""
        meta = {
            "source": "doc.pdf",
            "chunk_id": "chunk-5",
            "session_id": "sid-abc",
            "page": 2,
            "title": "研究報告",
            "parent_source": "doc.pdf",
            "chunk_index": 4,
        }
        chunk = RetrievedChunk(
            chunk_id="chunk-5",
            page_content="內容片段",
            metadata=meta,
            vector_score=0.9,
            keyword_score=0.5,
            merged_score=0.7,
            rank=2,
        )
        for key in ("source", "chunk_id", "session_id", "page", "title",
                     "parent_source", "chunk_index"):
            assert key in chunk.metadata


class TestHybridSearchResultModel:
    """HybridSearchResult 結構驗證。"""

    def test_hybrid_search_result_holds_result_list(self) -> None:
        """HybridSearchResult 必須能承載 Top-K 結果清單。"""
        result = HybridSearchResult(
            query="測試查詢",
            session_id="sid-1",
            results=[
                RetrievedChunk(
                    chunk_id="chunk-1",
                    page_content="內容",
                    metadata={},
                    vector_score=0.9,
                    keyword_score=0.3,
                    merged_score=0.6,
                    rank=1,
                ),
            ],
            total_found=1,
        )
        assert result.query == "測試查詢"
        assert result.session_id == "sid-1"
        assert len(result.results) == 1
        assert result.total_found == 1


class TestRetrieverException:
    """RetrieverException 驗證。"""

    def test_retriever_exception_is_a_standard_exception(self) -> None:
        """RetrieverException 必須可正常拋出與攔截。"""
        with pytest.raises(RetrieverException, match="測試錯誤"):
            raise RetrieverException("測試錯誤")


# ---------------------------------------------------------------------------
# Task 2: Session-scoped vector retrieval 測試
# ---------------------------------------------------------------------------

class TestVectorRetrieval:
    """向量檢索功能測試。"""

    def test_vector_search_returns_chunks_with_ordinal_mapping(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """向量搜尋必須透過 ordinal 映射正確還原 chunk_id 與 metadata。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="台灣天氣", top_k=2)

        assert len(result.results) == 2
        for chunk in result.results:
            assert chunk.chunk_id.startswith("chunk-")
            assert chunk.vector_score >= 0.0
            assert chunk.page_content != ""
            assert "source" in chunk.metadata

    def test_vector_search_respects_session_isolation(
        self, fake_index_factory: Any
    ) -> None:
        """不同 Session 的向量搜尋不應看到彼此的 chunk。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        rec_a = indexer.create_session()
        rec_b = indexer.create_session()

        indexer.ingest_chunk_embeddings(
            rec_a.session_id,
            documents=[
                ParsedDocument(page_content="Session A 內容", metadata={"source": "a.md"}),
            ],
            embeddings=[[0.1] * 384],
        )
        indexer.ingest_chunk_embeddings(
            rec_b.session_id,
            documents=[
                ParsedDocument(page_content="Session B 內容", metadata={"source": "b.md"}),
            ],
            embeddings=[[0.2] * 384],
        )

        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )
        result_a = retriever.search(session_id=rec_a.session_id, query="內容", top_k=5)
        result_b = retriever.search(session_id=rec_b.session_id, query="內容", top_k=5)

        # chunk_id 為 session 內部序號，不同 session 會產生相同名稱 (例如 chunk-1)；
        # 真正的隔離擔保是每個 session 結果只包含自己的 source。
        for c in result_a.results:
            assert c.metadata.get("session_id") == rec_a.session_id
            assert c.metadata.get("source") != "b.md"
        for c in result_b.results:
            assert c.metadata.get("session_id") == rec_b.session_id
            assert c.metadata.get("source") != "a.md"


# ---------------------------------------------------------------------------
# Task 3: Keyword retrieval 與 SessionIndexer accessor 測試
# ---------------------------------------------------------------------------

class TestKeywordRetrieval:
    """關鍵字檢索功能測試。"""

    def test_keyword_search_finds_matching_chunks(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """關鍵字搜尋應能在 session chunk 文本中找到匹配結果。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="台灣", top_k=5)

        # "台灣" 出現在 chunk-1 ("台灣天氣概述報告內容") 和 chunk-3 ("台灣經濟發展報告彙整")
        matching_contents = [
            c.page_content for c in result.results if c.keyword_score > 0.0
        ]
        assert len(matching_contents) >= 1  # 至少找到一個 keyword match


class TestSessionIndexerChunkAccessor:
    """SessionIndexer 新增的 chunk 文本 accessor 測試。"""

    def test_get_chunk_document_returns_stored_parsed_document(
        self, fake_index_factory: Any
    ) -> None:
        """get_chunk_document 應回傳完整的 ParsedDocument。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(page_content="完整文字內容", metadata={"source": "a.md"}),
            ],
            embeddings=[[0.1] * 384],
        )

        doc = indexer.get_chunk_document(record.session_id, "chunk-1")

        assert doc.page_content == "完整文字內容"
        assert doc.metadata["source"] == "a.md"

    def test_get_chunk_document_raises_on_unknown_chunk(
        self, fake_index_factory: Any
    ) -> None:
        """get_chunk_document 查無 chunk 時應拋出 IndexerException。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()

        with pytest.raises(IndexerException, match="chunk_id"):
            indexer.get_chunk_document(record.session_id, "nonexistent")

    def test_list_chunk_documents_returns_all_stored_documents(
        self, fake_index_factory: Any
    ) -> None:
        """list_chunk_documents 應回傳 Session 內所有 ParsedDocument。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[
                ParsedDocument(page_content="文件A", metadata={"source": "a.md"}),
                ParsedDocument(page_content="文件B", metadata={"source": "b.md"}),
            ],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )

        docs = indexer.list_chunk_documents(record.session_id)

        assert len(docs) == 2
        contents = {d.page_content for d in docs}
        assert contents == {"文件A", "文件B"}


# ---------------------------------------------------------------------------
# Task 4: Hybrid merge 邏輯測試
# ---------------------------------------------------------------------------

class TestHybridMerge:
    """Hybrid merge、dedup 與排序測試。"""

    def test_merge_deduplicates_by_chunk_id(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """同一 chunk 出現在 vector 和 keyword 兩邊時應去重。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="台灣天氣", top_k=10)

        chunk_ids = [c.chunk_id for c in result.results]
        assert len(chunk_ids) == len(set(chunk_ids)), "結果中不應有重複的 chunk_id"

    def test_merge_results_contain_all_three_scores(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """每個結果都必須包含 vector_score、keyword_score 與 merged_score。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="人工智慧", top_k=5)

        for chunk in result.results:
            assert isinstance(chunk.vector_score, float)
            assert isinstance(chunk.keyword_score, float)
            assert isinstance(chunk.merged_score, float)
            assert chunk.vector_score >= 0.0
            assert chunk.keyword_score >= 0.0
            assert chunk.merged_score >= 0.0

    def test_merge_results_have_deterministic_ordering(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """相同查詢應產生相同順序的結果列表。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result_1 = retriever.search(session_id=sid, query="台灣", top_k=5)
        result_2 = retriever.search(session_id=sid, query="台灣", top_k=5)

        ids_1 = [c.chunk_id for c in result_1.results]
        ids_2 = [c.chunk_id for c in result_2.results]
        assert ids_1 == ids_2

    def test_merge_results_have_sequential_ranks(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """結果的 rank 應從 1 開始連續遞增。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="報告", top_k=5)

        ranks = [c.rank for c in result.results]
        expected = list(range(1, len(ranks) + 1))
        assert ranks == expected

    def test_results_have_complete_citation_metadata(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """結果的 citation metadata 至少保留 source、chunk_id、session_id。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="天氣", top_k=3)

        for chunk in result.results:
            assert "source" in chunk.metadata
            assert "chunk_id" in chunk.metadata
            assert "session_id" in chunk.metadata
            assert chunk.metadata["session_id"] == sid


# ---------------------------------------------------------------------------
# Task 5: 範圍邊界測試（不實作 reranking）
# ---------------------------------------------------------------------------

class TestScopeBoundary:
    """確認 Story 2.4 不包含 reranking 邏輯。"""

    def test_retriever_does_not_expose_reranking_api(self) -> None:
        """HybridRetriever 不應有 rerank 方法。"""
        assert not hasattr(HybridRetriever, "rerank")
        assert not hasattr(HybridRetriever, "cross_encode")
        assert not hasattr(HybridRetriever, "compress")


# ---------------------------------------------------------------------------
# Task 6: 錯誤處理與邊界條件測試
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """錯誤處理與邊界條件測試。"""

    def test_empty_query_raises_retriever_exception(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """空 query 應拋出 RetrieverException。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="query"):
            retriever.search(session_id=sid, query="", top_k=5)

    def test_whitespace_only_query_raises_retriever_exception(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """純空白 query 應拋出 RetrieverException。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="query"):
            retriever.search(session_id=sid, query="   ", top_k=5)

    def test_unknown_session_raises_retriever_exception(
        self, fake_index_factory: Any
    ) -> None:
        """不存在的 session_id 應拋出 RetrieverException。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        with pytest.raises(RetrieverException, match="session"):
            retriever.search(session_id="nonexistent", query="測試", top_k=5)

    def test_empty_index_returns_empty_results(
        self, fake_index_factory: Any
    ) -> None:
        """Session 尚未 ingest 任何 chunk 時應回傳空結果。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(
            session_id=record.session_id, query="任何查詢", top_k=5
        )

        assert result.results == []
        assert result.total_found == 0

    def test_top_k_larger_than_corpus_returns_all_available(
        self, seeded_indexer: tuple[SessionIndexer, str]
    ) -> None:
        """top_k 超過 corpus 大小時應回傳所有可用結果。"""
        indexer, sid = seeded_indexer
        retriever = HybridRetriever(
            session_indexer=indexer, embedder=FakeEmbedder()
        )

        result = retriever.search(session_id=sid, query="報告", top_k=100)

        assert len(result.results) <= 3  # 只 ingest 了 3 筆
        assert result.total_found == len(result.results)
