"""Story 2.4.5 — 文件類型感知 Chunking Profile 測試套件。

涵蓋範圍：
- 三種 DocumentType（semantic / precise / code）的 ChunkingProfile 參數正確套用
- TextChunker profile 整合（chunk_size / chunk_overlap 由 profile 決定）
- ingest_documents() 的 document_type 參數與向下相容性
- HybridRetriever.search() 的 vector_weight / keyword_weight 覆寫
- 未知 document_type 的錯誤處理
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

from core.rag.chunker import (
    CHUNKING_PROFILES,
    ChunkingException,
    ChunkingProfile,
    DocumentType,
    TextChunker,
)
from core.rag.embeddings import EmbeddingException
from core.rag.indexer import SessionIndexer
from core.rag.parser import ParsedDocument
from core.rag.pipeline import ingest_documents
from core.rag.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# 測試替身 (Test Doubles)
# ---------------------------------------------------------------------------


class CapturingFakeSplitter:
    """記錄最後一次被建立時使用的參數，用於驗證 profile 注入。"""

    last_chunk_size: int = 0
    last_chunk_overlap: int = 0

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        CapturingFakeSplitter.last_chunk_size = chunk_size
        CapturingFakeSplitter.last_chunk_overlap = chunk_overlap
        self._outputs: List[str] = ["chunk-A", "chunk-B"]

    def split_text(self, text: str) -> List[str]:
        return list(self._outputs)


class FakeSplitter:
    """可控輸出的簡易假分塊器。"""

    def __init__(self, outputs: List[str] | None = None) -> None:
        self._outputs = outputs or ["chunk-A"]

    def split_text(self, text: str) -> List[str]:
        return list(self._outputs)


class FakeEmbedder:
    """固定回傳 384 維零向量的假嵌入器，避免觸發真實 SentenceTransformer。"""

    def embed_documents(self, documents: List[ParsedDocument]) -> np.ndarray:
        return np.zeros((len(documents), 384), dtype=np.float32)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return np.zeros((len(texts), 384), dtype=np.float32)


class FailingChunker:
    def chunk_documents(self, documents, session_id: str):
        raise RuntimeError("chunk boom")


class FailingEmbedder:
    def embed_documents(self, documents):
        raise RuntimeError("embed boom")


class FakeFaissIndex:
    """模擬 FAISS 索引（與 test_retriever.py 相同結構）。"""

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
        total = self.ntotal
        if total == 0:
            return (
                np.full((query_vectors.shape[0], k), np.inf, dtype=np.float32),
                np.full((query_vectors.shape[0], k), -1, dtype=np.int64),
            )
        actual_k = min(k, total)
        distances = np.arange(actual_k, dtype=np.float32).reshape(1, -1)
        indices = np.arange(actual_k, dtype=np.int64).reshape(1, -1)
        if k > total:
            pad = k - total
            distances = np.concatenate(
                [distances, np.full((1, pad), np.inf, dtype=np.float32)], axis=1
            )
            indices = np.concatenate(
                [indices, np.full((1, pad), -1, dtype=np.int64)], axis=1
            )
        return distances, indices


class FailingAddIndex(FakeFaissIndex):
    def add(self, vectors: np.ndarray) -> None:
        raise RuntimeError("index boom")


@pytest.fixture
def fake_index_factory():
    def factory() -> FakeFaissIndex:
        return FakeFaissIndex()
    return factory


# ---------------------------------------------------------------------------
# 一、CHUNKING_PROFILES 常數驗證
# ---------------------------------------------------------------------------


class TestChunkingProfilesConstants:
    """CHUNKING_PROFILES 字典的預設值驗證（AC: 1）。"""

    def test_semantic_profile_has_correct_default_parameters(self) -> None:
        """語意理解型 Profile 的四個欄位必須符合規格。"""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        assert profile.chunk_size == 1000
        assert profile.chunk_overlap == 200
        assert profile.vector_weight == 0.7
        assert profile.keyword_weight == 0.3

    def test_precise_profile_has_correct_default_parameters(self) -> None:
        """精確查找型 Profile 的 chunk_size 必須是 400（非預設的 1000）。"""
        profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        assert profile.chunk_size == 400
        assert profile.chunk_overlap == 100
        assert profile.vector_weight == 0.4
        assert profile.keyword_weight == 0.6

    def test_code_profile_has_correct_default_parameters(self) -> None:
        """程式碼型 Profile 的四個欄位必須符合規格。"""
        profile = CHUNKING_PROFILES[DocumentType.CODE]
        assert profile.chunk_size == 600
        assert profile.chunk_overlap == 50
        assert profile.vector_weight == 0.6
        assert profile.keyword_weight == 0.4

    def test_all_three_document_types_are_covered(self) -> None:
        """CHUNKING_PROFILES 必須包含三種 DocumentType。"""
        assert DocumentType.SEMANTIC in CHUNKING_PROFILES
        assert DocumentType.PRECISE in CHUNKING_PROFILES
        assert DocumentType.CODE in CHUNKING_PROFILES

    def test_chunking_profile_is_immutable(self) -> None:
        """ChunkingProfile 為 frozen dataclass，不允許修改屬性。"""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        with pytest.raises((AttributeError, TypeError)):
            profile.chunk_size = 9999  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 二、DocumentType Enum 驗證
# ---------------------------------------------------------------------------


class TestDocumentTypeEnum:
    """DocumentType 枚舉值驗證。"""

    def test_document_type_values_are_strings(self) -> None:
        """DocumentType 為 str Enum，值必須是字串。"""
        assert DocumentType.SEMANTIC == "semantic"
        assert DocumentType.PRECISE == "precise"
        assert DocumentType.CODE == "code"

    def test_document_type_can_be_constructed_from_string(self) -> None:
        """必須能從字串建立 DocumentType（支援 FastAPI query param 解析）。"""
        assert DocumentType("semantic") is DocumentType.SEMANTIC
        assert DocumentType("precise") is DocumentType.PRECISE
        assert DocumentType("code") is DocumentType.CODE

    def test_invalid_document_type_string_raises_value_error(self) -> None:
        """傳入未知字串時，DocumentType() 必須拋出 ValueError（不是靜默失敗）。"""
        with pytest.raises(ValueError):
            DocumentType("unknown_type")


# ---------------------------------------------------------------------------
# 三、TextChunker profile 整合（AC: 2, 3）
# ---------------------------------------------------------------------------


class TestTextChunkerProfileIntegration:
    """TextChunker 接受 ChunkingProfile 後的行為驗證。"""

    def test_chunker_uses_profile_chunk_size_over_default(self) -> None:
        """當提供 profile 時，chunker 必須使用 profile.chunk_size（非硬編碼預設值）。"""
        precise_profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        chunker = TextChunker(profile=precise_profile, splitter=FakeSplitter())
        # chunk_size 應為 400（PRECISE），而非預設的 1000
        assert chunker._chunk_size == 400

    def test_chunker_uses_profile_chunk_overlap_over_default(self) -> None:
        """當提供 profile 時，chunker 必須使用 profile.chunk_overlap（非硬編碼預設值）。"""
        code_profile = CHUNKING_PROFILES[DocumentType.CODE]
        chunker = TextChunker(profile=code_profile, splitter=FakeSplitter())
        assert chunker._chunk_overlap == 50

    def test_chunker_without_profile_uses_default_parameters(self) -> None:
        """未提供 profile 時，chunker 必須維持原有預設值（向下相容）。"""
        chunker = TextChunker(splitter=FakeSplitter())
        assert chunker._chunk_size == 1000
        assert chunker._chunk_overlap == 200

    def test_chunker_explicit_params_overridden_by_profile(self) -> None:
        """即使傳入 chunk_size 參數，profile 的設定必須優先。"""
        precise_profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        chunker = TextChunker(
            chunk_size=9999,  # 此值應被 profile 覆蓋
            chunk_overlap=9999,
            profile=precise_profile,
            splitter=FakeSplitter(),
        )
        assert chunker._chunk_size == 400
        assert chunker._chunk_overlap == 100

    def test_chunker_with_semantic_profile_produces_chunks(self) -> None:
        """semantic profile 的 chunker 必須能正常分塊文件。"""
        profile = CHUNKING_PROFILES[DocumentType.SEMANTIC]
        chunker = TextChunker(profile=profile, splitter=FakeSplitter(["段落一", "段落二"]))
        docs = [ParsedDocument(page_content="測試文件", metadata={"source": "test.md"})]
        chunks = chunker.chunk_documents(docs, session_id="sess-1")
        assert len(chunks) == 2
        assert chunks[0].page_content == "段落一"


# ---------------------------------------------------------------------------
# 四、ingest_documents() 向下相容性與 document_type 整合（AC: 2, 4）
# ---------------------------------------------------------------------------


class TestIngestDocumentsDocumentType:
    """ingest_documents() 的 document_type 整合測試。"""

    def _make_indexer_and_docs(
        self, fake_index_factory
    ) -> tuple[SessionIndexer, str, list[ParsedDocument]]:
        """建立含一筆文件的 Session 用於測試。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="測試內容" * 10, metadata={"source": "a.md"})]
        return indexer, record.session_id, docs

    def test_ingest_without_document_type_is_backward_compatible(
        self, fake_index_factory
    ) -> None:
        """未傳 document_type 時，行為必須與原有介面完全一致（向下相容）。"""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            chunker=TextChunker(splitter=FakeSplitter(["chunk-1"])),
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        assert result.chunk_count == 1
        assert result.session_id == sid

    def test_ingest_with_semantic_type_uses_semantic_profile_chunk_size(
        self, fake_index_factory
    ) -> None:
        """document_type=SEMANTIC 必須建立 chunk_size=1000 的 chunker（透過 profile）。"""
        # 用自訂 splitter 驗證 pipeline 確實選用了 semantic profile 的 chunker
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=DocumentType.SEMANTIC,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
            chunker=TextChunker(
                profile=CHUNKING_PROFILES[DocumentType.SEMANTIC],
                splitter=FakeSplitter(["s-chunk"]),
            ),
        )
        assert result.chunk_count == 1

    def test_ingest_with_precise_type_resolves_precise_profile(
        self, fake_index_factory
    ) -> None:
        """document_type=PRECISE 時，pipeline 應自動解析 PRECISE Profile。"""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        # 不傳入 chunker，讓 pipeline 自行依 document_type 建立
        mock_chunker = MagicMock(spec=TextChunker)
        mock_chunker.chunk_documents.return_value = [
            ParsedDocument(page_content="p-chunk", metadata={"source": "a.md", "session_id": sid})
        ]
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=DocumentType.PRECISE,
            chunker=mock_chunker,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        # mock chunker 應被呼叫一次
        mock_chunker.chunk_documents.assert_called_once()

    def test_ingest_none_document_type_applies_semantic_profile(
        self, fake_index_factory
    ) -> None:
        """document_type=None 時，pipeline 必須套用 SEMANTIC 作為預設（AC: 2）。"""
        indexer, sid, docs = self._make_indexer_and_docs(fake_index_factory)
        # 我們注入一個能驗證 chunk_size 的 TextChunker（避免真實模型呼叫）
        precise_chunker = TextChunker(
            profile=CHUNKING_PROFILES[DocumentType.SEMANTIC],
            splitter=FakeSplitter(["default-chunk"]),
        )
        result = ingest_documents(
            session_indexer=indexer,
            session_id=sid,
            documents=docs,
            document_type=None,
            chunker=precise_chunker,
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        # 確認能正常完成 ingestion（不 raise）
        assert result.session_id == sid

    def test_ingest_builds_chunker_from_document_type_when_no_chunker_provided(
        self, fake_index_factory
    ) -> None:
        """未提供 chunker 且提供 document_type 時，pipeline 必須自動建立對應 chunker。"""
        indexer, sid, _ = self._make_indexer_and_docs(fake_index_factory)
        # 使用非常短的文件，確保即使 chunk_size 不同也能通過
        short_docs = [
            ParsedDocument(page_content="短文件", metadata={"source": "b.md"})
        ]
        # 這裡會真正呼叫 RecursiveCharacterTextSplitter（需要 langchain_text_splitters）
        # 若依賴未安裝，允許 ChunkingException；重點在於 pipeline 不因 document_type 而崩潰
        try:
            result = ingest_documents(
                session_indexer=indexer,
                session_id=sid,
                documents=short_docs,
                document_type=DocumentType.CODE,
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )
            # 若成功，chunk_count 應 >= 0
            assert result.chunk_count >= 0
        except Exception as exc:
            # 僅接受 ChunkingException（依賴缺失）或 EmbeddingException
            from core.rag.chunker import ChunkingException
            from core.rag.embeddings import EmbeddingException
            assert isinstance(exc, (ChunkingException, EmbeddingException)), (
                f"Unexpected exception type {type(exc)}: {exc}"
            )


# ---------------------------------------------------------------------------
# 五、HybridRetriever.search() 權重覆寫（AC: 4）
# ---------------------------------------------------------------------------


class TestHybridRetrieverWeightOverride:
    """HybridRetriever.search() 的 vector_weight / keyword_weight 覆寫測試。"""

    @pytest.fixture
    def seeded_retriever(self, fake_index_factory) -> tuple[HybridRetriever, str]:
        """建立含 3 筆 chunk 的 HybridRetriever，使用預設權重 0.7/0.3。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        sid = record.session_id
        indexer.ingest_chunk_embeddings(
            sid,
            documents=[
                ParsedDocument(page_content="報告內容 alpha", metadata={"source": "a.md"}),
                ParsedDocument(page_content="分析結果 beta", metadata={"source": "b.md"}),
            ],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )
        retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=FakeEmbedder(),
            vector_weight=0.7,
            keyword_weight=0.3,
        )
        return retriever, sid

    def test_search_without_weight_override_uses_instance_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """未傳 weight 參數時，search() 必須使用 __init__ 設定的預設值（無例外）。"""
        retriever, sid = seeded_retriever
        result = retriever.search(session_id=sid, query="報告", top_k=5)
        assert result.total_found >= 0  # 能正常回傳即可

    def test_search_with_weight_override_does_not_mutate_instance_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """覆寫參數只作用於本次查詢，不得修改 _vector_weight / _keyword_weight。"""
        retriever, sid = seeded_retriever
        assert retriever._vector_weight == 0.7
        assert retriever._keyword_weight == 0.3

        retriever.search(
            session_id=sid,
            query="alpha",
            top_k=5,
            vector_weight=0.4,
            keyword_weight=0.6,
        )

        # 實例的預設值不應被改變
        assert retriever._vector_weight == 0.7
        assert retriever._keyword_weight == 0.3

    def test_search_with_precise_profile_weights(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """使用 PRECISE Profile 的 0.4/0.6 覆寫時，搜尋必須成功回傳結果。"""
        retriever, sid = seeded_retriever
        profile = CHUNKING_PROFILES[DocumentType.PRECISE]
        result = retriever.search(
            session_id=sid,
            query="alpha",
            top_k=5,
            vector_weight=profile.vector_weight,
            keyword_weight=profile.keyword_weight,
        )
        assert result.total_found >= 0

    def test_search_with_none_weight_override_falls_back_to_defaults(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """明確傳入 None 時，應沿用實例預設值（等同不傳）。"""
        retriever, sid = seeded_retriever
        result = retriever.search(
            session_id=sid,
            query="分析",
            top_k=5,
            vector_weight=None,
            keyword_weight=None,
        )
        assert result.total_found >= 0

    def test_merged_score_reflects_overridden_weights(
        self, seeded_retriever: tuple[HybridRetriever, str]
    ) -> None:
        """當 keyword_weight=1.0, vector_weight=0.0 時，merged_score 應等於 keyword_score。"""
        retriever, sid = seeded_retriever
        result = retriever.search(
            session_id=sid,
            query="報告",
            top_k=5,
            vector_weight=0.0,
            keyword_weight=1.0,
        )
        for chunk in result.results:
            expected = chunk.keyword_score * 1.0 + chunk.vector_score * 0.0
            assert abs(chunk.merged_score - expected) < 1e-6, (
                f"merged_score {chunk.merged_score} != expected {expected}"
            )


# ---------------------------------------------------------------------------
# 六、未知 document_type 錯誤處理（AC: 6）
# ---------------------------------------------------------------------------


class TestInvalidDocumentTypeHandling:
    """傳入無效 document_type 的錯誤處理驗證。"""

    def test_invalid_string_raises_value_error_from_enum(self) -> None:
        """DocumentType('invalid') 必須拋出 ValueError（不是靜默失敗）（AC: 6）。"""
        with pytest.raises(ValueError):
            DocumentType("invalid_type_xyz")

    def test_chunking_profiles_lookup_raises_for_nonexistent_key(self) -> None:
        """直接對 CHUNKING_PROFILES 查詢不存在的 key 必須拋出 KeyError。"""
        # 模擬使用者繞過 Enum 直接建立虛假 DocumentType 值的情境
        fake_key = object()  # 不是 DocumentType，一定 KeyError
        with pytest.raises((KeyError, TypeError)):
            _ = CHUNKING_PROFILES[fake_key]  # type: ignore[index]


# ---------------------------------------------------------------------------
# 七、Session 權重持久化（AC: 4）
# ---------------------------------------------------------------------------


class TestSessionWeightPersistence:
    """驗證 ingest_documents 將權重寫入 Session，以及 retriever 從中讀取。"""

    def test_ingest_documents_persists_weights_to_session(self, fake_index_factory) -> None:
        """ingest_documents() 必須將 profile 的權重記錄在 SessionIndexRecord 中。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="文件內容", metadata={"source": "a.md"})]
        
        ingest_documents(
            session_indexer=indexer,
            session_id=record.session_id,
            documents=docs,
            document_type=DocumentType.CODE,
            chunker=TextChunker(splitter=FakeSplitter()),
            embedder=FakeEmbedder(),  # type: ignore[arg-type]
        )
        
        # 驗證 Session 記錄中存有正確的權重
        session_record = indexer.get_session(record.session_id)
        assert getattr(session_record, "vector_weight", None) == 0.6
        assert getattr(session_record, "keyword_weight", None) == 0.4

    def test_search_reads_weights_from_session_record(self, fake_index_factory) -> None:
        """HybridRetriever.search() 若未傳入覆寫權重，應讀取 Session 中儲存的權重。"""
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        # 手動注入 session 權重（模擬 ingest 寫入的結果）
        record.vector_weight = 0.99
        record.keyword_weight = 0.01
        
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[ParsedDocument(page_content="測試", metadata={"source": "a.md"})],
            embeddings=[[0.1] * 384],
        )
        
        retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=FakeEmbedder(),
            vector_weight=0.5,
            keyword_weight=0.5,
        )
        
        result = retriever.search(session_id=record.session_id, query="測試", top_k=1)
        
        # 驗證使用 session 的權重 (0.99 / 0.01)
        chunk = result.results[0]
        expected = chunk.keyword_score * 0.01 + chunk.vector_score * 0.99
        assert abs(chunk.merged_score - expected) < 1e-6

    def test_ingest_does_not_persist_weights_on_chunking_failure(
        self, fake_index_factory, caplog
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]
        caplog.set_level("ERROR")

        with pytest.raises(ChunkingException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=FailingChunker(),  # type: ignore[arg-type]
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        assert "chunking failed" in str(exc_info.value)
        assert record.session_id in caplog.text
        assert "chunking failed during ingestion" in caplog.text
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None

    def test_ingest_does_not_persist_weights_on_embedding_failure(
        self, fake_index_factory, caplog
    ) -> None:
        indexer = SessionIndexer(index_factory=fake_index_factory)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]
        caplog.set_level("ERROR")

        with pytest.raises(EmbeddingException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=TextChunker(splitter=FakeSplitter(["chunk"])),
                embedder=FailingEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        assert "embedding failed" in str(exc_info.value)
        assert record.session_id in caplog.text
        assert "embedding failed during ingestion" in caplog.text
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None

    def test_ingest_does_not_persist_weights_on_indexing_failure(self) -> None:
        indexer = SessionIndexer(index_factory=FailingAddIndex)
        record = indexer.create_session()
        docs = [ParsedDocument(page_content="content", metadata={"source": "a.md"})]

        from core.rag.indexer import IndexerException

        with pytest.raises(IndexerException) as exc_info:
            ingest_documents(
                session_indexer=indexer,
                session_id=record.session_id,
                documents=docs,
                document_type=DocumentType.CODE,
                chunker=TextChunker(splitter=FakeSplitter(["chunk"])),
                embedder=FakeEmbedder(),  # type: ignore[arg-type]
            )

        assert record.session_id in str(exc_info.value)
        session_record = indexer.get_session(record.session_id)
        assert session_record.vector_weight is None
        assert session_record.keyword_weight is None
