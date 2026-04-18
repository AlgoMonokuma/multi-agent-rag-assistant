"""Session 索引器與 ingestion 管線測試。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import numpy as np
import pytest

from core.rag.indexer import IndexerException, SessionIndexer
from core.rag.parser import ParsedDocument
from core.rag.pipeline import ingest_documents


class FakeFaissIndex:
    """模擬 FAISS 索引實例。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.added_vectors: list[np.ndarray] = []

    def add(self, vectors: np.ndarray) -> None:
        self.added_vectors.append(vectors.copy())


@pytest.fixture
def fake_index_factory() -> Any:
    """建立可觀察 add 行為的假索引工廠。"""

    counter = {"value": 0}

    def factory() -> FakeFaissIndex:
        counter["value"] += 1
        return FakeFaissIndex(label=f"index-{counter['value']}")

    return factory


def test_create_session_generates_uuid_and_initial_record(
    fake_index_factory: Any,
) -> None:
    """建立 Session 時應初始化空的索引記錄。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)

    record = indexer.create_session()

    assert UUID(record.session_id)
    assert record.index.label == "index-1"
    assert record.created_at.tzinfo is not None
    assert record.chunk_map == {}
    assert record.metadata_map == {}
    assert record.vector_chunk_ids == []


def test_session_data_is_isolated_between_sessions(
    fake_index_factory: Any,
) -> None:
    """不同 Session 的 metadata 應完全隔離。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    first = indexer.create_session()
    second = indexer.create_session()

    stored_chunk_ids = indexer.store_documents(
        first.session_id,
        [
            ParsedDocument(
                page_content="文件片段",
                metadata={"source": "doc-a.md", "page": 1},
            )
        ],
    )

    assert first.index is not second.index
    assert stored_chunk_ids == ["chunk-1"]
    assert indexer.list_chunk_metadata(first.session_id) == [
        {
            "chunk_id": "chunk-1",
            "session_id": first.session_id,
            "source": "doc-a.md",
            "page": 1,
        }
    ]
    assert indexer.list_chunk_metadata(second.session_id) == []

    with pytest.raises(IndexerException, match="chunk_id"):
        indexer.get_chunk_metadata(second.session_id, "chunk-1")


def test_store_documents_preserves_indexer_owned_metadata_fields(
    fake_index_factory: Any,
) -> None:
    """外部 metadata 不應覆蓋索引器擁有的識別欄位。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    stored_chunk_ids = indexer.store_documents(
        record.session_id,
        [
            ParsedDocument(
                page_content="測試文件",
                metadata={
                    "session_id": "forged-session",
                    "chunk_id": "forged-chunk",
                    "source": "guard.md",
                },
            )
        ],
    )

    assert stored_chunk_ids == ["chunk-1"]
    assert indexer.get_chunk_metadata(record.session_id, "chunk-1") == {
        "chunk_id": "chunk-1",
        "session_id": record.session_id,
        "source": "guard.md",
    }


def test_cleanup_session_removes_index_and_metadata(
    fake_index_factory: Any,
) -> None:
    """清除 Session 後不應再查得到任何資料。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()
    indexer.store_documents(
        record.session_id,
        [
            ParsedDocument(
                page_content="需要清理的內容",
                metadata={"source": "cleanup.md"},
            )
        ],
    )

    indexer.cleanup_session(record.session_id)

    with pytest.raises(IndexerException, match=record.session_id):
        indexer.get_session(record.session_id)

    with pytest.raises(IndexerException, match=record.session_id):
        indexer.list_chunk_metadata(record.session_id)


def test_ingest_chunk_embeddings_appends_vectors_and_tracks_ordinals(
    fake_index_factory: Any,
) -> None:
    """寫入向量後應同步保存 ordinal 到 chunk 的映射。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    chunk_ids = indexer.ingest_chunk_embeddings(
        record.session_id,
        documents=[
            ParsedDocument(page_content="第一段", metadata={"source": "doc-a.md"}),
            ParsedDocument(page_content="第二段", metadata={"source": "doc-b.md"}),
        ],
        embeddings=[[0.1] * 384, [0.2] * 384],
    )

    assert chunk_ids == ["chunk-1", "chunk-2"]
    assert record.index.added_vectors[0].dtype == np.float32
    assert record.index.added_vectors[0].shape == (2, 384)
    assert indexer.list_vector_chunk_ids(record.session_id) == ["chunk-1", "chunk-2"]
    assert indexer.get_chunk_id_by_ordinal(record.session_id, 1) == "chunk-2"


def test_ingest_chunk_embeddings_supports_append_semantics(
    fake_index_factory: Any,
) -> None:
    """同一 Session 多次 ingestion 應以 append 方式累積。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    first_chunk_ids = indexer.ingest_chunk_embeddings(
        record.session_id,
        documents=[ParsedDocument(page_content="第一批", metadata={"source": "a.md"})],
        embeddings=[[0.1] * 384],
    )
    second_chunk_ids = indexer.ingest_chunk_embeddings(
        record.session_id,
        documents=[ParsedDocument(page_content="第二批", metadata={"source": "b.md"})],
        embeddings=[[0.2] * 384],
    )

    assert first_chunk_ids == ["chunk-1"]
    assert second_chunk_ids == ["chunk-2"]
    assert indexer.list_vector_chunk_ids(record.session_id) == ["chunk-1", "chunk-2"]
    assert indexer.get_chunk_metadata(record.session_id, "chunk-2")["source"] == "b.md"


def test_ingest_chunk_embeddings_rejects_document_embedding_count_mismatch(
    fake_index_factory: Any,
) -> None:
    """文件數與向量數不一致時不應污染 Session 狀態。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    with pytest.raises(IndexerException, match="文件數量與向量數量不一致"):
        indexer.ingest_chunk_embeddings(
            record.session_id,
            documents=[ParsedDocument(page_content="A", metadata={})],
            embeddings=[[0.1] * 384, [0.2] * 384],
        )

    assert indexer.list_chunk_metadata(record.session_id) == []
    assert indexer.list_vector_chunk_ids(record.session_id) == []
    assert record.index.added_vectors == []


def test_get_chunk_id_by_ordinal_rejects_negative_ordinal(
    fake_index_factory: Any,
) -> None:
    """負數 ordinal 不應被當成 Python 反向索引。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    indexer.ingest_chunk_embeddings(
        record.session_id,
        documents=[ParsedDocument(page_content="A", metadata={})],
        embeddings=[[0.1] * 384],
    )

    with pytest.raises(IndexerException, match="ordinal: -1"):
        indexer.get_chunk_id_by_ordinal(record.session_id, -1)


def test_unknown_session_operations_raise_indexer_exception(
    fake_index_factory: Any,
) -> None:
    """未知 Session 的操作應回傳明確錯誤。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)

    with pytest.raises(IndexerException, match="missing-session"):
        indexer.store_documents("missing-session", [])

    with pytest.raises(IndexerException, match="missing-session"):
        indexer.cleanup_session("missing-session")


class FakeChunker:
    """模擬分塊器。"""

    def chunk_documents(
        self,
        documents: list[ParsedDocument],
        session_id: str,
    ) -> list[ParsedDocument]:
        return [
            ParsedDocument(
                page_content="chunk-a",
                metadata={
                    "source": "doc-a.md",
                    "chunk_index": 0,
                    "parent_source": "doc-a.md",
                    "session_id": session_id,
                },
            ),
            ParsedDocument(
                page_content="chunk-b",
                metadata={
                    "source": "doc-a.md",
                    "chunk_index": 1,
                    "parent_source": "doc-a.md",
                    "session_id": session_id,
                },
            ),
        ]


class FakeEmbedder:
    """模擬嵌入器。"""

    def embed_documents(self, documents: list[ParsedDocument]) -> np.ndarray:
        return np.asarray([[0.1] * 384, [0.2] * 384], dtype=np.float32)


class EmptyChunker:
    """模擬沒有產出 chunk 的分塊器。"""

    def chunk_documents(
        self,
        documents: list[ParsedDocument],
        session_id: str,
    ) -> list[ParsedDocument]:
        return []


def test_ingest_documents_pipeline_keeps_session_isolation(
    fake_index_factory: Any,
) -> None:
    """完整 ingestion 管線應只寫入指定 Session。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    first = indexer.create_session()
    second = indexer.create_session()

    result = ingest_documents(
        session_indexer=indexer,
        session_id=first.session_id,
        documents=[ParsedDocument(page_content="原文", metadata={"source": "doc-a.md"})],
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
    )

    assert result.chunk_ids == ["chunk-1", "chunk-2"]
    assert result.chunk_count == 2
    assert result.embedding_dimension == 384
    assert indexer.list_vector_chunk_ids(first.session_id) == ["chunk-1", "chunk-2"]
    assert indexer.list_vector_chunk_ids(second.session_id) == []


def test_ingest_documents_returns_empty_result_when_no_chunks_are_produced(
    fake_index_factory: Any,
) -> None:
    """當分塊結果為空時應回傳空結果，而不是拋出底層矩陣錯誤。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    result = ingest_documents(
        session_indexer=indexer,
        session_id=record.session_id,
        documents=[ParsedDocument(page_content="原文", metadata={"source": "doc-a.md"})],
        chunker=EmptyChunker(),
        embedder=FakeEmbedder(),
    )

    assert result.session_id == record.session_id
    assert result.chunk_ids == []
    assert result.chunk_count == 0
    assert result.embedding_dimension == 0
    assert indexer.list_chunk_metadata(record.session_id) == []
    assert indexer.list_vector_chunk_ids(record.session_id) == []
