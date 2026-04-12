"""Session 索引器單元測試。"""

from __future__ import annotations

from typing import Any
from uuid import UUID

import pytest

from core.rag.indexer import IndexerException, SessionIndexer
from core.rag.parser import ParsedDocument


class FakeFaissIndex:
    """模擬 FAISS 索引實例。"""

    def __init__(self, label: str) -> None:
        self.label = label


@pytest.fixture
def fake_index_factory() -> Any:
    """建立可追蹤的假索引工廠。"""

    counter = {"value": 0}

    def factory() -> FakeFaissIndex:
        counter["value"] += 1
        return FakeFaissIndex(label=f"index-{counter['value']}")

    return factory


def test_create_session_generates_uuid_and_initial_record(
    fake_index_factory: Any,
) -> None:
    """建立 Session 時應產生 UUID 並初始化索引與時間戳記。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)

    record = indexer.create_session()

    assert UUID(record.session_id)
    assert record.index.label == "index-1"
    assert record.created_at.tzinfo is not None
    assert record.chunk_map == {}
    assert record.metadata_map == {}


def test_session_data_is_isolated_between_sessions(
    fake_index_factory: Any,
) -> None:
    """不同 Session 的索引與 metadata 應彼此隔離。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    first = indexer.create_session()
    second = indexer.create_session()

    stored_chunk_ids = indexer.store_documents(
        first.session_id,
        [
            ParsedDocument(
                page_content="文件一內容",
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
    """外部 metadata 不應覆蓋索引器產生的 session 與 chunk 識別欄位。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()

    stored_chunk_ids = indexer.store_documents(
        record.session_id,
        [
            ParsedDocument(
                page_content="覆蓋保護測試",
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
    """清理 Session 後應移除索引與 metadata，且後續存取會失敗。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)
    record = indexer.create_session()
    indexer.store_documents(
        record.session_id,
        [
            ParsedDocument(
                page_content="待清理內容",
                metadata={"source": "cleanup.md"},
            )
        ],
    )

    indexer.cleanup_session(record.session_id)

    with pytest.raises(IndexerException, match=record.session_id):
        indexer.get_session(record.session_id)

    with pytest.raises(IndexerException, match=record.session_id):
        indexer.list_chunk_metadata(record.session_id)


def test_unknown_session_operations_raise_indexer_exception(
    fake_index_factory: Any,
) -> None:
    """不存在的 Session 存取應回傳一致錯誤。"""
    indexer = SessionIndexer(index_factory=fake_index_factory)

    with pytest.raises(IndexerException, match="missing-session"):
        indexer.store_documents("missing-session", [])

    with pytest.raises(IndexerException, match="missing-session"):
        indexer.cleanup_session("missing-session")
