"""提供 Session 範圍的混合檢索能力（向量 + 關鍵字）。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol, Sequence

from core.log import logger


class RetrieverException(Exception):
    """混合檢索相關錯誤。"""


class EmbedderProtocol(Protocol):
    """定義嵌入器所需最小介面。"""

    def embed_texts(self, texts: Sequence[str]) -> Any:
        """將文字轉為向量矩陣。"""


class IndexerProtocol(Protocol):
    """定義 SessionIndexer 所需最小介面。"""

    def get_session(self, session_id: str) -> Any:
        """取得 Session 記錄。"""

    def get_chunk_id_by_ordinal(self, session_id: str, ordinal: int) -> str:
        """依 ordinal 取得 chunk_id。"""

    def get_chunk_document(self, session_id: str, chunk_id: str) -> Any:
        """取得 chunk 的 ParsedDocument。"""

    def get_chunk_metadata(self, session_id: str, chunk_id: str) -> Dict[str, Any]:
        """取得 chunk 的 metadata。"""

    def list_chunk_documents(self, session_id: str) -> List[Any]:
        """列出 Session 所有 chunk 的 ParsedDocument。"""

    def list_vector_chunk_ids(self, session_id: str) -> List[str]:
        """列出 Session 的 vector ordinal 映射。"""


@dataclass(slots=True)
class RetrievedChunk:
    """單一檢索結果，包含分數與 citation metadata。"""

    chunk_id: str
    page_content: str
    metadata: Dict[str, Any]
    vector_score: float
    keyword_score: float
    merged_score: float
    rank: int


@dataclass(slots=True)
class HybridSearchResult:
    """混合檢索的完整回傳結構。"""

    query: str
    session_id: str
    results: List[RetrievedChunk]
    total_found: int


# ---------------------------------------------------------------------------
# 預設參數
# ---------------------------------------------------------------------------

DEFAULT_TOP_K = 10
DEFAULT_VECTOR_WEIGHT = 0.7
DEFAULT_KEYWORD_WEIGHT = 0.3


class HybridRetriever:
    """結合向量檢索與關鍵字檢索的 Session 範圍混合檢索器。"""

    def __init__(
        self,
        session_indexer: Any,
        embedder: Any,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ) -> None:
        """初始化混合檢索器。

        Args:
            session_indexer: 提供 Session 索引管理的 SessionIndexer 實例。
            embedder: 提供文字嵌入能力的嵌入器（需有 embed_texts 方法）。
            vector_weight: 向量分數在合併時的權重。
            keyword_weight: 關鍵字分數在合併時的權重。
        """
        self._indexer = session_indexer
        self._embedder = embedder
        self._vector_weight = vector_weight
        self._keyword_weight = keyword_weight

    def search(
        self,
        session_id: str,
        query: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> HybridSearchResult:
        """執行混合檢索，回傳去重合併的 Top-K 結果。

        Args:
            session_id: 目標 Session 的唯一識別碼。
            query: 使用者查詢文字。
            top_k: 回傳結果數量上限。

        Returns:
            HybridSearchResult 包含合併排序後的結果清單。

        Raises:
            RetrieverException: 查詢為空、Session 不存在或檢索過程失敗時。
        """
        self._validate_query(query)
        self._validate_session(session_id)

        record = self._indexer.get_session(session_id)

        # top_k 若 <= 0，直接回傳空結果，防免後續給 FAISS 帶來不預期參數
        if top_k <= 0:
            return HybridSearchResult(
                query=query,
                session_id=session_id,
                results=[],
                total_found=0,
            )

        # 空 index / 無 chunk corpus 直接回傳空結果
        if not record.vector_chunk_ids and not record.chunk_map:
            logger.info("Session %s 尚無任何 chunk，回傳空結果。", session_id)
            return HybridSearchResult(
                query=query,
                session_id=session_id,
                results=[],
                total_found=0,
            )

        # 執行兩個檢索分支
        vector_hits = self._vector_search(session_id, record, query, top_k)
        keyword_hits = self._keyword_search(session_id, record, query, top_k)

        # 合併與去重
        merged = self._merge_results(
            session_id=session_id,
            vector_hits=vector_hits,
            keyword_hits=keyword_hits,
            top_k=top_k,
        )

        logger.info(
            "Session %s 混合檢索完成，查詢='%s'，回傳 %s 筆結果。",
            session_id,
            query,
            len(merged),
        )

        return HybridSearchResult(
            query=query,
            session_id=session_id,
            results=merged,
            total_found=len(merged),
        )

    # ------------------------------------------------------------------
    # 內部方法：向量檢索分支
    # ------------------------------------------------------------------

    def _vector_search(
        self,
        session_id: str,
        record: Any,
        query: str,
        top_k: int,
    ) -> Dict[str, float]:
        """透過 FAISS index 執行向量搜尋，回傳 {chunk_id: normalized_score}。"""
        if not record.vector_chunk_ids:
            return {}

        try:
            query_vector = self._embedder.embed_texts([query])
        except Exception as error:
            logger.error("Session %s 查詢向量產生失敗: %s", session_id, error)
            raise RetrieverException(
                f"Session {session_id} 查詢向量產生失敗。"
            ) from error

        actual_k = min(top_k, len(record.vector_chunk_ids))

        try:
            distances, indices = record.index.search(query_vector, actual_k)
        except Exception as error:
            logger.error("Session %s FAISS 搜尋失敗: %s", session_id, error)
            raise RetrieverException(
                f"Session {session_id} FAISS 搜尋失敗。"
            ) from error

        hits: Dict[str, float] = {}
        for i in range(actual_k):
            ordinal = int(indices[0, i])
            distance = float(distances[0, i])

            # FAISS 可能回傳 -1 表示沒有足夠結果
            if ordinal < 0:
                continue

            try:
                chunk_id = self._indexer.get_chunk_id_by_ordinal(session_id, ordinal)
            except Exception:
                logger.warning(
                    "Session %s ordinal %s 無法映射到 chunk_id，跳過。",
                    session_id,
                    ordinal,
                )
                continue

            # L2 距離轉相似度分數：score = 1 / (1 + distance)
            score = 1.0 / (1.0 + distance)
            hits[chunk_id] = score

        return hits

    # ------------------------------------------------------------------
    # 內部方法：關鍵字檢索分支
    # ------------------------------------------------------------------

    def _keyword_search(
        self,
        session_id: str,
        record: Any,
        query: str,
        top_k: int,
    ) -> Dict[str, float]:
        """在 Session chunk 文本上執行簡易 BM25 風格的關鍵字搜尋。"""
        if not record.chunk_map:
            return {}

        query_terms = self._tokenize(query)
        if not query_terms:
            return {}
            
        # 獨一化（去重）查詢詞，避免像是 "天氣 天氣" 造成同一個詞的 IDF 被加倍計算
        unique_query_terms = set(query_terms)

        # 收集所有 chunk 文本
        chunk_entries: List[tuple[str, str]] = []
        for chunk_id, doc in record.chunk_map.items():
            chunk_entries.append((chunk_id, doc.page_content))

        # 計算每個文件的 term 出現頻率
        doc_count = len(chunk_entries)
        doc_freqs: Counter[str] = Counter()
        doc_term_counts: List[tuple[str, Counter[str]]] = []

        for chunk_id, text in chunk_entries:
            terms = self._tokenize(text)
            term_counter = Counter(terms)
            doc_term_counts.append((chunk_id, term_counter))
            for term in set(terms):
                doc_freqs[term] += 1

        # 簡化 BM25 計分
        avg_dl = sum(
            sum(tc.values()) for _, tc in doc_term_counts
        ) / max(doc_count, 1)

        k1 = 1.5
        b = 0.75

        scores: Dict[str, float] = {}
        for chunk_id, term_counter in doc_term_counts:
            doc_len = sum(term_counter.values())
            score = 0.0

            for q_term in unique_query_terms:
                tf = term_counter.get(q_term, 0)
                if tf == 0:
                    continue

                df = doc_freqs.get(q_term, 0)
                idf = math.log((doc_count - df + 0.5) / (df + 0.5) + 1.0)
                tf_norm = (tf * (k1 + 1)) / (
                    tf + k1 * (1 - b + b * doc_len / max(avg_dl, 1))
                )
                score += idf * tf_norm

            if score > 0.0:
                scores[chunk_id] = score

        # 正規化分數到 [0, 1]
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                scores = {cid: s / max_score for cid, s in scores.items()}

        return scores

    # ------------------------------------------------------------------
    # 內部方法：合併與去重
    # ------------------------------------------------------------------

    def _merge_results(
        self,
        session_id: str,
        vector_hits: Dict[str, float],
        keyword_hits: Dict[str, float],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """將兩個檢索分支的結果合併、去重、排序。"""
        all_chunk_ids = set(vector_hits.keys()) | set(keyword_hits.keys())

        if not all_chunk_ids:
            return []

        merged_entries: List[tuple[str, float, float, float]] = []
        for chunk_id in all_chunk_ids:
            v_score = vector_hits.get(chunk_id, 0.0)
            k_score = keyword_hits.get(chunk_id, 0.0)
            m_score = (
                self._vector_weight * v_score + self._keyword_weight * k_score
            )
            merged_entries.append((chunk_id, v_score, k_score, m_score))

        # 排序：merged_score 降序，平手時 chunk_id 升序（保持 deterministic）
        merged_entries.sort(key=lambda e: (-e[3], e[0]))

        # Top-K 截取
        merged_entries = merged_entries[:top_k]

        # 構建最終結果
        results: List[RetrievedChunk] = []
        for rank, (chunk_id, v_score, k_score, m_score) in enumerate(
            merged_entries, start=1
        ):
            try:
                doc = self._indexer.get_chunk_document(session_id, chunk_id)
                metadata = dict(doc.metadata)
            except Exception:
                logger.warning(
                    "Session %s chunk %s 無法取得文件資料，跳過。",
                    session_id,
                    chunk_id,
                )
                continue

            results.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    page_content=doc.page_content,
                    metadata=metadata,
                    vector_score=v_score,
                    keyword_score=k_score,
                    merged_score=m_score,
                    rank=rank,
                )
            )

        return results

    # ------------------------------------------------------------------
    # 驗證輔助方法
    # ------------------------------------------------------------------

    def _validate_query(self, query: str) -> None:
        """驗證查詢不為空。"""
        if not query or not query.strip():
            raise RetrieverException("query 不可為空白。")

    def _validate_session(self, session_id: str) -> None:
        """驗證 Session 存在。"""
        try:
            self._indexer.get_session(session_id)
        except Exception as error:
            raise RetrieverException(
                f"查無 session: {session_id}"
            ) from error

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """將文字拆分為可用於關鍵字比對的 token 列表。

        支援中文字元逐字拆分與英數字連續序列。
        """
        tokens: List[str] = []

        # CJK 字元逐字拆分
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                tokens.append(char)

        # 英數字連續序列
        ascii_tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
        tokens.extend(ascii_tokens)

        return tokens
