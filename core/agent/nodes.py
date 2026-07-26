"""Node functions for the LangGraph agent workflow.

Each node accepts an AgentState dict and returns a partial state update.

Current implementation status:
  - researcher_node -> real (HybridRetriever — implemented in Story 3.3)
  - web_search_node -> real (Tavily web search fallback — implemented in Story 3.4)
  - reporter_node   -> real (LLMGenerator / Groq — implemented in Story 3.2)
  - reviewer_node   -> stub (Reviewer LLM / quality gate deferred to Story 3.6)

Rules followed by all nodes:
  1. Accept state: AgentState, return dict of state updates.
  2. Use the shared logger from core.log (not print or logging directly).
  3. Do not raise exceptions for valid state input.
  4. Log a DEBUG message indicating which node ran and key state fields.
"""

from core.agent.state import AgentState
from core.log import get_logger

logger = get_logger(__name__)

# Module-level sentinel for lazy default retriever construction.
# Never call _get_default_retriever() at import time — it loads sentence-transformers
# and faiss eagerly which slows all tests and breaks the lazy-load guarantee.
_default_retriever = None

# Module-level sentinel for lazy default web search client construction.
# Never construct at import time — keeps Tavily SDK out of the import chain.
_default_web_search_client = None

WEB_SEARCH_THRESHOLD = 2
"""researcher_node sets needs_web_search=True when retrieved chunks < this value."""


def _get_default_retriever():
    """Return the module-level default retriever, constructing it lazily on first call.

    Uses a module-level sentinel so the heavy dependency stack (sentence-transformers,
    faiss, cross-encoder) is only initialised once per process lifetime.

    Returns:
        HybridRetriever configured with SentenceTransformerEmbedder,
        SessionIndexer, and CrossEncoderReranker (final_top_n=5).
    """
    global _default_retriever
    if _default_retriever is None:
        from core.rag.embeddings import SentenceTransformerEmbedder  # noqa: PLC0415
        from core.rag.indexer import get_default_session_indexer  # noqa: PLC0415
        from core.rag.reranker import CrossEncoderReranker  # noqa: PLC0415
        from core.rag.retriever import HybridRetriever  # noqa: PLC0415

        embedder = SentenceTransformerEmbedder()
        indexer = get_default_session_indexer()
        reranker = CrossEncoderReranker()
        _default_retriever = HybridRetriever(
            session_indexer=indexer,
            embedder=embedder,
            reranker=reranker,
            final_top_n=5,
        )
    return _default_retriever


def _get_default_web_search_client():
    """Return the module-level Tavily search client, constructing it lazily on first call.

    Returns:
        TavilySearchAPIWrapper configured with TAVILY_API_KEY from settings.

    Raises:
        WebSearchException: If TAVILY_API_KEY is not set in environment.
    """
    global _default_web_search_client
    if _default_web_search_client is None:
        from core.agent.exceptions import WebSearchException  # noqa: PLC0415
        from core.config import settings  # noqa: PLC0415

        if not settings.TAVILY_API_KEY:
            raise WebSearchException(
                "TAVILY_API_KEY is not set. "
                "Set TAVILY_API_KEY in your .env file to enable web search."
            )

        from langchain_tavily import TavilySearchAPIWrapper  # noqa: PLC0415

        _default_web_search_client = TavilySearchAPIWrapper(
            tavily_api_key=settings.TAVILY_API_KEY
        )
    return _default_web_search_client


def researcher_node(state: AgentState, *, _retriever=None) -> dict:
    """Retrieve relevant chunks for the user query.

    Calls HybridRetriever (with CrossEncoderReranker wired inside) using
    session_id and query extracted from state.  Fails open — returns
    retrieved_chunks=[] — on RetrieverException or RerankerException so
    the graph continues to the reporter node.

    Also sets needs_web_search=True when the retrieved chunk count falls below
    WEB_SEARCH_THRESHOLD, triggering the web_search_node fallback.

    Note on re-ranking: CrossEncoderReranker is injected into HybridRetriever
    at construction time (via _get_default_retriever). The retriever calls
    reranker.rerank() internally during search(); there is no separate
    reranker call in this node.

    Args:
        state: Current workflow state containing query and session_id.
        _retriever: Optional pre-built HybridRetriever for test injection.
                    When None, the default retrieval stack is constructed
                    lazily via _get_default_retriever().
                    LangGraph only passes state as a positional arg, so
                    keyword-only injection is safe for production use.

    Returns:
        Partial state update with retrieved_chunks (list[RetrievedChunk]),
        an iteration_count increment of 1, and needs_web_search (bool).
        needs_web_search=True when chunk count < WEB_SEARCH_THRESHOLD or
        on retrieval error (fail-open triggers web search fallback).
    """
    from core.rag.reranker import RerankerException  # noqa: PLC0415
    from core.rag.retriever import RetrieverException  # noqa: PLC0415

    query = state.get("query", "")
    session_id = state.get("session_id", "")

    logger.debug(
        "researcher_node: query=%r session_id=%r",
        query,
        session_id,
    )

    retriever = _retriever or _get_default_retriever()

    try:
        search_result = retriever.search(session_id=session_id, query=query, top_k=10)
        chunks = search_result.results
    except (RetrieverException, RerankerException) as exc:
        logger.error("researcher_node: retrieval failed: %s", exc)
        # Fail-open: empty chunks always triggers web search
        return {"retrieved_chunks": [], "iteration_count": 1, "needs_web_search": True}

    chunk_count = len(chunks)
    needs_web_search = chunk_count < WEB_SEARCH_THRESHOLD
    logger.debug(
        "researcher_node: retrieved %d chunks, needs_web_search=%s",
        chunk_count,
        needs_web_search,
    )

    return {
        "retrieved_chunks": list(chunks),
        "iteration_count": 1,
        "needs_web_search": needs_web_search,
    }


def web_search_node(state: AgentState, *, _client=None) -> dict:
    """Perform live web search when local context is insufficient.

    Called only when researcher_node sets needs_web_search=True.
    Fails open (returns web_search_results=[]) on any error so the
    graph continues to reporter_node.

    Args:
        state: Current workflow state containing query.
        _client: Optional pre-built TavilySearchAPIWrapper for test injection.
                 When None, the default client is constructed lazily.
                 LangGraph only passes state positionally — keyword-only is safe.

    Returns:
        Partial state update with web_search_results (list[dict]).
        Each dict contains: url (str), content (str), score (float).
    """
    from core.agent.exceptions import WebSearchException  # noqa: PLC0415

    query = state.get("query", "")
    logger.debug("web_search_node: query=%r", query)

    try:
        client = _client or _get_default_web_search_client()
        raw_results = client.results(query=query, max_results=5)
        results = [
            {
                "url": r.get("url", ""),
                "content": r.get("content", ""),
                "score": r.get("score", 0.0),
            }
            for r in (raw_results or [])
            if isinstance(r, dict)
        ]
    except WebSearchException as exc:
        logger.error("web_search_node: configuration error: %s", exc)
        return {"web_search_results": []}
    except Exception as exc:  # noqa: BLE001
        logger.error("web_search_node: search failed: %s", exc)
        return {"web_search_results": []}

    logger.debug("web_search_node: got %d web results", len(results))
    return {"web_search_results": results}


def reporter_node(state: AgentState) -> dict:
    """Generate a draft answer from retrieved chunks using LLMGenerator.

    Calls LLMGenerator (Groq) to produce a grounded answer from the
    retrieved and re-ranked chunks stored in state. Also merges any
    web_search_results from web_search_node into the LLM prompt context.

    Args:
        state: Current workflow state containing retrieved_chunks, query,
               session_id, and optionally web_search_results.

    Returns:
        Partial state update with draft_answer (str) and citations (list[dict]).
        Citations are stored as plain dicts for JSON-compatibility with
        downstream nodes.
    """
    from core.rag.generator import GeneratorException, LLMGenerator  # lazy import

    query = state.get("query", "")
    chunks = state.get("retrieved_chunks") or []
    web_results = state.get("web_search_results") or []  # NEW — Story 3.4
    session_id = state.get("session_id", "")

    logger.debug(
        "reporter_node: chunk_count=%d web_result_count=%d session_id=%r",
        len(chunks),
        len(web_results),
        session_id,
    )

    generator = LLMGenerator()
    try:
        result = generator.generate(
            query=query,
            chunks=chunks,
            session_id=session_id,
            web_context=web_results,  # NEW — Story 3.4
        )
    except GeneratorException as exc:
        logger.error("reporter_node: generation failed: %s", exc)
        return {
            "draft_answer": f"[error] Answer generation failed: {exc}",
            "citations": [],
        }

    return {
        "draft_answer": result.answer,
        "citations": [{"source": c.source, "chunk_id": c.chunk_id} for c in result.citations],
    }


def reviewer_node(state: AgentState) -> dict:
    """Review and approve (or reject) the draft answer.

    Stub: will call a reviewer LLM in Story 3.6.
    The stub always approves to keep the graph operational during development.

    Args:
        state: Current workflow state containing draft_answer.

    Returns:
        Partial state update with review_passed, review_feedback, and final_answer.
    """
    draft = (state.get("draft_answer") or "")[:80]
    logger.debug("reviewer_node: reviewing draft=%r", draft)
    return {
        "review_passed": True,
        "review_feedback": "",
        "final_answer": state.get("draft_answer", ""),
    }
