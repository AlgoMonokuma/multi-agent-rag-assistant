"""Groq-backed LLM answer generation for retrieved context chunks.

Provides:
  - GeneratorException  — domain exception for all API / config failures
  - CitationRef         — (source, chunk_id) citation metadata
  - GeneratorResponse   — generated answer + citation list
  - LLMGenerator        — orchestrates prompt assembly and Groq API call
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, NamedTuple, Optional

from core.log import get_logger

if TYPE_CHECKING:
    from core.rag.retriever import RetrievedChunk

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_TOKEN_BUDGET = 3000  # context tokens (naive estimate: 1 token ≈ 4 chars)
DEFAULT_TIMEOUT = 60.0  # seconds

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question using ONLY "
    "the provided context chunks. Do not fabricate facts. If the context is "
    "insufficient, say so clearly."
)


# ---------------------------------------------------------------------------
# Domain exception
# ---------------------------------------------------------------------------


class GeneratorException(Exception):
    """LLM generation error: API failure, missing config, or import error."""


# ---------------------------------------------------------------------------
# Response dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CitationRef:
    """Source citation extracted from a retrieved chunk."""

    source: str
    chunk_id: str


@dataclass
class GeneratorResponse:
    """Result of a successful (or fallback) LLM generation call."""

    answer: str
    citations: List[CitationRef] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helper: keeps prompt and citations in sync
# ---------------------------------------------------------------------------


class _PromptResult(NamedTuple):
    """Return type of _build_prompt — bundles messages with the admitted chunks."""

    messages: List[dict]
    admitted_chunks: List[Any]  # only chunks that actually fit in the prompt


# ---------------------------------------------------------------------------
# LLM Generator
# ---------------------------------------------------------------------------


class LLMGenerator:
    """Generate grounded answers from retrieved chunks using the Groq API.

    Supports dependency injection of a pre-built Groq client for testing.
    When ``client`` is *None* (production path), the real Groq client is
    instantiated lazily on the first ``generate()`` call.

    Args:
        client:       Optional pre-built Groq client (for test injection).
        model:        Groq model identifier to use for generation.
        token_budget: Maximum context tokens to include in the prompt.
    """

    def __init__(
        self,
        client: Any = None,
        model: str = DEFAULT_GROQ_MODEL,
        token_budget: int = DEFAULT_TOKEN_BUDGET,
    ) -> None:
        self._client = client
        self._model = model
        self._token_budget = token_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        query: str,
        chunks: Optional[List[Any]],
        session_id: str,
        web_context: Optional[List[dict]] = None,
    ) -> GeneratorResponse:
        """Generate a grounded answer from retrieved chunks.

        Args:
            query:       The user's original question.
            chunks:      Ordered list of ``RetrievedChunk`` objects from
                         ``HybridRetriever`` / ``CrossEncoderReranker``.
            session_id:  Used only for log context; never sent to Groq.
            web_context: Optional list of web search result dicts from
                         web_search_node (Story 3.4). Each dict has
                         ``url``, ``content``, and ``score`` keys.
                         Appended to the prompt context block when non-empty.
                         Defaults to None (backward-compatible).

        Returns:
            A ``GeneratorResponse`` with ``answer`` and ``citations``.
            Citations only reference chunks that were *actually included*
            in the prompt (i.e. chunks admitted within the token budget).

        Raises:
            GeneratorException: If the Groq API call fails or times out,
                                if ``groq`` is not installed, or if the
                                API key is missing / misconfigured.
        """
        # AC 6: graceful empty-context fallback — no API call
        # Story 3.4: if web_context is provided, skip the no-chunks fallback so
        # web results can still be used to generate an answer.
        if not chunks and not web_context:
            logger.info(
                "LLMGenerator.generate: no chunks or web context for session %s; returning fallback.",
                session_id,
            )
            return GeneratorResponse(
                answer="I could not find relevant information to answer your question.",
                citations=[],
            )

        # Lazy-load real Groq client if not injected
        # Issue 1 / 6 fix: wrap ValueError from require_groq_api_key inside GeneratorException
        if self._client is None:
            self._client = self._load_client()

        # Build prompt and track which chunks were actually admitted
        # Issue 3 fix: _build_prompt now returns admitted_chunks alongside messages,
        # so citations only reference chunks present in the prompt.
        # Story 3.4: pass web_context so web snippets are appended to the context block.
        prompt_result = self._build_prompt(query, chunks or [], self._token_budget, web_context)
        messages = prompt_result.messages
        admitted_chunks = prompt_result.admitted_chunks

        # Issue 4 fix: if the first chunk alone exceeded the budget, admitted_chunks
        # is empty — the context block reads "(no context available)".  In that case
        # calling the LLM is wasteful and misleading; return the no-context fallback.
        # Story 3.4: skip this fallback if web_context has content (web results compensate).
        if not admitted_chunks and not web_context:
            logger.warning(
                "LLMGenerator.generate: all chunks exceeded token budget=%d for session %s; "
                "returning fallback without API call.",
                self._token_budget,
                session_id,
            )
            return GeneratorResponse(
                answer="I could not find relevant information to answer your question.",
                citations=[],
            )

        citations = self._extract_citations(admitted_chunks)

        logger.debug(
            "LLMGenerator.generate: calling Groq model=%r session=%r "
            "admitted_chunks=%d / total_chunks=%d",
            self._model,
            session_id,
            len(admitted_chunks),
            len(chunks),
        )

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
        except Exception as error:
            logger.error(
                "LLMGenerator.generate: Groq API call failed for session %s (query_len=%d): %s",
                session_id,
                len(query),
                error,
            )
            raise GeneratorException(
                f"Groq API call failed for session {session_id}."
            ) from error

        # Issue 5 fix: guard against empty choices or None content
        try:
            answer_text: str = response.choices[0].message.content
            if not answer_text:
                raise ValueError("Groq returned an empty or None answer.")
        except (IndexError, AttributeError, ValueError) as error:
            logger.error(
                "LLMGenerator.generate: malformed Groq response for session %s: %s",
                session_id,
                error,
            )
            raise GeneratorException(
                f"Groq returned a malformed or empty response for session {session_id}."
            ) from error

        logger.info(
            "LLMGenerator.generate: answer generated (len=%d) for session %s",
            len(answer_text),
            session_id,
        )
        return GeneratorResponse(answer=answer_text, citations=citations)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_client(self) -> Any:
        """Lazy-load the Groq client.  Mirrors CrossEncoderReranker._load_model().

        Raises:
            GeneratorException: If ``groq`` is not installed *or* if
                ``GROQ_API_KEY`` is missing / empty (wraps the ValueError
                from ``settings.require_groq_api_key()``).
        """
        try:
            from groq import Groq  # type: ignore  # noqa: PLC0415
        except ImportError as error:
            raise GeneratorException(
                "groq is not installed; LLM generation is unavailable. "
                "Run: uv add 'groq>=0.9.0'  (or pip install 'groq>=0.9.0')"
            ) from error

        # Issue 1 / 6 fix: require_groq_api_key raises ValueError when the key
        # is absent; wrap it so callers always receive a GeneratorException.
        try:
            from core.config import settings  # noqa: PLC0415

            api_key = settings.require_groq_api_key()
        except ValueError as error:
            raise GeneratorException(
                "GROQ_API_KEY is missing or empty. "
                "Set it in .env before enabling Groq-backed features."
            ) from error

        return Groq(api_key=api_key, timeout=DEFAULT_TIMEOUT)

    @staticmethod
    def _build_prompt(
        query: str,
        chunks: List[Any],
        token_budget: int,
        web_context: Optional[List[dict]] = None,
    ) -> _PromptResult:
        """Assemble a system + user message pair respecting the token budget.

        Token estimate: 1 token ≈ 4 characters (naive but deterministic).

        Returns a ``_PromptResult`` that bundles both the ``messages`` list
        and the ``admitted_chunks`` — the subset of ``chunks`` that were
        actually included in the context block.  Callers should derive
        citations only from ``admitted_chunks``, not from the full input list.

        Issue 2 fix: each context block now includes both ``[Source: …]``
        and ``[Chunk ID: …]`` so the model can produce precise references.

        Story 3.4: optional ``web_context`` list of dicts is appended as a
        ``## Web Search Results`` section after the RAG context block.

        Args:
            query:        The user's question.
            chunks:       ``RetrievedChunk`` objects ordered by relevance.
            token_budget: Maximum total context tokens to include.
            web_context:  Optional web search results from web_search_node.
                          Each dict has ``url``, ``content``, ``score`` keys.

        Returns:
            A ``_PromptResult(messages, admitted_chunks)`` named-tuple.
        """
        context_parts: List[str] = []
        admitted_chunks: List[Any] = []
        used_tokens = 0

        for chunk in chunks:
            chunk_tokens = len(chunk.page_content) // 4
            if used_tokens + chunk_tokens > token_budget:
                break
            source = chunk.metadata.get("source", "unknown")
            chunk_id = chunk.metadata.get("chunk_id", chunk.chunk_id)
            # Issue 2 fix: include chunk_id in the header so model can cite precisely
            context_parts.append(
                f"[Source: {source} | Chunk ID: {chunk_id}]\n{chunk.page_content}"
            )
            admitted_chunks.append(chunk)
            used_tokens += chunk_tokens

        context_block = (
            "\n\n---\n\n".join(context_parts) if context_parts else "(no context available)"
        )
        user_content = f"Context:\n{context_block}\n\nQuestion: {query}"

        # Story 3.4: append web search results section when present
        if web_context:
            web_section = "\n\n## Web Search Results\n"
            for item in web_context[:5]:  # cap at 5 to respect token budget
                if not isinstance(item, dict):
                    continue
                web_section += f"Source: {item.get('url', 'unknown')}\n"
                web_section += f"{item.get('content', '')}\n\n"
            user_content += web_section

        messages: List[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return _PromptResult(messages=messages, admitted_chunks=admitted_chunks)


    @staticmethod
    def _extract_citations(chunks: List[Any]) -> List[CitationRef]:
        """Extract ``CitationRef`` list from chunk metadata.

        Args:
            chunks: ``RetrievedChunk`` objects that were *admitted* into the
                    prompt (already filtered by token budget).

        Returns:
            List of ``CitationRef`` — one per admitted chunk, preserving order.
        """
        citations: List[CitationRef] = []
        for chunk in chunks:
            source = chunk.metadata.get("source", "unknown")
            chunk_id = chunk.metadata.get("chunk_id", chunk.chunk_id)
            citations.append(CitationRef(source=source, chunk_id=chunk_id))
        return citations
