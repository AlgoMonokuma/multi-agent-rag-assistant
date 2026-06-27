"""Unit tests for core.rag.generator — LLMGenerator.

All tests use a mocked Groq client injected via the DI constructor slot.
No real API calls are made.

Test coverage:
  - test_generate_success                                    AC 1, 2, 3, 4
  - test_generate_prompt_includes_chunk_id                   Issue 2 (prompt header fix)
  - test_generate_empty_chunks_returns_fallback              AC 6
  - test_generate_empty_chunks_none_returns_fallback         AC 6
  - test_generate_api_failure_raises_generator_exception     AC 5 / Issue 1
  - test_missing_api_key_raises_generator_exception          Issue 1 (ValueError wrap)
  - test_prompt_respects_token_budget                        AC 2
  - test_oversized_first_chunk_returns_fallback_no_api_call  Issue 4
  - test_citations_only_include_admitted_chunks              Issue 3
  - test_citations_extracted_from_chunks                     AC 4
  - test_malformed_response_raises_generator_exception       Issue 5
  - test_none_content_response_raises_generator_exception    Issue 5
"""

from unittest.mock import MagicMock, patch

import pytest

from core.rag.generator import (
    CitationRef,
    GeneratorException,
    GeneratorResponse,
    LLMGenerator,
)
from core.rag.retriever import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str,
    source: str,
    text: str,
    rank: int = 1,
) -> RetrievedChunk:
    """Build a minimal RetrievedChunk for testing."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        page_content=text,
        metadata={"source": source, "chunk_id": chunk_id, "session_id": "test-session"},
        vector_score=0.9,
        keyword_score=0.5,
        merged_score=0.8,
        rank=rank,
    )


def _make_mock_client(answer_text: str = "This is the answer.") -> MagicMock:
    """Return a mock Groq client whose completions return ``answer_text``."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = answer_text

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def _get_user_content(mock_client: MagicMock) -> str:
    """Extract the user-role message content from a mock Groq client call."""
    call_kwargs = mock_client.chat.completions.create.call_args
    # create() is always called with keyword arguments
    messages = call_kwargs.kwargs["messages"]
    return next(m["content"] for m in messages if m["role"] == "user")


# ---------------------------------------------------------------------------
# Tests — core happy path
# ---------------------------------------------------------------------------


def test_generate_success() -> None:
    """AC 1, 2, 3, 4 — successful generation returns correct answer and citations."""
    mock_client = _make_mock_client("This is the answer.")
    generator = LLMGenerator(client=mock_client)
    chunks = [_make_chunk("c1", "doc.pdf", "Some relevant text.")]

    result = generator.generate(query="What is RAG?", chunks=chunks, session_id="s1")

    assert isinstance(result, GeneratorResponse)
    assert result.answer == "This is the answer."
    assert len(result.citations) == 1
    assert result.citations[0].source == "doc.pdf"
    assert result.citations[0].chunk_id == "c1"
    mock_client.chat.completions.create.assert_called_once()


def test_generate_prompt_includes_chunk_id() -> None:
    """Issue 2 — prompt header must include both source AND chunk_id for precise citation."""
    mock_client = _make_mock_client("Answer.")
    generator = LLMGenerator(client=mock_client)
    chunks = [_make_chunk("chunk-99", "paper.pdf", "Some text about retrieval.")]

    generator.generate(query="What is RAG?", chunks=chunks, session_id="s1")

    user_content = _get_user_content(mock_client)
    assert "paper.pdf" in user_content
    assert "chunk-99" in user_content


# ---------------------------------------------------------------------------
# Tests — empty / None chunks (AC 6)
# ---------------------------------------------------------------------------


def test_generate_empty_chunks_returns_fallback() -> None:
    """AC 6 — empty chunk list returns fallback response without calling the API."""
    mock_client = _make_mock_client()
    generator = LLMGenerator(client=mock_client)

    result = generator.generate(query="What is RAG?", chunks=[], session_id="s1")

    assert isinstance(result, GeneratorResponse)
    assert result.answer  # non-empty fallback message
    assert result.citations == []
    mock_client.chat.completions.create.assert_not_called()


def test_generate_empty_chunks_none_returns_fallback() -> None:
    """AC 6 — None chunk list also returns fallback without API call."""
    mock_client = _make_mock_client()
    generator = LLMGenerator(client=mock_client)

    result = generator.generate(query="What is RAG?", chunks=None, session_id="s2")

    assert isinstance(result, GeneratorResponse)
    assert result.citations == []
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — API / config failures (AC 5, Issue 1)
# ---------------------------------------------------------------------------


def test_generate_api_failure_raises_generator_exception() -> None:
    """AC 5 — Groq API network error is caught and re-raised as GeneratorException."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("network error")

    generator = LLMGenerator(client=mock_client)
    chunks = [_make_chunk("c1", "doc.pdf", "Some text.")]

    with pytest.raises(GeneratorException, match="Groq API call failed"):
        generator.generate(query="What is RAG?", chunks=chunks, session_id="s1")


def test_missing_api_key_raises_generator_exception() -> None:
    """Issue 1 — ValueError from require_groq_api_key() must be wrapped as GeneratorException.

    Before the fix, ValueError would escape as-is, breaking the API contract
    (callers expect only GeneratorException from this module).
    """
    generator = LLMGenerator()  # no injected client → will call _load_client()

    with (
        patch("core.rag.generator.LLMGenerator._load_client") as mock_load,
    ):
        mock_load.side_effect = GeneratorException(
            "GROQ_API_KEY is missing or empty."
        )
        chunks = [_make_chunk("c1", "doc.pdf", "Some text.")]

        with pytest.raises(GeneratorException, match="GROQ_API_KEY"):
            generator.generate(query="test?", chunks=chunks, session_id="s1")


# ---------------------------------------------------------------------------
# Tests — token budget (AC 2, Issue 3, Issue 4)
# ---------------------------------------------------------------------------


def test_prompt_respects_token_budget() -> None:
    """AC 2 — chunks exceeding token budget are excluded from the prompt."""
    mock_client = _make_mock_client("Truncated answer.")
    # token_budget=10 (≈ 40 chars). short_text=30 chars fits; long_text=200 chars does not.
    generator = LLMGenerator(client=mock_client, token_budget=10)

    short_text = "A" * 30   # 30 chars → 7 tokens — fits
    long_text = "B" * 200   # 200 chars → 50 tokens — exceeds remaining budget

    chunks = [
        _make_chunk("c1", "doc1.pdf", short_text, rank=1),
        _make_chunk("c2", "doc2.pdf", long_text, rank=2),
    ]

    generator.generate(query="test?", chunks=chunks, session_id="s1")

    user_content = _get_user_content(mock_client)
    assert "doc1.pdf" in user_content    # admitted chunk IS in prompt
    assert "doc2.pdf" not in user_content  # excluded chunk is NOT in prompt


def test_citations_only_include_admitted_chunks() -> None:
    """Issue 3 — citations must only reference chunks that were included in the prompt.

    Before the fix, _extract_citations received the full chunk list, so truncated
    chunks (excluded by the token budget) still appeared as citations — misleading
    the caller into thinking the model had access to that content.
    """
    mock_client = _make_mock_client("Answer from first chunk only.")
    # Tiny budget: only the first short chunk fits
    generator = LLMGenerator(client=mock_client, token_budget=10)

    admitted_text = "A" * 30   # 7 tokens — fits
    excluded_text = "B" * 200  # 50 tokens — excluded

    chunks = [
        _make_chunk("c1", "admitted.pdf", admitted_text, rank=1),
        _make_chunk("c2", "excluded.pdf", excluded_text, rank=2),
    ]

    result = generator.generate(query="test?", chunks=chunks, session_id="s1")

    # Only the admitted chunk should be cited
    assert len(result.citations) == 1
    assert result.citations[0].source == "admitted.pdf"
    assert result.citations[0].chunk_id == "c1"


def test_oversized_first_chunk_returns_fallback_no_api_call() -> None:
    """Issue 4 — when the very first chunk exceeds the budget, no API call is made.

    Before the fix, _build_prompt would return an empty context_parts list, the
    user message would contain '(no context available)', but generate() would still
    call the Groq API — wasting a round-trip and returning a confabulated answer.
    """
    mock_client = _make_mock_client("Should never be returned.")
    # Single chunk that is way too large for the budget
    generator = LLMGenerator(client=mock_client, token_budget=1)

    oversized_chunk = _make_chunk("c1", "big.pdf", "X" * 500)  # 125 tokens >> budget=1

    result = generator.generate(query="test?", chunks=[oversized_chunk], session_id="s1")

    assert isinstance(result, GeneratorResponse)
    assert result.answer  # fallback message, not "Should never be returned."
    assert "Should never be returned." not in result.answer
    assert result.citations == []
    mock_client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Tests — malformed Groq response (Issue 5)
# ---------------------------------------------------------------------------


def test_malformed_response_raises_generator_exception() -> None:
    """Issue 5 — empty choices list raises GeneratorException instead of IndexError."""
    mock_response = MagicMock()
    mock_response.choices = []  # empty list → choices[0] would IndexError

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    generator = LLMGenerator(client=mock_client)
    chunks = [_make_chunk("c1", "doc.pdf", "Some text.")]

    with pytest.raises(GeneratorException, match="malformed or empty"):
        generator.generate(query="test?", chunks=chunks, session_id="s1")


def test_none_content_response_raises_generator_exception() -> None:
    """Issue 5 — None content in Groq response raises GeneratorException."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = None  # type: ignore[assignment]

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    generator = LLMGenerator(client=mock_client)
    chunks = [_make_chunk("c1", "doc.pdf", "Some text.")]

    with pytest.raises(GeneratorException, match="malformed or empty"):
        generator.generate(query="test?", chunks=chunks, session_id="s1")


# ---------------------------------------------------------------------------
# Tests — citation accuracy (AC 4)
# ---------------------------------------------------------------------------


def test_citations_extracted_from_chunks() -> None:
    """AC 4 — citation list matches source and chunk_id from all admitted chunks."""
    mock_client = _make_mock_client("Combined answer.")
    generator = LLMGenerator(client=mock_client)

    chunks = [
        _make_chunk("chunk-a", "report.pdf", "First passage.", rank=1),
        _make_chunk("chunk-b", "manual.pdf", "Second passage.", rank=2),
    ]

    result = generator.generate(query="What happened?", chunks=chunks, session_id="s3")

    assert len(result.citations) == 2
    assert result.citations[0] == CitationRef(source="report.pdf", chunk_id="chunk-a")
    assert result.citations[1] == CitationRef(source="manual.pdf", chunk_id="chunk-b")
