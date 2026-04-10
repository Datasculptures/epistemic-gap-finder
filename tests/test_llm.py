"""Tests for egf.llm."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from egf.domain import parse_domain
from egf.llm import (
    DEFAULT_LLM_HOST,
    DEFAULT_LLM_MODEL,
    LLMCandidate,
    generate_candidate,
    health_check,
)


def philosophy_domain():  # type: ignore[no-untyped-def]
    return parse_domain("philosophy")


def mock_response(content: str, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"message": {"content": content}}
    resp.raise_for_status = MagicMock()
    return resp


VALID_JSON = json.dumps({
    "name": "Pragmatic Phenomenology",
    "function_summary": "Bridges lived experience with practical outcomes.",
    "positioning_summary": (
        "Occupies the gap between pure phenomenology and pragmatism."
    ),
})


# ── health_check ──────────────────────────────────────────────────────────────

def test_health_check_true_on_200() -> None:
    with patch("egf.llm.httpx.get") as mock_get:
        mock_get.return_value = mock_response("{}", 200)
        assert health_check() is True


def test_health_check_false_on_connection_error() -> None:
    import httpx
    with patch("egf.llm.httpx.get", side_effect=httpx.ConnectError("refused")):
        assert health_check() is False


def test_health_check_false_on_non_200() -> None:
    with patch("egf.llm.httpx.get") as mock_get:
        mock_get.return_value = mock_response("{}", 404)
        assert health_check() is False


def test_health_check_never_raises() -> None:
    with patch("egf.llm.httpx.get", side_effect=RuntimeError("unexpected")):
        result = health_check()
        assert isinstance(result, bool)


def test_default_constants() -> None:
    assert DEFAULT_LLM_HOST == "http://localhost:11434"
    assert DEFAULT_LLM_MODEL == "llama3"


# ── generate_candidate — success ──────────────────────────────────────────────

def test_generate_returns_llm_candidate() -> None:
    with patch("egf.llm.httpx.post") as mock_post:
        mock_post.return_value = mock_response(VALID_JSON)
        result = generate_candidate(
            gap_id=0,
            bounding_items=["stoicism.md", "existentialism.md"],
            vocabulary_terms=["virtue", "existence", "freedom"],
            domain=philosophy_domain(),
        )
    assert isinstance(result, LLMCandidate)


def test_generate_returns_correct_fields() -> None:
    with patch("egf.llm.httpx.post") as mock_post:
        mock_post.return_value = mock_response(VALID_JSON)
        result = generate_candidate(
            gap_id=0,
            bounding_items=["stoicism.md"],
            vocabulary_terms=["virtue"],
            domain=philosophy_domain(),
        )
    assert result is not None
    assert result.name == "Pragmatic Phenomenology"
    assert len(result.function_summary) > 0
    assert len(result.positioning_summary) > 0


# ── generate_candidate — markdown fence stripping ────────────────────────────

def test_strips_markdown_fences() -> None:
    fenced = f"```json\n{VALID_JSON}\n```"
    with patch("egf.llm.httpx.post") as mock_post:
        mock_post.return_value = mock_response(fenced)
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is not None


# ── generate_candidate — failure modes → None ────────────────────────────────

def test_malformed_json_returns_none() -> None:
    with patch("egf.llm.httpx.post") as mock_post:
        mock_post.return_value = mock_response("not json at all")
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is None


def test_missing_fields_returns_none() -> None:
    incomplete = json.dumps({"name": "Something"})
    with patch("egf.llm.httpx.post") as mock_post:
        mock_post.return_value = mock_response(incomplete)
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is None


def test_timeout_returns_none() -> None:
    import httpx
    with patch("egf.llm.httpx.post",
               side_effect=httpx.TimeoutException("timed out")):
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is None


def test_http_error_returns_none() -> None:
    import httpx
    with patch("egf.llm.httpx.post",
               side_effect=httpx.HTTPError("server error")):
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is None


def test_generate_never_raises() -> None:
    with patch("egf.llm.httpx.post", side_effect=RuntimeError("surprise")):
        result = generate_candidate(
            gap_id=0,
            bounding_items=["a.md"],
            vocabulary_terms=["x"],
            domain=philosophy_domain(),
        )
    assert result is None
