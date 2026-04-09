"""Tests for egf.embedder."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from egf.embedder import DEFAULT_MODEL, EmbedderError, embed_corpus
from egf.loader import Document

if TYPE_CHECKING:
    from pathlib import Path


def make_docs(n: int = 4) -> list[Document]:
    return [Document(name=f"doc_{i}.md", text=f"Document {i}. " * 20)
            for i in range(n)]


def mock_model(n_docs: int, dim: int = 384) -> MagicMock:
    """Return a mock SentenceTransformer that produces valid float32 arrays."""
    m = MagicMock()
    m.encode.return_value = np.random.rand(n_docs, dim).astype(np.float32)
    return m


# ── Happy path ────────────────────────────────────────────────────────────────

def test_embed_returns_correct_shape() -> None:
    docs = make_docs(4)
    with patch("egf.embedder.SentenceTransformer", return_value=mock_model(4)):
        result = embed_corpus(docs)
    assert result.shape == (4, 384)


def test_embed_returns_float32() -> None:
    docs = make_docs(4)
    with patch("egf.embedder.SentenceTransformer", return_value=mock_model(4)):
        result = embed_corpus(docs)
    assert result.dtype == np.float32


def test_embed_writes_npy(tmp_path: Path) -> None:
    docs = make_docs(4)
    out = tmp_path / "embeddings.npy"
    with patch("egf.embedder.SentenceTransformer", return_value=mock_model(4)):
        embed_corpus(docs, output_path=out)
    assert out.exists()
    loaded = np.load(out)
    assert loaded.shape == (4, 384)


def test_embed_no_write_without_output_path(tmp_path: Path) -> None:
    docs = make_docs(4)
    with patch("egf.embedder.SentenceTransformer", return_value=mock_model(4)):
        embed_corpus(docs)  # no output_path
    assert not (tmp_path / "embeddings.npy").exists()


# ── Error paths ───────────────────────────────────────────────────────────────

def test_empty_corpus_raises() -> None:
    with pytest.raises(EmbedderError, match="empty"):
        embed_corpus([])


def test_model_load_failure_raises() -> None:
    with (
        patch("egf.embedder.SentenceTransformer",
              side_effect=OSError("model not found")),
        pytest.raises(EmbedderError, match="Failed to load model"),
    ):
        embed_corpus(make_docs(4))


def test_non_finite_values_raise() -> None:
    docs = make_docs(4)
    bad = np.full((4, 384), np.nan, dtype=np.float32)
    m = MagicMock()
    m.encode.return_value = bad
    with (
        patch("egf.embedder.SentenceTransformer", return_value=m),
        pytest.raises(EmbedderError, match="non-finite"),
    ):
        embed_corpus(docs)


def test_shape_mismatch_raises() -> None:
    docs = make_docs(4)
    wrong_shape = np.random.rand(3, 384).astype(np.float32)
    m = MagicMock()
    m.encode.return_value = wrong_shape
    with (
        patch("egf.embedder.SentenceTransformer", return_value=m),
        pytest.raises(EmbedderError, match="count"),
    ):
        embed_corpus(docs)


def test_default_model_name() -> None:
    assert DEFAULT_MODEL == "all-MiniLM-L6-v2"
