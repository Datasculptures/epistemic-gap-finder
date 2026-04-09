"""Tests for egf.reducer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from egf.reducer import RANDOM_SEED, ReducerError, ReductionResult, reduce_embeddings

if TYPE_CHECKING:
    from pathlib import Path


def make_embeddings(n: int = 20, dim: int = 16) -> np.ndarray:
    """Small deterministic float32 array for tests."""
    rng = np.random.default_rng(0)
    return rng.random((n, dim)).astype(np.float32)


def mock_umap_class(n_components: int) -> MagicMock:
    """Return a mock UMAP instance whose fit_transform returns valid output."""
    instance = MagicMock()

    def fit_transform(x: np.ndarray) -> np.ndarray:
        return np.random.default_rng(0).random(
            (x.shape[0], n_components)
        ).astype(np.float32)

    instance.fit_transform.side_effect = fit_transform
    cls = MagicMock(return_value=instance)
    return cls


# ── Happy path ────────────────────────────────────────────────────────────────

def test_reduce_returns_result() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        mock_umap.UMAP.side_effect = [
            mock_umap_class(2).return_value,
            mock_umap_class(3).return_value,
        ]
        result = reduce_embeddings(emb)
    assert isinstance(result, ReductionResult)


def test_reduce_2d_shape() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.zeros((20, 2), dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        result = reduce_embeddings(emb)
    assert result.reduced_2d.shape == (20, 2)


def test_reduce_3d_shape() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.zeros((20, 2), dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        result = reduce_embeddings(emb)
    assert result.reduced_3d.shape == (20, 3)


def test_reduce_writes_npy_files(tmp_path: Path) -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.zeros((20, 2), dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        reduce_embeddings(emb, output_dir=tmp_path)
    assert (tmp_path / "reduced_2d.npy").exists()
    assert (tmp_path / "reduced_3d.npy").exists()


def test_reduce_output_is_float32() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        # Return float64 — reducer must cast to float32
        inst2.fit_transform.return_value = np.zeros((20, 2), dtype=np.float64)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float64)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        result = reduce_embeddings(emb)
    assert result.reduced_2d.dtype == np.float32
    assert result.reduced_3d.dtype == np.float32


def test_n_neighbors_stored_in_result() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.zeros((20, 2), dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        result = reduce_embeddings(emb, n_neighbors=10)
    assert result.n_neighbors == 10


def test_random_seed_constant() -> None:
    assert RANDOM_SEED == 42


# ── n_neighbors clamping ──────────────────────────────────────────────────────

def test_n_neighbors_clamped_when_too_large(capsys: pytest.CaptureFixture[str]) -> None:
    emb = make_embeddings(10)  # 10 docs
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.zeros((10, 2), dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((10, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        result = reduce_embeddings(emb, n_neighbors=15)  # 15 >= 10 → clamp
    assert result.n_neighbors == 9  # clamped to n_docs - 1
    captured = capsys.readouterr()
    assert "clamped" in captured.err


# ── Error paths ───────────────────────────────────────────────────────────────

def test_1d_input_raises() -> None:
    with pytest.raises(ReducerError):
        reduce_embeddings(np.zeros(10, dtype=np.float32))


def test_single_row_raises() -> None:
    with pytest.raises(ReducerError):
        reduce_embeddings(np.zeros((1, 16), dtype=np.float32))


def test_non_finite_output_raises() -> None:
    emb = make_embeddings(20)
    with patch("egf.reducer.umap") as mock_umap:
        inst2 = MagicMock()
        inst2.fit_transform.return_value = np.full((20, 2), np.nan,
                                                   dtype=np.float32)
        inst3 = MagicMock()
        inst3.fit_transform.return_value = np.zeros((20, 3), dtype=np.float32)
        mock_umap.UMAP.side_effect = [inst2, inst3]
        with pytest.raises(ReducerError, match="non-finite"):
            reduce_embeddings(emb)
