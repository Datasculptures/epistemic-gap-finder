"""Tests for egf.density."""

from __future__ import annotations

import numpy as np
import pytest

from egf.density import DensityError, DensityResult, estimate_density


def make_2d(n: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).astype(np.float32)


def make_clustered_2d(seed: int = 0) -> np.ndarray:
    """Two dense clusters with a gap between them."""
    rng = np.random.default_rng(seed)
    cluster_a = rng.random((10, 2)).astype(np.float32) * 0.3
    cluster_b = (
        rng.random((10, 2)).astype(np.float32) * 0.3
        + np.array([1.5, 1.5], dtype=np.float32)
    )
    return np.vstack([cluster_a, cluster_b])


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_density_result() -> None:
    result = estimate_density(make_2d(20))
    assert isinstance(result, DensityResult)


def test_point_density_shape() -> None:
    pts = make_2d(20)
    result = estimate_density(pts)
    assert result.point_density.shape == (20,)


def test_grid_density_shape() -> None:
    pts = make_2d(20)
    result = estimate_density(pts, grid_size=32)
    assert result.grid_density.shape == (32, 32)


def test_grid_axes_shape() -> None:
    pts = make_2d(20)
    result = estimate_density(pts, grid_size=32)
    assert result.grid_x.shape == (32,)
    assert result.grid_y.shape == (32,)


def test_point_density_dtype() -> None:
    result = estimate_density(make_2d(20))
    assert result.point_density.dtype == np.float32


def test_grid_density_dtype() -> None:
    result = estimate_density(make_2d(20))
    assert result.grid_density.dtype == np.float32


def test_all_values_finite() -> None:
    result = estimate_density(make_2d(20))
    assert np.all(np.isfinite(result.point_density))
    assert np.all(np.isfinite(result.grid_density))


def test_all_point_densities_positive() -> None:
    result = estimate_density(make_2d(20))
    assert np.all(result.point_density > 0)


# ── Clustered data has higher density in clusters ─────────────────────────────

def test_clustered_data_density_varies() -> None:
    pts = make_clustered_2d()
    result = estimate_density(pts, k=3)
    assert result.point_density.max() > result.point_density.min() * 2


# ── k and sigma stored ────────────────────────────────────────────────────────

def test_k_stored_in_result() -> None:
    result = estimate_density(make_2d(20), k=3)
    assert result.k == 3


def test_sigma_stored_in_result() -> None:
    result = estimate_density(make_2d(20), sigma=2.0)
    assert result.sigma == 2.0


# ── k clamping ────────────────────────────────────────────────────────────────

def test_k_clamped_when_too_large(capsys: pytest.CaptureFixture[str]) -> None:
    pts = make_2d(8)
    result = estimate_density(pts, k=20)  # 20 >= 8 → clamp
    assert result.k < 8
    assert "clamp" in capsys.readouterr().err.lower()


# ── Error paths ───────────────────────────────────────────────────────────────

def test_wrong_shape_raises() -> None:
    with pytest.raises(DensityError):
        estimate_density(np.zeros((20, 3), dtype=np.float32))


def test_single_point_raises() -> None:
    with pytest.raises(DensityError):
        estimate_density(np.zeros((1, 2), dtype=np.float32))


def test_minimum_two_points() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    result = estimate_density(pts, k=1)
    assert result.point_density.shape == (2,)
