"""Tests for egf.gaps."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from egf.density import estimate_density
from egf.gaps import GapRegion, GapsError, detect_gaps

if TYPE_CHECKING:
    from pathlib import Path


def make_clustered_corpus(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """
    Two dense clusters with a clear gap between them.
    Returns (reduced_2d, item_names).
    """
    rng = np.random.default_rng(seed)
    cluster_a = rng.random((8, 2)).astype(np.float32) * 0.25
    cluster_b = (
        rng.random((8, 2)).astype(np.float32) * 0.25
        + np.array([2.0, 2.0], dtype=np.float32)
    )
    pts = np.vstack([cluster_a, cluster_b])
    names = [f"item_{i:02d}.md" for i in range(16)]
    return pts, names


def make_density(pts: np.ndarray, k: int = 3) -> object:
    return estimate_density(pts, k=k, grid_size=32, sigma=1.0)


# ── Happy path — clustered corpus ─────────────────────────────────────────────

def test_detect_gaps_returns_list() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names)  # type: ignore[arg-type]
    assert isinstance(gaps, list)


def test_detect_gaps_finds_at_least_one_gap() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    assert len(gaps) >= 1


def test_gaps_are_gap_region_instances() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    for g in gaps:
        assert isinstance(g, GapRegion)


def test_gaps_sorted_by_isolation_descending() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    scores = [g.isolation_score for g in gaps]
    assert scores == sorted(scores, reverse=True)


def test_gap_ids_sequential() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    ids = [g.gap_id for g in gaps]
    assert ids == list(range(len(gaps)))


def test_isolation_scores_above_threshold() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    threshold = 0.2
    gaps = detect_gaps(dr, pts, names, isolation_min=threshold)  # type: ignore[arg-type]
    for g in gaps:
        assert g.isolation_score >= threshold


def test_max_gaps_respected() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.0, max_gaps=2)  # type: ignore[arg-type]
    assert len(gaps) <= 2


def test_nearest_items_are_known_names() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    for g in gaps:
        for name in g.nearest_items:
            assert name in names


def test_nearest_items_and_distances_same_length() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    for g in gaps:
        assert len(g.nearest_items) == len(g.nearest_item_distances)


def test_centroid_is_two_floats() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    for g in gaps:
        assert len(g.centroid_2d) == 2
        assert all(isinstance(v, float) for v in g.centroid_2d)


def test_radius_is_positive() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)  # type: ignore[arg-type]
    for g in gaps:
        assert g.radius > 0


# ── Empty result cases ────────────────────────────────────────────────────────

def test_high_isolation_min_returns_empty() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    gaps = detect_gaps(dr, pts, names, isolation_min=1.0)  # type: ignore[arg-type]
    assert gaps == []


# ── JSON output ───────────────────────────────────────────────────────────────

def test_writes_gaps_json(tmp_path: Path) -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    out = tmp_path / "gaps.json"
    detect_gaps(dr, pts, names, isolation_min=0.1, output_path=out)  # type: ignore[arg-type]
    assert out.exists()


def test_gaps_json_schema(tmp_path: Path) -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    out = tmp_path / "gaps.json"
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1, output_path=out)  # type: ignore[arg-type]
    if gaps:
        data = json.loads(out.read_text())
        assert isinstance(data, list)
        first = data[0]
        assert "gap_id" in first
        assert "isolation_score" in first
        assert "centroid_2d" in first
        assert "radius" in first
        assert "nearest_items" in first
        assert "nearest_item_distances" in first


def test_empty_gaps_writes_empty_array(tmp_path: Path) -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    out = tmp_path / "gaps.json"
    detect_gaps(dr, pts, names, isolation_min=1.0, output_path=out)  # type: ignore[arg-type]
    data = json.loads(out.read_text())
    assert data == []


# ── Input validation ──────────────────────────────────────────────────────────

def test_mismatched_names_raises() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    with pytest.raises(GapsError):
        detect_gaps(dr, pts, names[:-1])  # type: ignore[arg-type]


def test_wrong_shape_raises() -> None:
    pts, names = make_clustered_corpus()
    dr = make_density(pts)
    with pytest.raises(GapsError):
        detect_gaps(dr, pts.reshape(-1), names)  # type: ignore[arg-type]


def test_gaps_inside_corpus_bounding_box() -> None:
    """Regression: gaps were detected outside the corpus convex hull."""
    pts, names = make_clustered_corpus()
    dr = make_density(pts, k=3)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)
    if gaps:
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        pad = 0.5 * max(x_max - x_min, y_max - y_min)
        for g in gaps:
            assert x_min - pad <= g.centroid_2d[0] <= x_max + pad
            assert y_min - pad <= g.centroid_2d[1] <= y_max + pad


def test_no_duplicate_bounding_items_in_gaps() -> None:
    """Regression: adjacent gaps produced duplicate candidates."""
    pts, names = make_clustered_corpus()
    dr = make_density(pts, k=3)
    gaps = detect_gaps(dr, pts, names, isolation_min=0.1)
    bound_sets = [tuple(sorted(g.nearest_items)) for g in gaps]
    assert len(bound_sets) == len(set(bound_sets))
