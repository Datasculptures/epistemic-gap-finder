"""Local minima detection, isolation scoring, and gap region ranking."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import minimum_filter
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from pathlib import Path

    from egf.density import DensityResult


@dataclass(frozen=True)
class GapRegion:
    gap_id: int
    isolation_score: float
    centroid_2d: tuple[float, float]
    radius: float
    nearest_items: list[str]
    nearest_item_distances: list[float]


class GapsError(Exception):
    """Raised when gap detection fails."""


def _find_local_minima(grid_density: np.ndarray) -> np.ndarray:
    """
    Return boolean mask of local minima in the density grid.
    A cell is a local minimum if it is the minimum within a 3x3 window
    AND below the global mean.
    """
    local_min: np.ndarray = minimum_filter(grid_density, size=3) == grid_density
    below_mean: np.ndarray = grid_density < grid_density.mean()
    result: np.ndarray = local_min & below_mean
    return result


def _suppress_nearby_minima(
    minima_yx: np.ndarray,
    scores: np.ndarray,
    min_separation: int = 5,
) -> np.ndarray:
    """
    Non-maximum suppression: keep only the highest-scoring minimum within
    min_separation grid cells. Returns indices of surviving minima.
    """
    order = np.argsort(scores)[::-1]
    kept: list[int] = []
    for idx in order:
        too_close = False
        for k_idx in kept:
            dist = float(np.linalg.norm(minima_yx[idx] - minima_yx[k_idx]))
            if dist < min_separation:
                too_close = True
                break
        if not too_close:
            kept.append(int(idx))
    return np.array(kept, dtype=int)


def _isolation_score(
    centroid_2d: np.ndarray,
    reduced_2d: np.ndarray,
    point_density: np.ndarray,
    n_context: int = 5,
) -> float:
    """
    Isolation score: how much less dense the gap centroid is compared
    to the global corpus average density.

    Estimates the density at the centroid via inverse-distance weighting
    from the k nearest corpus points, then compares to the mean density
    of the full corpus.

    Returns a value in [0, 1]. Higher = more isolated from corpus density.
    """
    n = len(reduced_2d)
    k = min(n_context, n)

    nn = NearestNeighbors(n_neighbors=k).fit(reduced_2d)
    distances, indices = nn.kneighbors(centroid_2d.reshape(1, -1))

    neighbour_densities = point_density[indices[0]]

    # Global corpus mean — correct comparison baseline
    global_mean_density = float(point_density.mean())
    if global_mean_density <= 0:
        return 1.0

    # Estimate density at the centroid by inverse-distance weighting
    dists = distances[0]
    dists = np.where(dists < 1e-8, 1e-8, dists)
    weights = 1.0 / dists
    weights /= weights.sum()
    estimated_density = float(np.dot(weights, neighbour_densities))

    # Compare estimated gap density to global corpus mean
    score = 1.0 - (estimated_density / global_mean_density)
    return float(np.clip(score, 0.0, 1.0))


def _nearest_items(
    centroid: np.ndarray,
    reduced_2d: np.ndarray,
    item_names: list[str],
    n: int = 3,
) -> tuple[list[str], list[float]]:
    k = min(n, len(item_names))
    nn = NearestNeighbors(n_neighbors=k).fit(reduced_2d)
    distances, indices = nn.kneighbors(centroid.reshape(1, -1))
    names = [item_names[i] for i in indices[0]]
    dists = [round(float(d), 6) for d in distances[0]]
    return names, dists


def _inside_convex_hull(
    points: np.ndarray,
    candidates: np.ndarray,
    margin: float = 0.15,
) -> np.ndarray:
    """
    Boolean mask: True for candidates inside the corpus convex hull,
    shrunk inward by `margin` (fraction of the larger bounding box dimension).
    Falls back to all-True if hull computation fails.
    """
    from scipy.spatial import Delaunay

    if len(points) < 3:
        return np.ones(len(candidates), dtype=bool)

    try:
        centroid = points.mean(axis=0)
        x_range = float(np.ptp(points[:, 0]))
        y_range = float(np.ptp(points[:, 1]))
        shrink = margin * max(x_range, y_range, 1e-8)
        directions = points - centroid
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        shrunk = points - directions / norms * shrink
        hull = Delaunay(shrunk)
        result: np.ndarray = hull.find_simplex(candidates) >= 0
        return result
    except Exception:
        return np.ones(len(candidates), dtype=bool)


def detect_gaps(
    density_result: DensityResult,
    reduced_2d: np.ndarray,
    item_names: list[str],
    isolation_min: float = 0.3,
    max_gaps: int = 7,
    n_nearest: int = 3,
    output_path: Path | None = None,
) -> list[GapRegion]:
    """
    Detect low-density gap regions in the 2D semantic space.

    Args:
        density_result:  from density.estimate_density
        reduced_2d:      float32 array of shape (n, 2)
        item_names:      list of document names, length n
        isolation_min:   minimum isolation score to qualify as a gap
        max_gaps:        maximum number of gap regions to return
        n_nearest:       number of nearest corpus items to record per gap
        output_path:     if provided, write gaps.json here

    Returns:
        List of GapRegion, sorted by isolation_score descending.

    Raises:
        GapsError: on input validation failure
    """
    if reduced_2d.ndim != 2 or reduced_2d.shape[1] != 2:
        raise GapsError(
            f"Expected reduced_2d of shape (n, 2), got {reduced_2d.shape}"
        )
    n = reduced_2d.shape[0]
    if len(item_names) != n:
        raise GapsError(
            f"item_names length ({len(item_names)}) does not match "
            f"reduced_2d row count ({n})"
        )

    grid = density_result.grid_density

    # 1. Find local minima
    minima_mask = _find_local_minima(grid)
    minima_yx = np.argwhere(minima_mask)

    if minima_yx.shape[0] == 0:
        print(
            "No gap regions detected — corpus may be too small or "
            "uniformly distributed.",
            file=sys.stderr,
        )
        _write_gaps([], output_path)
        return []

    # 2. Convert minima grid indices to 2D data coordinates
    minima_coords = np.column_stack([
        density_result.grid_x[minima_yx[:, 1]],
        density_result.grid_y[minima_yx[:, 0]],
    ]).astype(np.float32)

    # 3. Discard edge artifacts — keep only candidates inside the corpus hull
    hull_mask = _inside_convex_hull(reduced_2d, minima_coords, margin=0.15)
    if hull_mask.any():
        minima_yx = minima_yx[hull_mask]
        minima_coords = minima_coords[hull_mask]
    else:
        print(
            "Note: no interior gap candidates found — corpus may be too small "
            "or uniformly distributed. Returning edge candidates.",
            file=sys.stderr,
        )

    if minima_yx.shape[0] == 0:
        _write_gaps([], output_path)
        return []

    # 4. Score each hull-filtered minimum relative to nearest corpus points
    scores = np.array(
        [
            _isolation_score(
                centroid_2d=minima_coords[i],
                reduced_2d=reduced_2d,
                point_density=density_result.point_density,
                n_context=5,
            )
            for i in range(len(minima_yx))
        ],
        dtype=np.float64,
    )

    # 5. Non-maximum suppression with dynamic separation radius
    grid_size = density_result.grid_density.shape[0]
    min_sep = max(5, grid_size // max(len(item_names) // 2, 1))
    kept_idx = _suppress_nearby_minima(minima_yx, scores, min_separation=min_sep)

    # 6. Filter by isolation_min
    # Strict >: isolation_min=1.0 is the sentinel for "no possible gap"
    # since _isolation_score is clipped to [0, 1].
    kept_idx = kept_idx[scores[kept_idx] > isolation_min]

    if kept_idx.shape[0] == 0:
        print(
            "No gap regions detected — corpus may be too small or "
            "uniformly distributed.",
            file=sys.stderr,
        )
        _write_gaps([], output_path)
        return []

    # 7. Limit to max_gaps (already ordered descending by score)
    kept_idx = kept_idx[:max_gaps]

    # 8. Build GapRegion list
    regions: list[GapRegion] = []
    for gap_id, idx in enumerate(kept_idx):
        cx = float(minima_coords[idx][0])
        cy = float(minima_coords[idx][1])
        centroid = np.array([cx, cy], dtype=np.float32)

        names, dists = _nearest_items(centroid, reduced_2d, item_names, n=n_nearest)
        radius = round(float(np.mean(dists)) / 2.0, 6)

        regions.append(GapRegion(
            gap_id=gap_id,
            isolation_score=round(float(scores[idx]), 6),
            centroid_2d=(cx, cy),
            radius=radius,
            nearest_items=names,
            nearest_item_distances=dists,
        ))

    # 9. Deduplicate by nearest_items — adjacent cells mapping to the same
    # corpus neighbours produce identical candidates downstream
    seen_bounds: set[tuple[str, ...]] = set()
    deduped: list[GapRegion] = []
    for r in regions:
        key = tuple(sorted(r.nearest_items))
        if key not in seen_bounds:
            seen_bounds.add(key)
            deduped.append(r)
    # Re-assign sequential gap_ids after dedup
    regions = [
        GapRegion(
            gap_id=i,
            isolation_score=r.isolation_score,
            centroid_2d=r.centroid_2d,
            radius=r.radius,
            nearest_items=r.nearest_items,
            nearest_item_distances=r.nearest_item_distances,
        )
        for i, r in enumerate(deduped)
    ]

    print(
        f"Gap detection: {len(regions)} region(s) found "
        f"(isolation_min={isolation_min}, max_gaps={max_gaps})",
        file=sys.stderr,
    )

    _write_gaps(regions, output_path)
    return regions


def _write_gaps(regions: list[GapRegion], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in regions]
    output_path.write_text(json.dumps(payload, indent=2))
