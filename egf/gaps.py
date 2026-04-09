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
    A cell is a local minimum if it is the minimum within a 3×3 window
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
    cell_density: float,
    grid_density: np.ndarray,
) -> float:
    """
    Isolation score = 1 - (cell_density / mean_density), clipped to [0, 1].
    A cell at zero density gets score 1.0. A cell at mean density gets 0.0.
    """
    mean = float(grid_density.mean())
    if mean == 0.0:
        return 0.0
    return float(np.clip(1.0 - cell_density / mean, 0.0, 1.0))


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

    # 2. Score each minimum
    scores = np.array(
        [_isolation_score(float(grid[row, col]), grid) for row, col in minima_yx],
        dtype=np.float64,
    )

    # 3. Non-maximum suppression
    kept_idx = _suppress_nearby_minima(minima_yx, scores, min_separation=5)

    # 4. Filter by isolation_min
    # Strict >: isolation_min=1.0 is the sentinel for "no possible gap"
    # since _isolation_score is clipped to [0, 1] and 1.0 is attainable
    # by empty-hull grid cells. Using > keeps semantics clean.
    kept_idx = kept_idx[scores[kept_idx] > isolation_min]

    if kept_idx.shape[0] == 0:
        print(
            "No gap regions detected — corpus may be too small or "
            "uniformly distributed.",
            file=sys.stderr,
        )
        _write_gaps([], output_path)
        return []

    # 5. Limit to max_gaps (already ordered descending by score)
    kept_idx = kept_idx[:max_gaps]

    # 6. Build GapRegion list
    regions: list[GapRegion] = []
    for gap_id, idx in enumerate(kept_idx):
        row, col = int(minima_yx[idx][0]), int(minima_yx[idx][1])
        cx = float(density_result.grid_x[col])
        cy = float(density_result.grid_y[row])
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
