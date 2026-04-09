"""k-NN density estimation and grid interpolation over the 2D reduced space."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class DensityResult:
    point_density: np.ndarray   # shape (n,), density score per corpus point
    grid_density: np.ndarray    # shape (grid_size, grid_size), smoothed surface
    grid_x: np.ndarray          # shape (grid_size,), x-axis coordinates
    grid_y: np.ndarray          # shape (grid_size,), y-axis coordinates
    k: int                      # k used for k-NN density
    sigma: float                # gaussian smoothing sigma applied to grid


class DensityError(Exception):
    """Raised when density estimation fails."""


def _knn_density(points: np.ndarray, k: int) -> np.ndarray:
    """
    Compute per-point density as 1 / mean_knn_distance.
    Returns float32 array of shape (n,).
    """
    n = points.shape[0]
    k_actual = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_actual)
    nn.fit(points)
    distances, _ = nn.kneighbors(points)
    # distances[:, 0] is always 0 (self) — skip it
    if distances.shape[1] > 1:
        mean_dist: np.ndarray = distances[:, 1:].mean(axis=1)
    else:
        mean_dist = distances[:, 0]
    # Avoid division by zero
    mean_dist = np.clip(mean_dist, 1e-8, None)
    return (1.0 / mean_dist).astype(np.float32)


def _grid_density(
    points: np.ndarray,
    point_density: np.ndarray,
    grid_size: int,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate point density onto a regular grid and smooth.

    Returns:
        grid_density: shape (grid_size, grid_size)
        grid_x:       shape (grid_size,)
        grid_y:       shape (grid_size,)
    """
    margin = 0.05
    x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())

    x_range = x_max - x_min or 1.0
    y_range = y_max - y_min or 1.0
    x_min -= margin * x_range
    x_max += margin * x_range
    y_min -= margin * y_range
    y_max += margin * y_range

    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    gx, gy = np.meshgrid(grid_x, grid_y)

    # linear interpolation requires a Delaunay triangulation (≥3 non-colinear
    # points). Fall back to nearest-neighbour for very small corpora.
    method = "linear" if len(points) >= 3 else "nearest"
    grid_z = griddata(
        points=points,
        values=point_density,
        xi=(gx, gy),
        method=method,
        fill_value=0.0,
    )

    grid_z_smooth = gaussian_filter(grid_z.astype(np.float64), sigma=sigma)
    return grid_z_smooth.astype(np.float32), grid_x, grid_y


def estimate_density(
    reduced_2d: np.ndarray,
    k: int = 5,
    grid_size: int = 64,
    sigma: float = 1.5,
) -> DensityResult:
    """
    Estimate density over the 2D reduced space.

    Args:
        reduced_2d:  float32 array of shape (n, 2)
        k:           number of neighbours for k-NN density
        grid_size:   resolution of the interpolated density grid
        sigma:       gaussian smoothing sigma applied to grid density

    Returns:
        DensityResult

    Raises:
        DensityError: on invalid input
    """
    if reduced_2d.ndim != 2 or reduced_2d.shape[1] != 2:
        raise DensityError(
            f"Expected array of shape (n, 2), got {reduced_2d.shape}"
        )
    n = reduced_2d.shape[0]
    if n < 2:
        raise DensityError(
            f"Need at least 2 points for density estimation, got {n}"
        )

    if k >= n:
        clamped = max(1, n - 1)
        print(
            f"Warning: density k clamped from {k} to {clamped} "
            f"(corpus size: {n})",
            file=sys.stderr,
        )
        k = clamped

    point_density = _knn_density(reduced_2d, k)
    grid_z, grid_x, grid_y = _grid_density(reduced_2d, point_density, grid_size, sigma)

    print(
        f"Density estimated: k={k}, grid={grid_size}×{grid_size}, sigma={sigma}",
        file=sys.stderr,
    )

    return DensityResult(
        point_density=point_density,
        grid_density=grid_z,
        grid_x=grid_x,
        grid_y=grid_y,
        k=k,
        sigma=sigma,
    )
