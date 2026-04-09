"""UMAP dimensionality reduction — 2D and 3D with deterministic seed."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import umap  # must be module-level so tests can patch egf.reducer.umap

if TYPE_CHECKING:
    from pathlib import Path

RANDOM_SEED = 42


@dataclass(frozen=True)
class ReductionResult:
    reduced_2d: np.ndarray   # shape (n, 2), float32
    reduced_3d: np.ndarray   # shape (n, 3), float32
    n_neighbors: int
    min_dist: float
    model_name: str          # sentence-transformer model used upstream


class ReducerError(Exception):
    """Raised when UMAP reduction fails or output is invalid."""


def reduce_embeddings(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    model_name: str = "all-MiniLM-L6-v2",
    output_dir: Path | None = None,
) -> ReductionResult:
    """
    Reduce embeddings to 2D and 3D using UMAP.

    Args:
        embeddings:  float32 array of shape (n_docs, embedding_dim)
        n_neighbors: UMAP n_neighbors parameter
        min_dist:    UMAP min_dist parameter
        model_name:  upstream model name, stored in result for provenance
        output_dir:  if provided, write reduced_2d.npy and reduced_3d.npy here

    Returns:
        ReductionResult with both reductions

    Raises:
        ReducerError: on invalid input or UMAP failure
    """
    # Validate input shape
    if embeddings.ndim != 2:
        raise ReducerError(
            f"Expected 2D embedding array, got {embeddings.ndim}D"
        )
    n_docs, n_dims = embeddings.shape
    if n_docs < 2:
        raise ReducerError(
            f"Need at least 2 documents for reduction, got {n_docs}"
        )
    if n_dims < 2:
        raise ReducerError(
            f"Embedding dimension must be at least 2, got {n_dims}"
        )

    # Clamp n_neighbors if necessary
    if n_neighbors >= n_docs:
        clamped = n_docs - 1
        print(
            f"Warning: n_neighbors clamped from {n_neighbors} to {clamped} "
            f"(corpus size: {n_docs})",
            file=sys.stderr,
        )
        n_neighbors = clamped

    try:
        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=RANDOM_SEED,
        )
        reducer_3d = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=RANDOM_SEED,
        )
        reduced_2d = np.array(reducer_2d.fit_transform(embeddings), dtype=np.float32)
        reduced_3d = np.array(reducer_3d.fit_transform(embeddings), dtype=np.float32)
    except ReducerError:
        raise
    except Exception as e:
        raise ReducerError(f"UMAP reduction failed: {e}") from e

    # Validate outputs
    if not np.isfinite(reduced_2d).all():
        raise ReducerError("UMAP produced non-finite values")
    if not np.isfinite(reduced_3d).all():
        raise ReducerError("UMAP produced non-finite values")

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "reduced_2d.npy", reduced_2d)
        np.save(output_dir / "reduced_3d.npy", reduced_3d)

    print(
        f"Reduced to 2D: shape {reduced_2d.shape} | 3D: shape {reduced_3d.shape} "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist})",
        file=sys.stderr,
    )

    return ReductionResult(
        reduced_2d=reduced_2d,
        reduced_3d=reduced_3d,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        model_name=model_name,
    )
