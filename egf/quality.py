"""Topology-preservation quality metrics for dimensionality reduction."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class QualityReport:
    trustworthiness: float
    continuity: float
    lcmc: float
    warning: bool
    warning_message: str | None


def _continuity(
    original: np.ndarray, reduced: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Continuity measures whether points that are neighbours in the original
    space remain neighbours in the reduced space.
    Symmetric counterpart to trustworthiness.
    """
    n = original.shape[0]
    k = min(n_neighbors, n - 1)

    nn_orig = NearestNeighbors(n_neighbors=k).fit(original)
    nn_red = NearestNeighbors(n_neighbors=k).fit(reduced)

    _, orig_ind = nn_orig.kneighbors(original)
    _, red_ind = nn_red.kneighbors(reduced)

    # Rank of each original neighbour in the reduced space
    total = 0.0
    for i in range(n):
        orig_set = set(orig_ind[i])
        for j in orig_set:
            # Find rank of j in reduced neighbourhood of i
            red_neighbours = list(red_ind[i])
            if j not in red_neighbours:
                # Compute rank of j in the full reduced neighbourhood of i.
                # Use n_neighbors=n so all points are present (i itself appears
                # first at distance 0 when i is in the training set).
                all_red = NearestNeighbors(n_neighbors=n).fit(reduced)
                _, full_ind = all_red.kneighbors(reduced[i].reshape(1, -1))
                where_result = np.where(full_ind[0] == j)[0]
                rank = n if len(where_result) == 0 else int(where_result[0]) + 1
                total += rank - k

    normaliser = 1.0 - (2.0 * k * (2.0 * n - 3.0 * k - 1.0)) / (n * (n - 1.0))
    if normaliser == 0:
        return 1.0
    score = 1.0 - (2.0 / (n * k * normaliser)) * total
    return float(max(0.0, min(1.0, score)))


def _lcmc(
    original: np.ndarray, reduced: np.ndarray, n_neighbors: int = 5
) -> float:
    """Local Continuity Meta-Criterion — average k-NN neighbourhood overlap."""
    n = original.shape[0]
    k = min(n_neighbors, n - 1)

    nn_orig = NearestNeighbors(n_neighbors=k).fit(original)
    nn_red = NearestNeighbors(n_neighbors=k).fit(reduced)

    _, orig_ind = nn_orig.kneighbors(original)
    _, red_ind = nn_red.kneighbors(reduced)

    overlap = sum(
        len(set(orig_ind[i]) & set(red_ind[i]))
        for i in range(n)
    )
    return float(overlap / (n * k)) - float(k) / float(n - 1)


def assess_quality(
    original: np.ndarray,
    reduced_2d: np.ndarray,
    quality_threshold: float = 0.75,
    output_path: Path | None = None,
) -> QualityReport:
    """
    Compute topology-preservation metrics for a 2D reduction.

    Args:
        original:          high-dimensional embeddings, shape (n, d)
        reduced_2d:        2D reduction, shape (n, 2)
        quality_threshold: trustworthiness floor; below this a warning is set
        output_path:       if provided, write quality.json here

    Returns:
        QualityReport dataclass

    Raises:
        ValueError: if array shapes are incompatible
    """
    if original.shape[0] != reduced_2d.shape[0]:
        raise ValueError(
            f"Row count mismatch: original has {original.shape[0]} rows, "
            f"reduced_2d has {reduced_2d.shape[0]} rows"
        )

    trustworthiness = float(
        sklearn_trustworthiness(original, reduced_2d, n_neighbors=5)
    )
    continuity = _continuity(original, reduced_2d, n_neighbors=5)
    lcmc = _lcmc(original, reduced_2d, n_neighbors=5)

    warning = trustworthiness < quality_threshold
    warning_message = (
        f"Trustworthiness score {trustworthiness:.3f} is below the threshold "
        f"{quality_threshold:.2f}. The 2D reduction may not faithfully preserve "
        "neighbourhood structure. Gap detection results should be interpreted "
        "with caution. Consider increasing --n-neighbors or using a larger corpus."
    ) if warning else None

    report = QualityReport(
        trustworthiness=trustworthiness,
        continuity=continuity,
        lcmc=lcmc,
        warning=warning,
        warning_message=warning_message,
    )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(report), indent=2))
        print(
            f"Quality report written to {output_path.name}\n"
            f"  trustworthiness: {trustworthiness:.4f}  "
            f"continuity: {continuity:.4f}  lcmc: {lcmc:.4f}",
            file=sys.stderr,
        )
        if warning:
            print(f"⚠  Quality warning: {warning_message}", file=sys.stderr)

    return report
