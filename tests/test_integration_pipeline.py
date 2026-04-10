"""
Integration test: full pipeline from documents to candidates.json.

Uses vocabulary-only mode (no ollama). Sentence-transformer is mocked to
avoid the ~90 MB model download in CI.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import pytest

from egf.loader import Document

SYNTHETIC_CORPUS = [
    Document(name=f"concept_{i:02d}.md", text=(text + " ") * 30)
    for i, text in enumerate([
        "logic deduction inference formal proof reasoning symbolic",
        "ethics virtue duty obligation moral good harm right",
        "knowledge belief justification truth evidence epistemology",
        "existence substance causation ontology reality being",
        "beauty art perception aesthetic experience form style",
        "justice rights freedom political state power authority",
        "consciousness mind qualia perception mental intentional",
        "language meaning reference semantics interpretation sign",
        "science method evidence explanation theory hypothesis",
        "experience body lived world phenomenal consciousness",
        "time change becoming process duration temporal flow",
    ])
]


def mock_embedder_for(documents: list[Document]) -> np.ndarray:
    """Deterministic pseudo-embeddings that form two loose clusters."""
    rng = np.random.default_rng(42)
    n = len(documents)
    coords = np.vstack([
        rng.random((n // 2, 32)).astype(np.float32) * 0.3,
        rng.random((n - n // 2, 32)).astype(np.float32) * 0.3 + 0.7,
    ])
    return coords


@pytest.fixture()
def run_pipeline(tmp_path: Path) -> dict:  # type: ignore[type-arg]
    """Run the full pipeline programmatically and return outputs."""
    documents = SYNTHETIC_CORPUS
    embeddings = mock_embedder_for(documents)
    np.save(tmp_path / "embeddings.npy", embeddings)

    # Phase 2: reduce (mock UMAP)
    from unittest.mock import patch

    from egf.reducer import reduce_embeddings

    rng = np.random.default_rng(0)
    inst2 = MagicMock()
    inst3 = MagicMock()
    inst2.fit_transform.return_value = rng.random(
        (len(documents), 2)
    ).astype(np.float32)
    inst3.fit_transform.return_value = rng.random(
        (len(documents), 3)
    ).astype(np.float32)

    with patch("egf.reducer.umap") as mock_umap:
        mock_umap.UMAP.side_effect = [inst2, inst3]
        reduction = reduce_embeddings(embeddings, output_dir=tmp_path)

    # Phase 2: quality
    from egf.quality import assess_quality
    quality = assess_quality(
        embeddings, reduction.reduced_2d,
        output_path=tmp_path / "quality.json",
    )

    # Phase 3: density + gaps
    from egf.density import estimate_density
    from egf.gaps import detect_gaps
    density = estimate_density(
        reduction.reduced_2d, k=3, grid_size=32, sigma=1.0
    )
    gaps = detect_gaps(
        density,
        reduced_2d=reduction.reduced_2d,
        item_names=[d.name for d in documents],
        isolation_min=0.1,
        output_path=tmp_path / "gaps.json",
    )

    # Phase 4: candidates
    from egf.candidates import generate_candidates
    from egf.domain import parse_domain
    candidates = generate_candidates(
        gaps=gaps,
        documents=documents,
        reduced_2d=reduction.reduced_2d,
        quality_report=quality,
        domain=parse_domain("philosophy"),
        use_llm=False,
        output_path=tmp_path / "candidates.json",
    )

    return {
        "documents": documents,
        "embeddings": embeddings,
        "reduction": reduction,
        "quality": quality,
        "gaps": gaps,
        "candidates": candidates,
        "output_dir": tmp_path,
    }


# ── Output file existence ─────────────────────────────────────────────────────

def test_embeddings_npy_exists(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    assert (run_pipeline["output_dir"] / "embeddings.npy").exists()


def test_reduced_2d_npy_exists(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    assert (run_pipeline["output_dir"] / "reduced_2d.npy").exists()


def test_quality_json_exists(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    assert (run_pipeline["output_dir"] / "quality.json").exists()


def test_gaps_json_exists(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    assert (run_pipeline["output_dir"] / "gaps.json").exists()


def test_candidates_json_exists(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    assert (run_pipeline["output_dir"] / "candidates.json").exists()


# ── Content checks ────────────────────────────────────────────────────────────

def test_quality_json_valid(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    data = json.loads(
        (run_pipeline["output_dir"] / "quality.json").read_text()
    )
    assert "trustworthiness" in data
    assert 0.0 <= data["trustworthiness"] <= 1.0


def test_candidates_json_valid_schema(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    data = json.loads(
        (run_pipeline["output_dir"] / "candidates.json").read_text()
    )
    assert isinstance(data, list)
    if data:
        c = data[0]
        for field in ["rank", "candidate_name", "function_summary",
                      "positioning_summary", "confidence_score",
                      "gap_id", "bounding_items", "generation_mode"]:
            assert field in c


def test_candidates_all_vocabulary_mode(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    data = json.loads(
        (run_pipeline["output_dir"] / "candidates.json").read_text()
    )
    for c in data:
        assert c["generation_mode"] == "vocabulary"


def test_candidate_ranks_sequential(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    data = json.loads(
        (run_pipeline["output_dir"] / "candidates.json").read_text()
    )
    if data:
        assert [c["rank"] for c in data] == list(range(len(data)))


def test_confidence_scores_descending(run_pipeline: dict) -> None:  # type: ignore[type-arg]
    data = json.loads(
        (run_pipeline["output_dir"] / "candidates.json").read_text()
    )
    scores = [c["confidence_score"] for c in data]
    assert scores == sorted(scores, reverse=True)
