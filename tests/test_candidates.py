"""Tests for egf.candidates."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from egf.candidates import Candidate, generate_candidates
from egf.domain import parse_domain
from egf.gaps import GapRegion
from egf.loader import Document
from egf.quality import QualityReport


def make_docs(n: int = 10) -> list[Document]:
    topics = [
        "logic reasoning deduction inference formal",
        "ethics virtue duty obligation moral",
        "knowledge belief justification truth evidence",
        "existence substance causation ontology reality",
        "beauty art perception aesthetic experience",
        "justice rights freedom political state power",
        "consciousness mind qualia perception mental",
        "language meaning reference semantics pragmatics",
        "science method evidence explanation causation",
        "experience intentionality consciousness lived body",
    ]
    return [
        Document(name=f"doc_{i:02d}.md",
                 text=(topics[i % len(topics)] + " ") * 20)
        for i in range(n)
    ]


def make_reduced(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).astype(np.float32)


def make_gaps(n: int = 2) -> list[GapRegion]:
    return [
        GapRegion(
            gap_id=i,
            isolation_score=0.7 - i * 0.1,
            centroid_2d=(0.5 + i * 0.3, 0.5),
            radius=0.2,
            nearest_items=[f"doc_0{i}.md", f"doc_0{i + 1}.md"],
            nearest_item_distances=[0.3, 0.4],
        )
        for i in range(n)
    ]


def make_quality(trustworthiness: float = 0.85) -> QualityReport:
    return QualityReport(
        trustworthiness=trustworthiness,
        continuity=0.80,
        lcmc=0.65,
        warning=False,
        warning_message=None,
    )


# ── Happy path — vocabulary mode ──────────────────────────────────────────────

def test_returns_list_of_candidates() -> None:
    result = generate_candidates(
        gaps=make_gaps(2),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
    )
    assert isinstance(result, list)
    assert all(isinstance(c, Candidate) for c in result)


def test_one_candidate_per_gap() -> None:
    gaps = make_gaps(3)
    result = generate_candidates(
        gaps=gaps,
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
    )
    assert len(result) == 3


def test_candidates_sorted_by_confidence_descending() -> None:
    result = generate_candidates(
        gaps=make_gaps(3),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
    )
    scores = [c.confidence_score for c in result]
    assert scores == sorted(scores, reverse=True)


def test_ranks_sequential_from_zero() -> None:
    result = generate_candidates(
        gaps=make_gaps(3),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
    )
    assert [c.rank for c in result] == list(range(len(result)))


def test_generation_mode_vocabulary_when_no_llm() -> None:
    result = generate_candidates(
        gaps=make_gaps(2),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
        use_llm=False,
    )
    for c in result:
        assert c.generation_mode == "vocabulary"


def test_confidence_score_formula() -> None:
    gaps = [GapRegion(
        gap_id=0,
        isolation_score=0.8,
        centroid_2d=(0.5, 0.5),
        radius=0.2,
        nearest_items=["doc_00.md"],
        nearest_item_distances=[0.3],
    )]
    quality = make_quality(trustworthiness=0.9)
    result = generate_candidates(
        gaps=gaps,
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=quality,
        domain=parse_domain("software-tool"),
    )
    assert len(result) == 1
    assert abs(result[0].confidence_score - round(0.8 * 0.9, 4)) < 0.001


# ── Empty gaps → empty candidates ────────────────────────────────────────────

def test_empty_gaps_returns_empty_list() -> None:
    result = generate_candidates(
        gaps=[],
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
    )
    assert result == []


def test_empty_gaps_writes_empty_json(tmp_path: Path) -> None:
    out = tmp_path / "candidates.json"
    generate_candidates(
        gaps=[],
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
        output_path=out,
    )
    assert json.loads(out.read_text()) == []


# ── JSON output ───────────────────────────────────────────────────────────────

def test_writes_candidates_json(tmp_path: Path) -> None:
    out = tmp_path / "candidates.json"
    generate_candidates(
        gaps=make_gaps(2),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
        output_path=out,
    )
    assert out.exists()


def test_candidates_json_schema(tmp_path: Path) -> None:
    out = tmp_path / "candidates.json"
    generate_candidates(
        gaps=make_gaps(1),
        documents=make_docs(),
        reduced_2d=make_reduced(),
        quality_report=make_quality(),
        domain=parse_domain("philosophy"),
        output_path=out,
    )
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    c = data[0]
    for field in ["rank", "candidate_name", "function_summary",
                  "positioning_summary", "confidence_score",
                  "gap_id", "bounding_items", "generation_mode"]:
        assert field in c, f"Missing field: {field}"


# ── LLM fallback on unavailable server ───────────────────────────────────────

def test_llm_fallback_when_health_check_fails() -> None:
    with patch("egf.candidates.health_check", return_value=False):
        result = generate_candidates(
            gaps=make_gaps(1),
            documents=make_docs(),
            reduced_2d=make_reduced(),
            quality_report=make_quality(),
            domain=parse_domain("philosophy"),
            use_llm=True,
        )
    assert len(result) == 1
    assert result[0].generation_mode == "vocabulary"


def test_llm_mode_when_health_check_passes() -> None:
    from egf.llm import LLMCandidate
    mock_candidate = LLMCandidate(
        name="Transcendental Pragmatism",
        function_summary="Bridges a priori structures with practical experience.",
        positioning_summary="Occupies the gap between Kant and Dewey.",
    )
    with (
        patch("egf.candidates.health_check", return_value=True),
        patch("egf.candidates.llm_generate", return_value=mock_candidate),
    ):
        result = generate_candidates(
            gaps=make_gaps(1),
            documents=make_docs(),
            reduced_2d=make_reduced(),
            quality_report=make_quality(),
            domain=parse_domain("philosophy"),
            use_llm=True,
        )
    assert len(result) == 1
    assert result[0].generation_mode == "llm"
    assert result[0].candidate_name == "Transcendental Pragmatism"
