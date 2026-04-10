"""Tests for egf.report."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from egf.candidates import Candidate
from egf.domain import parse_domain
from egf.gaps import GapRegion
from egf.quality import QualityReport
from egf.report import ReportContext, build_report_context, render_report

if TYPE_CHECKING:
    from pathlib import Path


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_reduced_2d_file(tmp_path: Path, n: int = 10, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    pts = rng.random((n, 2)).astype(np.float32)
    path = tmp_path / "reduced_2d.npy"
    np.save(path, pts)
    return path


def make_quality(trustworthiness: float = 0.88) -> QualityReport:
    return QualityReport(
        trustworthiness=trustworthiness,
        continuity=0.84,
        lcmc=0.61,
        warning=False,
        warning_message=None,
    )


def make_gaps(n: int = 2) -> list[GapRegion]:
    return [
        GapRegion(
            gap_id=i,
            isolation_score=0.7 - i * 0.15,
            centroid_2d=(0.5 + i * 0.3, 0.5),
            radius=0.2,
            nearest_items=[f"item_0{i}.md", f"item_0{i + 1}.md"],
            nearest_item_distances=[0.3, 0.45],
        )
        for i in range(n)
    ]


def make_candidates(n: int = 2) -> list[Candidate]:
    modes = ["vocabulary", "llm", "llm_fallback"]
    return [
        Candidate(
            rank=i,
            candidate_name=f"Candidate {i}",
            function_summary=f"This candidate does thing {i}.",
            positioning_summary=f"It occupies the gap between items {i} and {i + 1}.",
            confidence_score=round(0.7 - i * 0.1, 4),
            gap_id=i,
            bounding_items=[f"item_0{i}.md"],
            generation_mode=modes[i % len(modes)],
        )
        for i in range(n)
    ]


def make_doc_names(n: int = 10) -> list[str]:
    return [f"item_{i:02d}.md" for i in range(n)]


# ── build_report_context ──────────────────────────────────────────────────────

def test_build_returns_report_context(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(),
        candidates=make_candidates(),
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert isinstance(ctx, ReportContext)


def test_context_corpus_size(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(8),
        reduced_2d_path=make_reduced_2d_file(tmp_path, n=8),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert ctx.corpus_size == 8


def test_context_domain_label(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert (
        "philosophical" in ctx.domain_label.lower()
        or "philosophical" in ctx.domain_label_plural.lower()
    )


def test_context_has_gaps_true(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(2),
        candidates=make_candidates(2),
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert ctx.has_gaps is True
    assert ctx.has_candidates is True


def test_context_has_gaps_false(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert ctx.has_gaps is False
    assert ctx.has_candidates is False


def test_quality_warning_propagated(tmp_path: Path) -> None:
    q = QualityReport(
        trustworthiness=0.55,
        continuity=0.60,
        lcmc=0.30,
        warning=True,
        warning_message="Low trustworthiness.",
    )
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=q,
        gaps=[],
        candidates=[],
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert ctx.has_quality_warning is True
    assert ctx.quality_warning_message == "Low trustworthiness."


def test_scatter_json_is_string(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    assert isinstance(ctx.scatter_json, str)
    assert isinstance(ctx.density_json, str)


def test_nearest_items_display_strips_extension(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(1),
        candidates=[],
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    for g in ctx.gaps:
        for name in g["nearest_items_display"]:
            assert not name.endswith(".md")
            assert not name.endswith(".txt")


# ── render_report ─────────────────────────────────────────────────────────────

def test_render_writes_html_file(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(2),
        candidates=make_candidates(2),
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    assert out.exists()


def test_render_produces_html(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(2),
        candidates=make_candidates(2),
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html
    assert "Epistemic Gap Finder" in html


def test_render_contains_all_candidate_names(tmp_path: Path) -> None:
    candidates = make_candidates(3)
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(3),
        candidates=candidates,
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    for c in candidates:
        assert c.candidate_name in html


def test_render_contains_quality_metrics(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(trustworthiness=0.9123),
        gaps=[],
        candidates=[],
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "0.9123" in html


def test_render_quality_warning_in_html(tmp_path: Path) -> None:
    q = QualityReport(
        trustworthiness=0.55,
        continuity=0.60,
        lcmc=0.30,
        warning=True,
        warning_message="Trust is low.",
    )
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=q,
        gaps=[],
        candidates=[],
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "Trust is low." in html


def test_render_empty_gaps_empty_state(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "empty-state" in html or "No gap regions" in html


def test_render_methodology_note_present(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("philosophy"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "absent from this corpus" in html


def test_render_domain_label_in_html(tmp_path: Path) -> None:
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=[],
        candidates=[],
        domain=parse_domain("vehicle"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "vehicle" in html.lower()


def test_render_mode_badges_in_html(tmp_path: Path) -> None:
    candidates = [
        Candidate(
            rank=0,
            candidate_name="Vocab Candidate",
            function_summary=".",
            positioning_summary=".",
            confidence_score=0.5,
            gap_id=0,
            bounding_items=["a.md"],
            generation_mode="vocabulary",
        ),
        Candidate(
            rank=1,
            candidate_name="LLM Candidate",
            function_summary=".",
            positioning_summary=".",
            confidence_score=0.4,
            gap_id=0,
            bounding_items=["b.md"],
            generation_mode="llm",
        ),
    ]
    ctx = build_report_context(
        documents_names=make_doc_names(),
        reduced_2d_path=make_reduced_2d_file(tmp_path),
        quality_report=make_quality(),
        gaps=make_gaps(1),
        candidates=candidates,
        domain=parse_domain("software-tool"),
        model_name="all-MiniLM-L6-v2",
        input_dir=tmp_path,
    )
    out = tmp_path / "report.html"
    render_report(ctx, out)
    html = out.read_text(encoding="utf-8")
    assert "vocab" in html
    assert "llm" in html
