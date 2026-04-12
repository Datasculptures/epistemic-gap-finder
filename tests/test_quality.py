"""Tests for egf.quality."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from egf.quality import QualityReport, assess_quality

if TYPE_CHECKING:
    from pathlib import Path


def make_pair(
    n: int = 30, dim: int = 16, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (original, reduced_2d) pair with controlled random state."""
    rng = np.random.default_rng(seed)
    original = rng.random((n, dim)).astype(np.float32)
    reduced = rng.random((n, 2)).astype(np.float32)
    return original, reduced


def make_perfect_pair(n: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """
    2D structure embedded in 16D with tiny noise — the 2D coordinates dominate
    all pairwise distances, so topology is faithfully preserved. Trustworthiness
    should be very high.
    """
    rng = np.random.default_rng(1)
    reduced = rng.random((n, 2)).astype(np.float32)
    # Embed in 16D: first 2 dims are the signal, remaining 14 are negligible noise
    noise = (rng.random((n, 14)) * 0.01).astype(np.float32)
    original = np.hstack([reduced, noise])
    return original, reduced


# ── Return type ───────────────────────────────────────────────────────────────

def test_returns_quality_report() -> None:
    orig, red = make_pair()
    report = assess_quality(orig, red)
    assert isinstance(report, QualityReport)


def test_all_metrics_are_floats() -> None:
    orig, red = make_pair()
    report = assess_quality(orig, red)
    assert isinstance(report.trustworthiness, float)
    assert isinstance(report.continuity, float)
    assert isinstance(report.lcmc, float)


def test_metrics_in_valid_range() -> None:
    orig, red = make_pair()
    report = assess_quality(orig, red)
    assert 0.0 <= report.trustworthiness <= 1.0
    assert 0.0 <= report.continuity <= 1.0
    # LCMC can be slightly negative for poor reductions — check upper bound only
    assert report.lcmc <= 1.0


# ── Perfect reduction has high trustworthiness ────────────────────────────────

def test_perfect_reduction_high_trustworthiness() -> None:
    orig, red = make_perfect_pair(n=40)
    report = assess_quality(orig, red)
    assert report.trustworthiness > 0.85


# ── Warning logic ─────────────────────────────────────────────────────────────

def test_no_warning_above_threshold() -> None:
    orig, red = make_perfect_pair(n=40)
    report = assess_quality(orig, red, quality_threshold=0.75)
    assert report.warning is False
    assert report.warning_message is None


def test_warning_below_threshold() -> None:
    # Random pair will typically have low trustworthiness
    orig, red = make_pair(seed=999)
    # Force warning by setting a very high threshold
    report = assess_quality(orig, red, quality_threshold=0.9999)
    assert report.warning is True
    assert report.warning_message is not None
    assert "0.9999" in report.warning_message or "threshold" in report.warning_message


def test_warning_message_mentions_trustworthiness_score() -> None:
    orig, red = make_pair()
    report = assess_quality(orig, red, quality_threshold=0.9999)
    if report.warning:
        assert str(round(report.trustworthiness, 3)) in report.warning_message


# ── JSON output ───────────────────────────────────────────────────────────────

def test_writes_quality_json(tmp_path: Path) -> None:
    orig, red = make_pair()
    out = tmp_path / "quality.json"
    assess_quality(orig, red, output_path=out)
    assert out.exists()


def test_quality_json_schema(tmp_path: Path) -> None:
    orig, red = make_pair()
    out = tmp_path / "quality.json"
    assess_quality(orig, red, output_path=out)
    data = json.loads(out.read_text())
    assert set(data.keys()) == {
        "trustworthiness", "continuity", "lcmc", "warning", "warning_message"
    }
    assert isinstance(data["trustworthiness"], float)
    assert isinstance(data["warning"], bool)


def test_no_file_without_output_path(tmp_path: Path) -> None:
    orig, red = make_pair()
    assess_quality(orig, red)  # no output_path
    assert not (tmp_path / "quality.json").exists()


# ── Input validation ──────────────────────────────────────────────────────────

def test_shape_mismatch_raises() -> None:
    orig = np.zeros((20, 16), dtype=np.float32)
    red = np.zeros((15, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        assess_quality(orig, red)


def test_continuity_not_zero_for_small_corpus() -> None:
    """Regression: continuity was returning 0.0 for 10-item corpora."""
    orig, red = make_perfect_pair(n=10)
    report = assess_quality(orig, red)
    assert report.continuity > 0.0
