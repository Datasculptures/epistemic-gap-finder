"""Tests for egf.vocabulary."""

from __future__ import annotations

import numpy as np

from egf.loader import Document
from egf.vocabulary import VocabularyResult, build_vocabulary_index, project_gap


def make_docs(n: int = 10) -> list[Document]:
    topics = [
        "logic reasoning inference deduction formal systems",
        "ethics morality virtue duty obligation",
        "epistemology knowledge belief justification truth",
        "metaphysics ontology existence substance causation",
        "aesthetics beauty art perception experience",
        "political philosophy justice rights freedom state",
        "philosophy of mind consciousness perception qualia",
        "philosophy of language meaning reference semantics",
        "philosophy of science method evidence explanation",
        "phenomenology experience consciousness intentionality",
    ]
    return [
        Document(name=f"item_{i:02d}.md", text=(topics[i % len(topics)] + " ") * 20)
        for i in range(n)
    ]


def make_reduced(n: int = 10, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)).astype(np.float32)


# ── build_vocabulary_index ────────────────────────────────────────────────────

def test_build_returns_fitted_vectorizer() -> None:
    docs = make_docs()
    vec = build_vocabulary_index(docs)
    assert hasattr(vec, "vocabulary_")
    assert len(vec.vocabulary_) > 0


def test_vectorizer_has_features() -> None:
    docs = make_docs()
    vec = build_vocabulary_index(docs)
    features = vec.get_feature_names_out()
    assert len(features) > 0


# ── project_gap ───────────────────────────────────────────────────────────────

def test_project_gap_returns_result() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec)
    assert isinstance(result, VocabularyResult)


def test_terms_are_strings() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec, n_terms=5)
    assert all(isinstance(t, str) for t in result.terms)


def test_terms_count_respects_n_terms() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec, n_terms=3)
    assert len(result.terms) <= 3


def test_source_items_are_doc_names() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec, n_context_docs=2)
    for name in result.source_items:
        assert name in [d.name for d in docs]


def test_source_items_count_respects_n_context_docs() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec, n_context_docs=2)
    assert len(result.source_items) <= 2


def test_n_terms_stored() -> None:
    docs = make_docs()
    pts = make_reduced(len(docs))
    vec = build_vocabulary_index(docs)
    result = project_gap((0.5, 0.5), pts, docs, vec, n_terms=6)
    assert result.n_terms == len(result.terms)


def test_vectorizer_reuse_consistent() -> None:
    """Same vectorizer applied to two centroids returns different source items."""
    docs = make_docs(10)
    pts = make_reduced(10)
    pts[:5] = pts[:5] * 0.1
    pts[5:] = pts[5:] * 0.1 + 1.8
    vec = build_vocabulary_index(docs)
    result_a = project_gap((0.05, 0.05), pts, docs, vec)
    result_b = project_gap((1.85, 1.85), pts, docs, vec)
    assert result_a.source_items != result_b.source_items
