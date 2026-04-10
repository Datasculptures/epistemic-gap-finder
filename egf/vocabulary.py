"""TF-IDF vocabulary index and gap centroid projection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from egf.loader import Document


@dataclass(frozen=True)
class VocabularyResult:
    terms: list[str]          # ranked terms, most relevant first
    source_items: list[str]   # document names that contributed terms
    n_terms: int              # number of terms returned


def build_vocabulary_index(documents: list[Document]) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectorizer on the full corpus.

    Returns a fitted TfidfVectorizer for reuse across multiple gaps.
    """
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    vectorizer.fit([doc.text for doc in documents])
    return vectorizer


def project_gap(
    gap_centroid_2d: tuple[float, float],
    reduced_2d: np.ndarray,
    documents: list[Document],
    vectorizer: TfidfVectorizer,
    n_context_docs: int = 3,
    n_terms: int = 8,
) -> VocabularyResult:
    """
    Project a gap centroid onto the vocabulary space.

    Finds the n_context_docs corpus points nearest to the centroid,
    extracts their highest TF-IDF terms, and returns a ranked term list.
    """
    centroid = np.array(gap_centroid_2d, dtype=np.float32).reshape(1, -1)
    k = min(n_context_docs, len(documents))
    nn = NearestNeighbors(n_neighbors=k).fit(reduced_2d)
    _, indices = nn.kneighbors(centroid)

    context_docs = [documents[i] for i in indices[0]]
    context_texts = [doc.text for doc in context_docs]
    source_names = [doc.name for doc in context_docs]

    tfidf_matrix = vectorizer.transform(context_texts).toarray()
    mean_scores: np.ndarray = tfidf_matrix.mean(axis=0)

    feature_names: list[str] = vectorizer.get_feature_names_out().tolist()
    top_indices = mean_scores.argsort()[::-1][:n_terms]
    top_terms = [feature_names[i] for i in top_indices if mean_scores[i] > 0]

    return VocabularyResult(
        terms=top_terms,
        source_items=source_names,
        n_terms=len(top_terms),
    )
