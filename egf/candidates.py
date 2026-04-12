"""Candidate generation — orchestrates vocabulary and LLM paths."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

# Imported at module level so tests can patch egf.candidates.health_check
# and egf.candidates.llm_generate
from egf.llm import generate_candidate as llm_generate
from egf.llm import health_check
from egf.vocabulary import VocabularyResult, build_vocabulary_index, project_gap

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from egf.domain import DomainTemplate
    from egf.gaps import GapRegion
    from egf.loader import Document
    from egf.quality import QualityReport

GenerationMode = str  # "vocabulary" | "llm" | "llm_fallback"


@dataclass(frozen=True)
class Candidate:
    rank: int
    candidate_name: str
    function_summary: str
    positioning_summary: str
    confidence_score: float
    gap_id: int
    bounding_items: list[str]
    generation_mode: GenerationMode


def _vocabulary_candidate(
    gap: GapRegion,
    vocab: VocabularyResult,
    domain: DomainTemplate,
) -> tuple[str, str, str]:
    """
    Derive candidate name, function_summary, positioning_summary from
    vocabulary terms and gap metadata alone.
    """
    top_term = vocab.terms[0].title() if vocab.terms else "Unknown"
    second_term = vocab.terms[1].title() if len(vocab.terms) > 1 else ""
    candidate_name = f"{top_term} {second_term}".strip() if second_term else top_term

    clean_bounds = [n.rsplit(".", 1)[0] for n in gap.nearest_items]
    bounds_str = ", ".join(clean_bounds)
    terms_str = ", ".join(vocab.terms[:4]) if vocab.terms else "unknown"

    function_summary = (
        f"A {domain.label_noun} operating in the space defined by "
        f"concepts such as: {terms_str}."
    )
    positioning_summary = (
        f"Occupies the gap between {bounds_str}, "
        f"addressing territory not covered by either."
    )

    return candidate_name, function_summary, positioning_summary


def generate_candidates(
    gaps: list[GapRegion],
    documents: list[Document],
    reduced_2d: np.ndarray,
    quality_report: QualityReport,
    domain: DomainTemplate,
    use_llm: bool = False,
    llm_host: str = "http://localhost:11434",
    llm_model: str = "llama3",
    n_terms: int = 8,
    n_context_docs: int = 3,
    output_path: Path | None = None,
) -> list[Candidate]:
    """
    Generate candidate descriptions for each gap region.

    Returns a list of Candidate sorted by confidence_score descending.
    Empty list if gaps is empty.
    """
    if not gaps:
        if output_path:
            output_path.write_text("[]", encoding="utf-8")
        return []

    vectorizer = build_vocabulary_index(documents)

    llm_available = False
    if use_llm:
        llm_available = health_check(llm_host)
        if not llm_available:
            print(
                f"⚠  ollama not reachable at {llm_host} — "
                "falling back to vocabulary mode",
                file=sys.stderr,
            )

    candidates: list[Candidate] = []

    for gap in gaps:
        vocab_result = project_gap(
            gap_centroid_2d=gap.centroid_2d,
            reduced_2d=reduced_2d,
            documents=documents,
            vectorizer=vectorizer,
            n_context_docs=n_context_docs,
            n_terms=n_terms,
        )

        generation_mode: GenerationMode = "vocabulary"
        candidate_name: str
        function_summary: str
        positioning_summary: str

        if use_llm and llm_available:
            llm_result = llm_generate(
                gap_id=gap.gap_id,
                bounding_items=gap.nearest_items,
                vocabulary_terms=vocab_result.terms,
                domain=domain,
                host=llm_host,
                model=llm_model,
            )
            if llm_result is not None:
                candidate_name = llm_result.name
                function_summary = llm_result.function_summary
                positioning_summary = llm_result.positioning_summary
                generation_mode = "llm"
            else:
                print(
                    f"⚠  Gap {gap.gap_id}: LLM fallback to vocabulary mode",
                    file=sys.stderr,
                )
                candidate_name, function_summary, positioning_summary = (
                    _vocabulary_candidate(gap, vocab_result, domain)
                )
                generation_mode = "llm_fallback"
        else:
            candidate_name, function_summary, positioning_summary = (
                _vocabulary_candidate(gap, vocab_result, domain)
            )

        confidence_score = round(
            gap.isolation_score * quality_report.trustworthiness, 4
        )

        candidates.append(Candidate(
            rank=0,
            candidate_name=candidate_name,
            function_summary=function_summary,
            positioning_summary=positioning_summary,
            confidence_score=confidence_score,
            gap_id=gap.gap_id,
            bounding_items=gap.nearest_items,
            generation_mode=generation_mode,
        ))

    # Deduplicate: keep only the highest-confidence candidate per unique
    # bounding item set
    seen_bounds: set[tuple[str, ...]] = set()
    deduped: list[Candidate] = []
    for c in candidates:
        key = tuple(sorted(c.bounding_items))
        if key not in seen_bounds:
            seen_bounds.add(key)
            deduped.append(c)
    candidates = deduped

    candidates.sort(key=lambda c: c.confidence_score, reverse=True)
    ranked = [
        Candidate(
            rank=i,
            candidate_name=c.candidate_name,
            function_summary=c.function_summary,
            positioning_summary=c.positioning_summary,
            confidence_score=c.confidence_score,
            gap_id=c.gap_id,
            bounding_items=c.bounding_items,
            generation_mode=c.generation_mode,
        )
        for i, c in enumerate(candidates)
    ]

    if output_path:
        output_path.write_text(
            json.dumps([asdict(c) for c in ranked], indent=2),
            encoding="utf-8",
        )
        print(f"Candidates written to {output_path.name}", file=sys.stderr)

    return ranked
