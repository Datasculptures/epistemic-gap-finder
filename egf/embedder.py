"""Embedding pipeline — wraps sentence-transformers."""

from __future__ import annotations

import logging
import math
import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from pathlib import Path

    from egf.loader import Document

DEFAULT_MODEL = "all-MiniLM-L6-v2"


class EmbedderError(Exception):
    """Raised when embedding fails or output validation fails."""


def _load_model(model_name: str) -> SentenceTransformer:
    """Load sentence-transformer model with noise suppressed."""
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SentenceTransformer(model_name)  # type: ignore[no-any-return]


def embed_corpus(
    documents: list[Document],
    model_name: str = DEFAULT_MODEL,
    output_path: Path | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    Embed a list of documents using the specified sentence-transformer model.

    Args:
        documents:   list of Document objects (non-empty)
        model_name:  sentence-transformers model identifier
        output_path: if provided, write embeddings.npy to this path
        verbose:     if True, show model load output; if False, suppress noise

    Returns:
        numpy array of shape (n_docs, embedding_dim), dtype float32

    Raises:
        EmbedderError: on empty input, model load failure, or invalid output
    """
    if not documents:
        raise EmbedderError("Cannot embed empty corpus")

    try:
        model = SentenceTransformer(model_name) if verbose else _load_model(model_name)
    except Exception as e:
        raise EmbedderError(f"Failed to load model '{model_name}': {e}") from e

    raw = model.encode(
        [doc.text for doc in documents],
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    result = np.array(raw, dtype=np.float32)

    # Validate: no non-finite values
    for v in result.flat:
        if not math.isfinite(v):
            raise EmbedderError("Embedding output contains non-finite values")

    # Validate: shape matches input
    if result.shape[0] != len(documents):
        raise EmbedderError("Embedding count does not match document count")

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, result)
        print(f"Embeddings written to {output_path.name}", file=sys.stderr)

    print(
        f"Embedded {len(documents)} documents — shape: {result.shape} "
        f"using {model_name}",
        file=sys.stderr,
    )
    return result
