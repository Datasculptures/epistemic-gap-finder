"""Corpus loader — reads .md and .txt files from a directory."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB
_MIN_CHARS = 50
_MIN_CORPUS = 5


@dataclass(frozen=True)
class Document:
    name: str  # filename without directory, e.g. "stoicism.md"
    text: str  # full file content, stripped of leading/trailing whitespace


class LoaderError(Exception):
    """Raised when a file fails validation. Message includes filename only."""


def load_corpus(directory: Path) -> list[Document]:
    """
    Load all .md and .txt files from directory.

    Raises:
        FileNotFoundError: if directory does not exist
        LoaderError: if any file fails validation (reported per-file, combined)
        ValueError: if fewer than 7 valid documents are found
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory.name}")

    candidates: list[Path] = sorted(
        p
        for pattern in ("**/*.md", "**/*.txt")
        for p in directory.glob(pattern)
        if not p.name.startswith(".")
    )

    errors: list[str] = []
    documents: list[Document] = []

    for path in candidates:
        filename = path.name
        try:
            # Validation 1: size
            file_size = os.stat(path).st_size
            if file_size > _MAX_FILE_BYTES:
                errors.append(f"{filename}: file exceeds 10 MB limit")
                continue

            # Validation 2: encoding
            try:
                raw = path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                errors.append(f"{filename}: not valid UTF-8")
                continue

            # Validation 3: minimum length
            text = raw.strip()
            if len(text) < _MIN_CHARS:
                errors.append(
                    f"{filename}: content too short (minimum 50 characters)"
                )
                continue

            documents.append(Document(name=filename, text=text))

        except LoaderError:
            raise
        except OSError as e:
            errors.append(f"{filename}: {e.strerror}")

    if errors:
        joined = "\n  ".join(errors)
        raise LoaderError(
            f"{len(errors)} file(s) failed validation:\n  {joined}"
        )

    if len(documents) < _MIN_CORPUS:
        raise ValueError(
            f"Corpus contains only {len(documents)} document(s). "
            "EGF requires at least 5 to produce meaningful gap detection. "
            "Add more descriptions and re-run."
        )

    print(f"Loaded {len(documents)} documents from {directory.name}", file=sys.stderr)
    return documents
