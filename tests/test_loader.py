"""Tests for egf.loader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from egf.loader import Document, load_corpus

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_corpus(tmp_path: Path, n: int = 8, suffix: str = ".md") -> Path:
    """Write n valid description files to tmp_path."""
    for i in range(n):
        (tmp_path / f"item_{i:02d}{suffix}").write_text(
            f"This is item {i}. " * 10,  # well over 50 chars
            encoding="utf-8",
        )
    return tmp_path


# ── Happy path ────────────────────────────────────────────────────────────────

def test_load_returns_documents(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=8)
    docs = load_corpus(tmp_path)
    assert len(docs) == 8
    assert all(isinstance(d, Document) for d in docs)


def test_load_txt_files(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7, suffix=".txt")
    docs = load_corpus(tmp_path)
    assert len(docs) == 7


def test_load_mixed_extensions(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=4, suffix=".md")
    make_corpus(tmp_path, n=4, suffix=".txt")
    docs = load_corpus(tmp_path)
    assert len(docs) == 8


def test_documents_sorted_by_name(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    docs = load_corpus(tmp_path)
    names = [d.name for d in docs]
    assert names == sorted(names)


def test_gitkeep_ignored(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    docs = load_corpus(tmp_path)
    assert all(not d.name.startswith(".") for d in docs)


# ── Directory errors ──────────────────────────────────────────────────────────

def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nonexistent")


# ── Corpus size validation ────────────────────────────────────────────────────

def test_fewer_than_seven_raises(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=6)
    with pytest.raises(ValueError, match="at least 7"):
        load_corpus(tmp_path)


def test_exactly_seven_passes(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    docs = load_corpus(tmp_path)
    assert len(docs) == 7


# ── Per-file validation ───────────────────────────────────────────────────────

def test_too_short_raises(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    (tmp_path / "short.md").write_text("Too short.", encoding="utf-8")
    with pytest.raises(Exception, match="too short"):
        load_corpus(tmp_path)


def test_invalid_utf8_raises(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    (tmp_path / "bad_encoding.md").write_bytes(b"\xff\xfe" + b"x" * 100)
    with pytest.raises(Exception, match="UTF-8"):
        load_corpus(tmp_path)


def test_oversized_file_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_corpus(tmp_path, n=7)
    big = tmp_path / "big.md"
    big.write_text("x" * 100, encoding="utf-8")
    real_stat = os.stat

    def fake_stat(path: str | Path, **kwargs: object) -> os.stat_result:
        result = real_stat(path, **kwargs)
        if Path(path).name == "big.md":
            return os.stat_result((
                result.st_mode, result.st_ino, result.st_dev, result.st_nlink,
                result.st_uid, result.st_gid, 11 * 1024 * 1024,
                result.st_atime, result.st_mtime, result.st_ctime,
            ))
        return result

    monkeypatch.setattr(os, "stat", fake_stat)
    with pytest.raises(Exception, match="10 MB"):
        load_corpus(tmp_path)


def test_error_message_uses_filename_only(tmp_path: Path) -> None:
    make_corpus(tmp_path, n=7)
    (tmp_path / "bad.md").write_text("short", encoding="utf-8")
    with pytest.raises(Exception) as exc_info:
        load_corpus(tmp_path)
    assert str(tmp_path) not in str(exc_info.value)
    assert "bad.md" in str(exc_info.value)
