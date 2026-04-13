"""Smoke tests for the CLI entry point."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from click.testing import CliRunner

from egf.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def test_cli_help_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "analyse" in result.output


def test_analyse_help_exits_zero() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["analyse", "--help"])
    assert result.exit_code == 0
    assert "--domain" in result.output
    assert "--isolation-min" in result.output


def test_analyse_describe_format_exits_zero(tmp_path: Path) -> None:
    corpus = pathlib.Path(str(tmp_path)) / "corpus"
    corpus.mkdir()
    (corpus / "placeholder.md").write_text("x" * 60)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["analyse", str(corpus), "--domain", "philosophy", "--describe-format"],
    )
    assert result.exit_code == 0
    assert "philosophical" in result.output.lower()


def test_isolation_min_auto_accepted(tmp_path: Path) -> None:
    """--isolation-min auto is accepted without a click parse error (exit 2)."""
    corpus = pathlib.Path(str(tmp_path)) / "corpus"
    corpus.mkdir()
    # Single-file corpus — fails at loader size check (exit 1), not at
    # click option parsing (exit 2). Confirms auto is a valid option value.
    (corpus / "placeholder.md").write_text("x" * 60)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["analyse", str(corpus), "--isolation-min", "auto"],
    )
    assert result.exit_code != 2


def test_isolation_min_invalid_string_exits_cleanly(tmp_path: Path) -> None:
    """Invalid --isolation-min string exits with code 1 and a clear message."""
    corpus = pathlib.Path(str(tmp_path)) / "corpus"
    corpus.mkdir()
    (corpus / "placeholder.md").write_text("x" * 60)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["analyse", str(corpus), "--isolation-min", "banana"],
    )
    assert result.exit_code == 1
    assert "auto" in result.output or "number" in result.output
