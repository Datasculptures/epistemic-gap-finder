"""Scaffold smoke test — verifies the package is importable and versioned."""

import egf


def test_version_string() -> None:
    assert egf.__version__ == "0.1.3.1"
