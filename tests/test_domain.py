"""Tests for egf.domain."""

from __future__ import annotations

import pytest

from egf.domain import DOMAIN_REGISTRY, DomainTemplate, parse_domain

# ── Registry ──────────────────────────────────────────────────────────────────

def test_all_builtin_domains_present() -> None:
    expected = {
        "concept", "software-tool", "philosophy", "vehicle", "genre", "discipline"
    }
    assert set(DOMAIN_REGISTRY.keys()) == expected


def test_all_templates_are_frozen_dataclasses() -> None:
    for template in DOMAIN_REGISTRY.values():
        assert isinstance(template, DomainTemplate)


def test_all_templates_have_non_empty_fields() -> None:
    for name, t in DOMAIN_REGISTRY.items():
        assert t.name, f"{name}: name is empty"
        assert t.label_noun, f"{name}: label_noun is empty"
        assert t.label_plural, f"{name}: label_plural is empty"
        assert t.system_prompt_fragment, f"{name}: system_prompt_fragment is empty"
        assert t.describe_format_text, f"{name}: describe_format_text is empty"


def test_describe_format_text_mentions_boundary_condition() -> None:
    for name, t in DOMAIN_REGISTRY.items():
        assert "boundary" in t.describe_format_text.lower(), (
            f"{name}: describe_format_text does not mention boundary condition"
        )


# ── parse_domain — built-in ───────────────────────────────────────────────────

@pytest.mark.parametrize("key", [
    "concept", "software-tool", "philosophy", "vehicle", "genre", "discipline",
])
def test_parse_builtin_domains(key: str) -> None:
    t = parse_domain(key)
    assert t.name == key


def test_parse_is_case_insensitive() -> None:
    t = parse_domain("Philosophy")
    assert t.name == "philosophy"


# ── parse_domain — custom ─────────────────────────────────────────────────────

def test_custom_domain_returns_template() -> None:
    t = parse_domain("custom:musical instrument")
    assert t.label_noun == "musical instrument"


def test_custom_domain_label_plural() -> None:
    t = parse_domain("custom:bridge")
    assert "bridge" in t.label_plural


def test_custom_domain_system_prompt_contains_noun() -> None:
    t = parse_domain("custom:cooking technique")
    assert "cooking technique" in t.system_prompt_fragment


def test_custom_domain_describe_format_contains_noun() -> None:
    t = parse_domain("custom:enzyme")
    assert "enzyme" in t.describe_format_text


def test_custom_already_plural_noun() -> None:
    """Regression: 'genres' was pluralised as 'genress'."""
    t = parse_domain("custom:genres")
    assert not t.label_plural.endswith("ss")
    assert t.label_plural == "genres"
    assert t.label_noun == "genre"


def test_custom_singular_noun_pluralised() -> None:
    t = parse_domain("custom:vehicle")
    assert t.label_plural == "vehicles"
    assert t.label_noun == "vehicle"


def test_custom_multi_word_already_plural() -> None:
    t = parse_domain("custom:Science Fiction and Fantasy Genres")
    assert not t.label_plural.endswith("ss")


def test_custom_domain_empty_noun_raises() -> None:
    with pytest.raises(ValueError, match="requires a noun"):
        parse_domain("custom:")


def test_custom_domain_whitespace_noun_raises() -> None:
    with pytest.raises(ValueError, match="requires a noun"):
        parse_domain("custom:   ")


# ── parse_domain — unknown ────────────────────────────────────────────────────

def test_unknown_domain_raises() -> None:
    with pytest.raises(ValueError, match="Unknown domain"):
        parse_domain("nonsense")


def test_unknown_domain_error_lists_options() -> None:
    with pytest.raises(ValueError, match="software-tool"):
        parse_domain("bogus")
