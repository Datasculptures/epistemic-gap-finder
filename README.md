# Epistemic Gap Finder (EGF)

A conceptual cartography instrument. Feed it a corpus of descriptions from
any categorisable domain — software tools, philosophical schools, vehicle types,
musical genres — and it maps the semantic space those concepts occupy, identifies
the low-density regions (the deserts), and generates ranked candidate descriptions
for what could inhabit those gaps.

**Status:** Early development — v0.0.1 scaffold.

## Installation

Requires Python 3.10+.

```powershell
git clone https://github.com/datasculptures/epistemic-gap-finder.git
cd epistemic-gap-finder
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

## Usage

```
egf analyse <input_dir> [options]
```

Full documentation and demo walkthrough will be added in v0.1.0.

## Description Format

Run the following to get a description template for your domain:

```
egf analyse <input_dir> --domain <domain> --describe-format
```

## Domains

Built-in domains: `software-tool` (default), `philosophy`, `vehicle`,
`genre`, `discipline`. Custom: `custom:<your domain noun>`.

## Development

```powershell
ruff check egf tests
mypy egf
pytest
```

### Note on hdbscan (Windows / Python 3.14+)

`hdbscan` requires a C compiler (MSVC) to build from source, and prebuilt
wheels are not available for Python 3.14 on Windows. It is declared as an
optional `clustering` extra rather than a core dependency:

```powershell
# If a wheel becomes available for your Python version:
pip install -e ".[clustering]"
# Or try binary-only:
pip install hdbscan --only-binary :all:
```

The clustering extra is not required for the Phase 0 scaffold. It will be
re-evaluated when hdbscan wheel coverage catches up to Python 3.14.

## Part of the datasculptures portfolio

[datasculptures.com](https://datasculptures.com)
