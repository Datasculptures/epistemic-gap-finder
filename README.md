# Epistemic Gap Finder (EGF)

A conceptual cartography instrument. Feed it a corpus of descriptions from
any categorisable domain and it maps the semantic space those concepts occupy,
identifies the low-density regions — the deserts — and generates ranked
candidate descriptions for what could inhabit those gaps.

**It is not a search engine. It is not a recommendation system.**
It is a strategic positioning instrument for anyone who wants to know what
is structurally absent from a space before committing to a direction.

---

## What it does

1. Embeds a directory of plain-text or markdown descriptions using a local
   sentence-transformer model (offline after the first run)
2. Reduces to 2D and 3D using UMAP, assessing topology preservation with
   trustworthiness, continuity, and LCMC metrics
3. Estimates density across the 2D space using k-NN radius density
4. Detects low-density gap regions via local minima on the smoothed density surface
5. Generates ranked candidate descriptions for each gap — either from
   vocabulary projection (offline) or a local LLM via ollama (optional)
6. Renders a standalone HTML report with an interactive map, gap table,
   and candidate cards

---

## Domains

EGF is domain-agnostic. Built-in templates:

| Domain | Label | Example use |
|---|---|---|
| `software-tool` (default) | software tool | Developer tooling landscape |
| `philosophy` | philosophical position | Schools of thought |
| `vehicle` | vehicle type | Automotive taxonomy |
| `genre` | genre | Musical or literary genres |
| `discipline` | academic discipline | Research field mapping |
| `custom:<noun>` | your noun | Anything else |

---

## Installation

Requires Python 3.10+.

```powershell
git clone https://github.com/datasculptures/epistemic-gap-finder.git
cd epistemic-gap-finder
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

The first run will download the `all-MiniLM-L6-v2` sentence-transformer model
(~90 MB) to `.cache/`. Subsequent runs are fully offline.

---

## Quick start

```powershell
# See the description format for your domain before writing files
egf analyse demo\software --domain software-tool --describe-format

# Run the software demo (datasculptures portfolio)
.\demo\run_demo_software.ps1

# Run the philosophy demo (ten schools of Western philosophy)
.\demo\run_demo_philosophy.ps1
```

---

## Usage

```
egf analyse <input_dir> [OPTIONS]
```

`<input_dir>` must contain at least 7 `.md` or `.txt` files, each at least
50 characters long.

### Options

| Flag | Default | Description |
|---|---|---|
| `--output`, `-o` | `./egf_output` | Output directory |
| `--domain` | `software-tool` | Domain template |
| `--describe-format` | — | Print description template and exit |
| `--model` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `--n-neighbors` | `15` | UMAP n_neighbors |
| `--min-dist` | `0.1` | UMAP min_dist |
| `--quality-threshold` | `0.75` | Trustworthiness warning floor |
| `--density-k` | `5` | k for k-NN density |
| `--isolation-min` | `0.3` | Minimum isolation score |
| `--max-gaps` | `7` | Maximum gap regions |
| `--llm` | off | Enable ollama candidate generation |
| `--llm-model` | `llama3` | Ollama model name |
| `--llm-host` | `http://localhost:11434` | Ollama host |
| `--open` | off | Open report in browser after generation |
| `--verbose`, `-v` | off | Verbose output |

---

## Writing descriptions

EGF works best when descriptions are precise and include a boundary condition —
a sentence that says what the concept explicitly does *not* do or cover.

Run `egf analyse <dir> --domain <domain> --describe-format` to get the
four-sentence template for your domain.

**Example (software-tool domain):**

> Sentence 1 — What it does.
> Sentence 2 — What it takes as input.
> Sentence 3 — What it produces as output.
> Sentence 4 — What it explicitly does NOT do (boundary condition).

The boundary condition sentence is the most important for positioning accuracy.
A description without it places a concept in a vaguer region of the space.

---

## Output files

After a run, the output directory contains:

| File | Contents |
|---|---|
| `embeddings.npy` | float32 array, shape (n, embedding_dim) |
| `reduced_2d.npy` | float32 array, shape (n, 2) |
| `reduced_3d.npy` | float32 array, shape (n, 3) |
| `quality.json` | Trustworthiness, continuity, LCMC, warning flag |
| `gaps.json` | Gap regions ranked by isolation score |
| `candidates.json` | Candidate descriptions ranked by confidence |
| `report.html` | Standalone interactive HTML report |

---

## LLM-enhanced candidates (optional)

With `--llm`, EGF sends each gap's bounding items and vocabulary terms to a
local [ollama](https://ollama.com) instance for richer natural-language
candidate descriptions.

```powershell
# Install ollama and pull a model first
ollama pull llama3

# Then run with --llm
egf analyse demo\philosophy --domain philosophy --llm --open
```

If ollama is not running, EGF falls back to vocabulary mode automatically.
The `generation_mode` field in each candidate card records what was used:
`vocab`, `llm`, or `llm→vocab` (LLM attempted, fell back).

---

## Development

```powershell
ruff check egf tests
mypy egf
pytest
```

Coverage target: ≥ 80%.

---

## Part of the datasculptures portfolio

[datasculptures.com](https://datasculptures.com)

> "Are there deserts in vector space?"
