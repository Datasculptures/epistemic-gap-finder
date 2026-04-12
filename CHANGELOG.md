# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
This project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.3.1] — hotfix

### Fixed
- Isolation scoring regression from v0.1.3: comparing IDW-estimated density
  against local neighbourhood mean (same points used for both numerator and
  denominator) collapsed all scores to ~0 and produced zero gaps on real
  corpora; fixed by comparing against global corpus mean density instead
- Naive pluralisation produced "genress" for custom domain nouns already
  ending in "s"; replaced with `_make_singular_plural` that detects and
  handles already-plural nouns correctly

## [0.1.3]

### Fixed
- Isolation scores were all 1.000 on every run — fixed by computing isolation
  relative to nearest corpus point densities rather than against the
  zero-filled grid mean; gap ranking is now meaningful
- "either" in positioning statement was grammatically wrong for 3+ bounding
  items — now uses "any of these" for three or more bounds
- Custom domain nouns were incorrectly title-cased (`"tabletop rpg"` →
  `"Tabletop Rpg"`) — now preserves user capitalisation with `.capitalize()`
  for first letter only

### Changed
- Default LLM timeout raised from 30s to 90s — cold model load no longer
  causes automatic fallback to vocabulary mode
- `--llm-timeout` flag added for explicit control
- Sentence-transformer, HuggingFace, and UMAP noise suppressed by default;
  shown only with `--verbose`

## [0.1.2] — patch

### Fixed
- `continuity` metric still returning 0.0 after v0.1.1 — root cause was
  wrong normaliser (`2nk` instead of `nk(2n-3k-1)` per Venna & Kaski 2006);
  also moved full-neighbourhood matrix computation outside the inner loop
  for correctness and efficiency

### Changed
- HTML reports now use timestamped filenames (`report_YYYYMMDD_HHMMSS.html`)
  so successive runs accumulate reports rather than overwriting
- Default domain changed from `software-tool` to `concept` — neutral labels
  that work for any corpus without being misleading
- Added `concept` domain to the built-in registry

### Added
- `HOW-TO.md` — detailed end-to-end guide covering corpus preparation,
  running, interpreting results, ollama setup, tuning, and common errors

## [0.1.1] — patch

### Fixed
- CI: mypy `type-arg` errors on Python 3.10 — suppressed project-wide via
  `disable_error_code = ["type-arg"]` in `pyproject.toml` (bare `np.ndarray`
  annotations are valid; full generic type args add no practical value here)
- `--open` flag crashed with `ValueError: relative paths can't be expressed
  as file URIs` — fixed by calling `report_path.resolve().as_uri()`
- Quality warning message emitted twice to stderr — removed duplicate echo
  in `cli.py`
- `continuity` metric returned 0.0 for small corpora — corrected
  `_continuity` to properly exclude self-neighbours and bound rank lookup
- Gap detection found artifacts at grid edges outside the corpus convex hull
  rather than interior deserts — gap candidates now filtered against an
  inward-shrunk convex hull of the corpus points
- Duplicate candidates from adjacent edge gaps — added dynamic suppression
  radius scaling and deduplication by nearest_items in both `gaps.py` and
  `candidates.py`

## [0.1.0] — first release

### Fixed
- Quality assessment n_neighbors clamped to `n_samples // 2 - 1` to satisfy
  sklearn's trustworthiness constraint for small corpora (surfaced by
  10-document philosophy demo)
- Minimum corpus size lowered from 7 to 5 to allow the 5-item software demo
  to run end-to-end; loader error message updated accordingly

### Added
- `egf analyse` CLI command — full pipeline from corpus to HTML report
- Domain template system: `software-tool`, `philosophy`, `vehicle`, `genre`,
  `discipline`, `custom:<noun>` — controls LLM prompts, description guidance,
  and report labels throughout
- `--describe-format` flag — prints four-sentence description template for
  any domain and exits
- `--verbose` / `-v` flag — shows detailed per-step progress output
- `--open` flag — opens the HTML report in the default browser after generation
- Loader (`egf/loader.py`) — reads `.md`/`.txt` corpora with UTF-8 validation,
  size limits, and a minimum corpus size of 7 documents
- Embedder (`egf/embedder.py`) — wraps `sentence-transformers` with offline
  model caching and finite-value validation
- Reducer (`egf/reducer.py`) — UMAP reduction to 2D and 3D, deterministic
  seed, automatic n_neighbors clamping
- Quality assessment (`egf/quality.py`) — trustworthiness, continuity, LCMC;
  quality gate warning below configurable threshold
- Density estimation (`egf/density.py`) — k-NN radius density, grid
  interpolation, Gaussian smoothing
- Gap detection (`egf/gaps.py`) — local minima, non-maximum suppression,
  isolation scoring, `gaps.json` output
- Vocabulary projection (`egf/vocabulary.py`) — TF-IDF inverse projection
  from gap centroids to ranked term lists
- Ollama client (`egf/llm.py`) — domain-aware LLM prompts, health check,
  graceful fallback, never raises
- Candidate generation (`egf/candidates.py`) — vocabulary and LLM paths,
  confidence scoring, `generation_mode` tracking
- HTML report (`egf/report.py` + `egf/templates/report.html.j2`) — dark
  terminal aesthetic, Plotly interactive scatter map, candidate cards,
  saturation indicator, methodology note
- Demo corpora: datasculptures portfolio (software domain) and ten schools
  of Western philosophy (philosophy domain)
- PowerShell demo scripts: `run_demo_software.ps1`, `run_demo_philosophy.ps1`

### Implementation notes (spec-level fixes recorded across phases)
- Phase 2: `_continuity` self-reference bug fixed (sklearn `kneighbors` on
  training data returns the point itself; guarded with `n_neighbors=n` and
  empty-where check)
- Phase 2: `make_perfect_pair` test fixture corrected (signal-first embedding
  to ensure genuine topology preservation)
- Phase 2: `continuity` score clamped to `[0, 1]` to match sklearn convention
- Phase 3: `griddata method="nearest"` fallback when corpus has fewer than 3
  non-colinear points
- Phase 3: isolation threshold uses `>` not `>=` so 1.0 acts as the
  impossible sentinel
- Phase 3: `DensityResult` import moved to `TYPE_CHECKING` block (ruff TC001)
- Phase 4: `except Exception` widening in `llm.py` to honour "never raises"
  contract
- Phase 5: `list[dict[str, object]]` annotation for mypy strict compliance

### Dependencies
- `click`, `numpy`, `scikit-learn`, `umap-learn`, `sentence-transformers`,
  `scipy`, `jinja2`, `plotly`, `httpx`
- `hdbscan` available as optional extra `[clustering]` — no prebuilt wheel
  for Python >= 3.13 on Windows at time of release

## [0.0.1] — scaffold

### Added
- Repository structure and pyproject.toml
- Ruff, mypy, pytest with coverage configured
- CI workflow (ubuntu-latest, Python 3.10 and 3.11)
- README stub and CHANGELOG
