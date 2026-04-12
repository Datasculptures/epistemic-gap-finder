# Epistemic Gap Finder — How-To Guide

A complete guide to running EGF from corpus to candidates.

---

## What EGF does

EGF takes a directory of plain-text description files, embeds them into a
shared semantic space, finds the low-density regions — the conceptual
territory your corpus does not cover — and generates ranked candidate
descriptions for what could occupy those gaps.

It works on any domain. You provide the descriptions. EGF finds the deserts.

---

## Before you begin

**Requirements:**
- Python 3.10, 3.11, or 3.12 (not 3.13 or 3.14 — wheel gaps exist)
- Git
- Internet connection for first run only (model download ~90 MB)
- Optional: [ollama](https://ollama.com) for LLM-enhanced candidates

**Activate the virtual environment before every session:**

```powershell
cd "C:\path\to\EpistemicGapFinder"
.venv\Scripts\Activate.ps1
```

You should see `(.venv)` at the start of your prompt. If you don't, the
environment is not active and EGF commands will not work.

---

## Step 1 — Write your corpus

Create a directory and add one `.md` file per concept. Minimum 7 files,
maximum around 50. Ten to twenty is the sweet spot.

**Get the description template for your domain first:**

```powershell
egf analyse my_corpus --domain concept --describe-format
```

This prints the four-sentence template and exits. No analysis runs.

**The four-sentence format:**

Every description should have exactly four sentences:

1. **What it is or does.** The primary function or identity. One sentence,
   active voice, plain language.

2. **What it operates on.** The inputs, subject matter, or domain it
   engages with.

3. **What it produces.** The output, result, or effect.

4. **The boundary condition.** What it explicitly does NOT do, cover, or
   include. This is the most important sentence. It is what precisely
   positions the concept in the space. Do not skip it even if it feels
   obvious.

**Example — a tabletop RPG:**

```
Traveller is a science fiction tabletop roleplaying game focused on
interstellar travel, trade, and survival in a hard science fiction
universe with minimal fantasy elements.

It operates on a detailed subsector-generation system, life path character
creation, and encounter-driven exploration of space and alien worlds.

It produces long-form campaigns of spacefaring adventure, mercenary work,
and political intrigue across a persistent, procedurally generated galaxy.

Traveller does not include fantasy magic, superheroes, or horror as design
pillars, and it does not provide a strongly authored narrative arc —
emergent play from sandbox systems is its primary mode.
```

**Practical rules:**

- One file per concept. Name it after the concept: `traveller.md`, `dnd.md`
- Be consistent across all files — same length, same structure
- The boundary condition sentence does the positioning work. Write it carefully
- Do not pad with extra sentences. Four is enough.
- Save all files as UTF-8 (in VS Code: bottom-right corner → encoding label
  → Save with Encoding → UTF-8)

---

## Step 2 — Choose your domain

The `--domain` flag sets the labels in the report and shapes the LLM prompt.

| Flag value | Report label | Best for |
|---|---|---|
| `concept` (default) | concept | Generic — works for anything |
| `software-tool` | software tool | Developer tools, CLIs, libraries |
| `philosophy` | philosophical position | Schools of thought |
| `vehicle` | vehicle type | Cars, aircraft, vessels |
| `genre` | genre | Music, literature, film |
| `discipline` | academic discipline | Research fields |
| `custom:tabletop rpg` | tabletop RPG | Any noun you specify |

If you forget `--domain`, the default is `concept` — neutral labels that
work for any corpus without being misleading.

---

## Step 3 — Run the analysis

**Basic run:**

```powershell
egf analyse my_corpus --domain concept --open
```

**With a specific domain:**

```powershell
egf analyse my_corpus --domain "custom:tabletop rpg" --open
```

**With LLM candidates (requires ollama — see Step 4):**

```powershell
egf analyse my_corpus --domain "custom:tabletop rpg" --llm --llm-model llama3.2 --open
```

**What happens during the run:**

1. Files loaded and validated
2. Sentence-transformer model loaded (first run: ~90 MB download)
3. Documents embedded to 384-dimensional vectors
4. UMAP reduces to 2D and 3D
5. Quality metrics computed
6. Density surface estimated
7. Gap regions detected
8. Candidates generated (vocabulary or LLM)
9. HTML report written and opened

The entire run takes 20–90 seconds depending on corpus size and whether
the model is already cached.

**Output files** are written to `./egf_output/` by default, or to
whatever you specify with `--output`:

| File | Contents |
|---|---|
| `report_YYYYMMDD_HHMMSS.html` | Standalone HTML report (timestamped) |
| `embeddings.npy` | Raw sentence embeddings |
| `reduced_2d.npy` | UMAP 2D positions |
| `reduced_3d.npy` | UMAP 3D positions |
| `quality.json` | Quality metrics |
| `gaps.json` | Detected gap regions |
| `candidates.json` | Generated candidate descriptions |

Each run overwrites the `.npy` and `.json` files but creates a new
timestamped HTML report — so your reports accumulate across runs.

---

## Step 4 — LLM-enhanced candidates (optional)

Vocabulary-only candidates are sparse — assembled from TF-IDF term
projections. The LLM path produces readable, paragraph-quality descriptions.

**Install ollama:** https://ollama.com (Windows installer, ~30 seconds)

**Pull a model** (do this once):

```powershell
ollama pull llama3.2
```

This downloads ~2 GB. Wait for it to complete.

**Start the ollama server** (keep this window open):

```powershell
ollama serve
```

**Warm up the model** before running EGF (prevents timeout on first request):

```powershell
ollama run llama3.2 "Hello"
```

Wait for a response, then close the `ollama run` session (`/bye` or Ctrl-D).

**Run EGF with LLM:**

```powershell
egf analyse my_corpus --domain "custom:tabletop rpg" --llm --llm-model llama3.2 --open
```

**Generation mode badges** in the report show what was used per candidate:
- `vocab` — vocabulary projection only (no LLM)
- `llm` — LLM generated successfully
- `llm→vocab` — LLM attempted but failed, fell back to vocabulary

---

## Step 5 — Read the report

Open `egf_output/report_*.html` in any browser. The report requires an
internet connection only to load the Plotly map library.

**Header** — corpus size, domain, model, number of gaps and candidates found.

**Reduction Quality** — three scores:

| Score | Good | Acceptable | Warning |
|---|---|---|---|
| Trustworthiness | ≥ 0.85 | ≥ 0.75 | < 0.75 |
| Continuity | ≥ 0.85 | ≥ 0.70 | < 0.70 |
| LCMC | ≥ 0.50 | ≥ 0.20 | < 0.20 |

Low trustworthiness means the 2D layout may be misleading and gap positions
should be treated as approximate. This is common with small corpora (< 15
items). Adding more items or using `--n-neighbors 5` often helps.

**Semantic Map** — interactive Plotly scatter. Blue dots are corpus items,
orange circles are gap regions. Hover over any point for its name and
isolation score. Zoom and pan with mouse. If the map is blank, you are
offline.

**Gap Regions** — table of detected gaps ranked by isolation score. Higher
isolation score = more absent from the corpus space. The "Nearest items"
column shows what corpus items bound each gap — these are the concepts the
gap sits between.

**Candidate descriptions** — one card per gap. Each card shows:
- Name (generated by vocabulary or LLM)
- Function summary (what it does)
- Positioning statement (where it sits relative to the bounding items)
- Confidence score (isolation × trustworthiness)
- Generation mode badge

**Methodology note** — always read this. All gaps are absent from *this
corpus*, not from the world.

---

## Tuning for better results

**If no gaps are found:**

```powershell
# Lower the isolation threshold
egf analyse my_corpus --domain concept --isolation-min 0.1 --open
```

**If the quality warning fires (trustworthiness < 0.75):**

```powershell
# Lower n-neighbors for small corpora
egf analyse my_corpus --domain concept --n-neighbors 5 --open
```

**If candidates are sparse or repetitive:**

- Add more items to the corpus — 15–20 is better than 10
- Make sure descriptions are diverse — similar items cluster together
  and leave no interior gaps
- Use `--llm` for richer candidate language

**If gap circles are at the edges of the map rather than between items:**

This is normal for small corpora where UMAP compresses everything together.
The tool falls back to edge candidates in this case. Adding more items with
diverse boundary conditions spreads the corpus in embedding space and
produces interior gaps.

---

## Common errors

**`Corpus contains only N document(s). EGF requires at least 5.`**
Add more `.md` or `.txt` files to the input directory.

**`not valid UTF-8`**
Save the file as UTF-8. In VS Code: bottom-right corner → click the
encoding label → Save with Encoding → UTF-8.

**`content too short (minimum 50 characters)`**
The file has fewer than 50 characters. Add more content.

**Scatter map is blank in the report.**
You are offline. The Plotly library loads from CDN. Everything else in
the report renders without a connection.

**`⚠ ollama not reachable`**
The ollama server is not running. Open a separate PowerShell window and
run `ollama serve`, then re-run EGF.

**LLM timeout on first gap:**
The model is cold. Run `ollama run llama3.2 "Hello"` to warm it up, then
retry EGF immediately.

**`The term 'egf' is not recognized`**
The virtual environment is not active. Run `.venv\Scripts\Activate.ps1`.

---

## Quick reference

```powershell
# Activate environment (do this every session)
.venv\Scripts\Activate.ps1

# Get description template for a domain
egf analyse my_corpus --domain concept --describe-format

# Basic run
egf analyse my_corpus --domain concept --open

# Run with custom domain
egf analyse my_corpus --domain "custom:tabletop rpg" --open

# Run with LLM candidates
egf analyse my_corpus --domain "custom:tabletop rpg" --llm --llm-model llama3.2 --open

# Small corpus — lower n-neighbors
egf analyse my_corpus --domain concept --n-neighbors 5 --open

# More gaps — lower isolation threshold
egf analyse my_corpus --domain concept --isolation-min 0.1 --open

# Separate output directory per run
egf analyse my_corpus --domain concept --output my_run_01 --open

# Demo runs
egf analyse demo\philosophy --domain philosophy --isolation-min 0.15 --llm --llm-model llama3.2 --open
egf analyse demo\software --domain software-tool --isolation-min 0.2 --open
```
