"""EGF command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from egf.domain import parse_domain
from egf.embedder import embed_corpus
from egf.loader import load_corpus


@click.command()
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False,
                path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default=Path("egf_output"), show_default=True,
              help="Output directory.")
@click.option("--model", default="all-MiniLM-L6-v2", show_default=True,
              help="Sentence-transformer model name.")
@click.option("--domain", "domain_str", default="software-tool",
              show_default=True,
              help=(
                  "Domain template. Built-in: software-tool, philosophy, "
                  "vehicle, genre, discipline. Custom: custom:<noun>."
              ))
@click.option("--describe-format", "describe_format", is_flag=True,
              default=False,
              help="Print the description template for the active domain and exit.")
@click.option("--n-neighbors", default=15, show_default=True,
              help="UMAP n_neighbors parameter.")
@click.option("--min-dist", default=0.1, show_default=True,
              help="UMAP min_dist parameter.")
@click.option("--quality-threshold", default=0.75, show_default=True,
              help="Trustworthiness floor below which a warning is issued.")
@click.option("--density-k", default=5, show_default=True,
              help="k for k-NN density estimation.")
@click.option("--isolation-min", default=0.3, show_default=True,
              help="Minimum isolation score to qualify as a gap region.")
@click.option("--max-gaps", default=7, show_default=True,
              help="Maximum number of gap regions to return.")
@click.option("--llm", "use_llm", is_flag=True, default=False,
              help="Enable ollama candidate generation (default: vocabulary only).")
@click.option("--llm-model", default="llama3", show_default=True,
              help="Ollama model name.")
@click.option("--llm-host", default="http://localhost:11434", show_default=True,
              help="Ollama host URL.")
def main(
    input_dir: Path,
    output: Path,
    model: str,
    domain_str: str,
    describe_format: bool,
    n_neighbors: int,
    min_dist: float,
    quality_threshold: float,
    density_k: int,
    isolation_min: float,
    max_gaps: int,
    use_llm: bool,
    llm_host: str,
    llm_model: str,
) -> None:
    """Epistemic Gap Finder — map the conceptual space and find what's absent."""

    # Resolve domain first — needed for --describe-format early exit
    try:
        domain = parse_domain(domain_str)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # --describe-format: print template and exit immediately
    if describe_format:
        click.echo(f"Description template — domain: {domain.label_noun}\n")
        click.echo(domain.describe_format_text)
        sys.exit(0)

    # Load corpus
    click.echo(f"Loading corpus from: {input_dir}", err=True)
    try:
        documents = load_corpus(input_dir)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading corpus:\n{e}", err=True)
        sys.exit(1)

    # Prepare output directory
    output.mkdir(parents=True, exist_ok=True)

    # Embed
    click.echo(f"Embedding {len(documents)} documents with model: {model}",
               err=True)
    try:
        embed_corpus(
            documents,
            model_name=model,
            output_path=output / "embeddings.npy",
        )
    except Exception as e:
        click.echo(f"Error during embedding: {e}", err=True)
        sys.exit(1)

    # Load embeddings from disk — on-disk array is the source of truth
    import numpy as np
    embeddings = np.load(output / "embeddings.npy")

    # Reduce
    click.echo("Reducing embeddings...", err=True)
    try:
        from egf.reducer import reduce_embeddings
        reduction = reduce_embeddings(
            embeddings,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            model_name=model,
            output_dir=output,
        )
    except Exception as e:
        click.echo(f"Error during reduction: {e}", err=True)
        sys.exit(1)

    # Assess quality
    click.echo("Assessing reduction quality...", err=True)
    try:
        from egf.quality import assess_quality
        report = assess_quality(
            embeddings,
            reduction.reduced_2d,
            quality_threshold=quality_threshold,
            output_path=output / "quality.json",
        )
    except Exception as e:
        click.echo(f"Error during quality assessment: {e}", err=True)
        sys.exit(1)

    if report.warning:
        click.echo(f"⚠  {report.warning_message}", err=True)

    # Density estimation
    click.echo("Estimating density...", err=True)
    try:
        from egf.density import estimate_density
        reduced_2d = np.load(output / "reduced_2d.npy")
        density_result = estimate_density(reduced_2d, k=density_k)
    except Exception as e:
        click.echo(f"Error during density estimation: {e}", err=True)
        sys.exit(1)

    # Gap detection
    click.echo("Detecting gap regions...", err=True)
    try:
        from egf.gaps import detect_gaps
        gaps = detect_gaps(
            density_result,
            reduced_2d=reduced_2d,
            item_names=[doc.name for doc in documents],
            isolation_min=isolation_min,
            max_gaps=max_gaps,
            output_path=output / "gaps.json",
        )
    except Exception as e:
        click.echo(f"Error during gap detection: {e}", err=True)
        sys.exit(1)

    if gaps:
        click.echo(f"Found {len(gaps)} gap region(s).", err=True)
    else:
        click.echo(
            "No gap regions found. Try --isolation-min with a lower value, "
            "or add more descriptions to the corpus.",
            err=True,
        )

    # Candidate generation
    click.echo("Generating candidates...", err=True)
    try:
        import json as _json

        from egf.candidates import generate_candidates
        from egf.quality import QualityReport

        quality_data = _json.loads((output / "quality.json").read_text())
        q_report = QualityReport(**quality_data)

        candidates = generate_candidates(
            gaps=gaps,
            documents=documents,
            reduced_2d=reduced_2d,
            quality_report=q_report,
            domain=domain,
            use_llm=use_llm,
            llm_host=llm_host,
            llm_model=llm_model,
            output_path=output / "candidates.json",
        )
    except Exception as e:
        click.echo(f"Error during candidate generation: {e}", err=True)
        sys.exit(1)

    click.echo(
        f"Generated {len(candidates)} candidate(s).",
        err=True,
    )
    click.echo(
        "Phase 4 complete. HTML report coming in Phase 5.",
        err=True,
    )
