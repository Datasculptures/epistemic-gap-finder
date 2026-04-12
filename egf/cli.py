"""EGF command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from egf.domain import parse_domain
from egf.embedder import embed_corpus
from egf.loader import load_corpus


@click.group()
def main() -> None:
    """Epistemic Gap Finder — conceptual cartography for any knowledge domain."""


@main.command("analyse")
@click.argument("input_dir", type=click.Path(exists=True, file_okay=False,
                path_type=Path))
@click.option("--output", "-o", type=click.Path(path_type=Path),
              default=Path("egf_output"), show_default=True,
              help="Output directory.")
@click.option("--model", default="all-MiniLM-L6-v2", show_default=True,
              help="Sentence-transformer model name.")
@click.option("--domain", "domain_str", default="concept",
              show_default=True,
              help=(
                  "Domain template. Built-in: concept, software-tool, philosophy, "
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
@click.option("--open", "open_browser", is_flag=True, default=False,
              help="Open the HTML report in the default browser after generation.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show detailed progress for each pipeline step.")
def analyse(
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
    open_browser: bool,
    verbose: bool,
) -> None:
    """Map the conceptual space and find what's absent."""

    def progress(msg: str) -> None:
        if verbose:
            click.echo(msg, err=True)

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
    progress(f"Loading corpus from: {input_dir}")
    try:
        documents = load_corpus(input_dir)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error loading corpus:\n{e}", err=True)
        sys.exit(1)

    click.echo(
        f"Loaded {len(documents)} document(s) from {input_dir.name}.", err=True
    )

    # Prepare output directory
    output.mkdir(parents=True, exist_ok=True)

    # Embed
    progress(f"Embedding {len(documents)} documents with model: {model}")
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
    progress("Reducing embeddings with UMAP...")
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
    progress("Assessing reduction quality...")
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

    progress(f"Quality: trustworthiness={report.trustworthiness:.4f}")

    # Density estimation
    progress("Estimating density...")
    try:
        from egf.density import estimate_density
        reduced_2d = np.load(output / "reduced_2d.npy")
        density_result = estimate_density(reduced_2d, k=density_k)
    except Exception as e:
        click.echo(f"Error during density estimation: {e}", err=True)
        sys.exit(1)

    # Gap detection
    progress("Detecting gap regions...")
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
    progress("Generating candidates...")
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

    click.echo(f"Generated {len(candidates)} candidate(s).", err=True)

    # Render report
    progress("Rendering HTML report...")
    try:
        from datetime import datetime

        from egf.report import build_report_context, render_report

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output / f"report_{timestamp}.html"

        context = build_report_context(
            documents_names=[doc.name for doc in documents],
            reduced_2d_path=output / "reduced_2d.npy",
            quality_report=q_report,
            gaps=gaps,
            candidates=candidates,
            domain=domain,
            model_name=model,
            input_dir=input_dir,
        )
        render_report(context, report_path)
    except Exception as e:
        click.echo(f"Error rendering report: {e}", err=True)
        sys.exit(1)

    click.echo(f"Report: {report_path.name}", err=True)

    if open_browser:
        import webbrowser
        webbrowser.open(report_path.resolve().as_uri())

    click.echo("Done.", err=True)
