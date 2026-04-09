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
def main(
    input_dir: Path,
    output: Path,
    model: str,
    domain_str: str,
    describe_format: bool,
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

    click.echo("Phase 1 complete. Reduction and gap detection coming in Phase 2.",
               err=True)
