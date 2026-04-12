"""HTML report generation for EGF pipeline output."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from egf import __version__

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from egf.candidates import Candidate
    from egf.domain import DomainTemplate
    from egf.gaps import GapRegion
    from egf.quality import QualityReport


@dataclass
class ReportContext:
    # Header
    version: str
    timestamp: str              # ISO 8601, UTC
    domain_label: str           # domain.label_noun, capitalised
    domain_label_plural: str    # domain.label_plural
    corpus_size: int
    model_name: str
    input_dir: str              # display name only, not full path

    # Quality
    trustworthiness: float
    continuity: float
    lcmc: float
    quality_warning: bool
    quality_warning_message: str | None
    saturation_warning: bool
    saturation_warning_message: str | None

    # Map data (JSON strings for embedding in template)
    scatter_json: str           # plotly figure JSON
    density_json: str           # plotly contour trace JSON

    # Gaps
    gaps: list[dict[str, object]]            # serialised GapRegion dicts

    # Candidates
    candidates: list[dict[str, object]]      # serialised Candidate dicts

    # Flags
    has_gaps: bool
    has_candidates: bool
    has_quality_warning: bool
    has_saturation_warning: bool


def _build_plotly_data(
    reduced_2d: np.ndarray,
    doc_names: list[str],
    gaps: list[GapRegion],
) -> tuple[str, str]:
    """
    Build plotly trace data as JSON strings.

    Returns:
        scatter_json: main corpus scatter trace
        density_json: gap annotation scatter trace (circles)
    """
    # Strip extensions for display
    display_names = [n.rsplit(".", 1)[0] for n in doc_names]

    scatter_trace = {
        "type": "scatter",
        "mode": "markers+text",
        "x": reduced_2d[:, 0].tolist(),
        "y": reduced_2d[:, 1].tolist(),
        "text": display_names,
        "textposition": "top center",
        "textfont": {"size": 10, "color": "#a8b2c0"},
        "marker": {
            "size": 8,
            "color": "#4fc3f7",
            "opacity": 0.85,
            "line": {"width": 1, "color": "#1a2332"},
        },
        "name": "Corpus",
        "hovertemplate": "<b>%{text}</b><extra></extra>",
    }

    gap_traces = []
    for g in gaps:
        cx, cy = g.centroid_2d
        r = g.radius
        gap_traces.append({
            "type": "scatter",
            "mode": "markers+text",
            "x": [cx],
            "y": [cy],
            "text": [f"Gap {g.gap_id}"],
            "textposition": "bottom center",
            "textfont": {"size": 9, "color": "#ff8a65"},
            "marker": {
                "size": max(12, int(r * 60)),
                "color": "rgba(255, 138, 101, 0.12)",
                "line": {"width": 1.5, "color": "#ff8a65"},
                "symbol": "circle",
            },
            "name": f"Gap {g.gap_id}",
            "hovertemplate": (
                f"<b>Gap {g.gap_id}</b><br>"
                f"Isolation: {g.isolation_score:.3f}<br>"
                "Nearest: "
                + ", ".join(
                    n.rsplit(".", 1)[0] for n in g.nearest_items[:2]
                )
                + "<extra></extra>"
            ),
        })

    return json.dumps(scatter_trace), json.dumps(gap_traces)


def build_report_context(
    documents_names: list[str],
    reduced_2d_path: Path,
    quality_report: QualityReport,
    gaps: list[GapRegion],
    candidates: list[Candidate],
    domain: DomainTemplate,
    model_name: str,
    input_dir: Path,
    density_grid_path: Path | None = None,
) -> ReportContext:
    """
    Assemble all pipeline data into a ReportContext for template rendering.
    """
    import dataclasses

    import numpy as np
    from sklearn.metrics import pairwise_distances

    reduced_2d = np.load(reduced_2d_path)

    # Saturation check: low variance in density indicates a saturated space
    # Proxy: std of pairwise distances among corpus points
    if len(reduced_2d) >= 2:
        dists = pairwise_distances(reduced_2d)
        np.fill_diagonal(dists, np.nan)
        dist_std = float(np.nanstd(dists))
        saturation_warning = dist_std < 0.15 and len(gaps) == 0
    else:
        dist_std = 0.0
        saturation_warning = False

    saturation_msg = (
        "The corpus points are tightly clustered (distance std = "
        f"{dist_std:.3f}). The space may be saturated — existing concepts "
        "cover it densely. Gap candidates, if any, should be interpreted "
        "with caution."
    ) if saturation_warning else None

    # Build plotly scatter + gap annotation traces
    scatter_json, density_json = _build_plotly_data(
        reduced_2d=reduced_2d,
        doc_names=documents_names,
        gaps=gaps,
    )

    # Serialise gaps and candidates
    gaps_dicts = [dataclasses.asdict(g) for g in gaps]
    candidates_dicts = [dataclasses.asdict(c) for c in candidates]

    # Strip extensions from nearest_items in gaps for display
    for g in gaps_dicts:
        g["nearest_items_display"] = [
            n.rsplit(".", 1)[0] for n in g["nearest_items"]
        ]

    # Strip extensions from bounding_items in candidates for display
    for c in candidates_dicts:
        c["bounding_items_display"] = [
            n.rsplit(".", 1)[0] for n in c["bounding_items"]
        ]
        # Generation mode badge label
        c["mode_badge"] = {
            "vocabulary": "vocab",
            "llm": "llm",
            "llm_fallback": "llm\u2192vocab",
        }.get(c["generation_mode"], c["generation_mode"])

    return ReportContext(
        version=__version__,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        domain_label=domain.label_noun.title(),
        domain_label_plural=domain.label_plural.title(),
        corpus_size=len(documents_names),
        model_name=model_name,
        input_dir=input_dir.name,
        trustworthiness=round(quality_report.trustworthiness, 4),
        continuity=round(quality_report.continuity, 4),
        lcmc=round(quality_report.lcmc, 4),
        quality_warning=quality_report.warning,
        quality_warning_message=quality_report.warning_message,
        saturation_warning=saturation_warning,
        saturation_warning_message=saturation_msg,
        scatter_json=scatter_json,
        density_json=density_json,
        gaps=gaps_dicts,
        candidates=candidates_dicts,
        has_gaps=len(gaps) > 0,
        has_candidates=len(candidates) > 0,
        has_quality_warning=quality_report.warning,
        has_saturation_warning=saturation_warning,
    )


def render_report(
    context: ReportContext,
    output_path: Path,
) -> None:
    """
    Render the Jinja2 template with context and write to output_path.

    Raises:
        RuntimeError: if template is not found
    """
    import dataclasses
    from pathlib import Path as _Path

    from jinja2 import Environment, FileSystemLoader, select_autoescape

    templates_dir = _Path(__file__).parent / "templates"
    if not templates_dir.exists():
        raise RuntimeError(f"Templates directory not found: {templates_dir}")

    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    template = env.get_template("report.html.j2")
    html = template.render(ctx=dataclasses.asdict(context))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"Report written to {output_path.name}", file=sys.stderr)
