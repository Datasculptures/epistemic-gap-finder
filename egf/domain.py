"""Domain template system — built-in registry and parse_domain()."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainTemplate:
    name: str                    # internal identifier, e.g. "software-tool"
    label_noun: str              # singular, e.g. "software tool"
    label_plural: str            # plural, e.g. "software tools"
    system_prompt_fragment: str  # injected into LLM system prompt (Phase 4)
    describe_format_text: str    # printed by --describe-format


# ── Built-in templates ────────────────────────────────────────────────────────

SOFTWARE_TOOL = DomainTemplate(
    name="software-tool",
    label_noun="software tool",
    label_plural="software tools",
    system_prompt_fragment=(
        "You are an expert software architect with broad knowledge of developer "
        "tooling, CLI utilities, and software ecosystems."
    ),
    describe_format_text="""\
Four-sentence description template for a software tool:

  Sentence 1 — What it does:
    Describe the tool's primary function in one sentence.
    Example: "RQB evaluates the quality of dimensionality reduction
    by running a multi-metric battery against the original and
    reduced representations of a dataset."

  Sentence 2 — Input and subject matter:
    What kind of data or artefacts does it operate on?
    Example: "It takes a high-dimensional numpy array and its
    reduced counterpart as input."

  Sentence 3 — Output:
    What does it produce?
    Example: "It produces a structured JSON report of trustworthiness,
    continuity, and LCMC scores, along with a ranked quality verdict."

  Sentence 4 — Boundary condition (most important):
    What does it explicitly NOT do? Where does it stop?
    Example: "It does not perform dimensionality reduction itself —
    it only assesses reductions that have already been produced."

The boundary condition sentence is the most important for positioning
accuracy. Include it even if it feels obvious.""",
)

PHILOSOPHY = DomainTemplate(
    name="philosophy",
    label_noun="philosophical position",
    label_plural="philosophical positions",
    system_prompt_fragment=(
        "You are a philosopher with expertise in the history of ideas, ethical "
        "theory, epistemology, and metaphysics across Western and non-Western "
        "traditions."
    ),
    describe_format_text="""\
Four-sentence description template for a philosophical position:

  Sentence 1 — Core claim:
    What is the central thesis or commitment of this position?
    Example: "Stoicism holds that virtue is the only genuine good
    and that external circumstances are indifferent to human flourishing."

  Sentence 2 — Domain and method:
    What questions does it address, and how does it argue?
    Example: "It operates in ethics and psychology, arguing through
    reason and the discipline of attention toward what is within
    one's control."

  Sentence 3 — Practical or theoretical output:
    What does this position produce — a way of life, a theory, a critique?
    Example: "It produces a practice of equanimity and a framework
    for distinguishing preferred from required outcomes."

  Sentence 4 — Boundary condition (most important):
    What does this position explicitly reject, deny, or leave outside
    its scope?
    Example: "It does not deny that suffering occurs — it denies
    that suffering caused by external events is rationally necessary."

The boundary condition sentence is the most important for positioning
accuracy. Include it even if it feels obvious.""",
)

VEHICLE = DomainTemplate(
    name="vehicle",
    label_noun="vehicle type",
    label_plural="vehicle types",
    system_prompt_fragment=(
        "You are an automotive and transport historian with broad knowledge of "
        "vehicle design, classification systems, and engineering trade-offs."
    ),
    describe_format_text="""\
Four-sentence description template for a vehicle type:

  Sentence 1 — Primary function:
    What is this vehicle type designed to do?

  Sentence 2 — Configuration and inputs:
    What distinguishes its physical form, drivetrain, or passenger/cargo
    arrangement?

  Sentence 3 — Use case and output:
    What market or operational role does it serve?

  Sentence 4 — Boundary condition (most important):
    What does it explicitly not do, carry, seat, or serve?
    This is the most important sentence for positioning accuracy.""",
)

GENRE = DomainTemplate(
    name="genre",
    label_noun="genre",
    label_plural="genres",
    system_prompt_fragment=(
        "You are a cultural historian and musicologist with broad knowledge of "
        "artistic and musical traditions, their formal conventions, and their "
        "social contexts."
    ),
    describe_format_text="""\
Four-sentence description template for a genre:

  Sentence 1 — Defining characteristics:
    What formal or aesthetic properties define this genre?

  Sentence 2 — Influences and inputs:
    What traditions, instruments, or cultural contexts does it draw from?

  Sentence 3 — Audience and output:
    What experience or artefact does it produce for its audience?

  Sentence 4 — Boundary condition (most important):
    What does it explicitly exclude, reject, or refuse to do formally
    or culturally?
    This is the most important sentence for positioning accuracy.""",
)

DISCIPLINE = DomainTemplate(
    name="discipline",
    label_noun="academic discipline",
    label_plural="academic disciplines",
    system_prompt_fragment=(
        "You are an interdisciplinary scholar with broad knowledge of academic "
        "fields, their methodologies, epistemological commitments, and "
        "institutional histories."
    ),
    describe_format_text="""\
Four-sentence description template for an academic discipline:

  Sentence 1 — Object of study:
    What does this discipline study or investigate?

  Sentence 2 — Method:
    What methods, tools, or epistemological commitments define its practice?

  Sentence 3 — Output:
    What kind of knowledge, models, or artefacts does it produce?

  Sentence 4 — Boundary condition (most important):
    What questions, methods, or domains does it explicitly place outside
    its scope?
    This is the most important sentence for positioning accuracy.""",
)

CONCEPT = DomainTemplate(
    name="concept",
    label_noun="concept",
    label_plural="concepts",
    system_prompt_fragment=(
        "You are a knowledgeable expert with broad understanding of ideas, "
        "categories, and conceptual distinctions across many fields."
    ),
    describe_format_text=(
        "Four-sentence description template for a concept:\n\n"
        "  Sentence 1 — What it is:\n"
        "    State the core identity or function of this concept.\n\n"
        "  Sentence 2 — Domain and scope:\n"
        "    What field, subject matter, or context does it operate in?\n\n"
        "  Sentence 3 — What it produces or enables:\n"
        "    What outcome, experience, or result does it generate?\n\n"
        "  Sentence 4 — Boundary condition (most important):\n"
        "    What does it explicitly NOT include, do, or cover?\n"
        "    This is the most important sentence for positioning accuracy."
    ),
)

DOMAIN_REGISTRY: dict[str, DomainTemplate] = {
    "concept": CONCEPT,
    "software-tool": SOFTWARE_TOOL,
    "philosophy": PHILOSOPHY,
    "vehicle": VEHICLE,
    "genre": GENRE,
    "discipline": DISCIPLINE,
}


# ── Pluralisation helper ──────────────────────────────────────────────────────

def _make_singular_plural(noun: str) -> tuple[str, str]:
    """
    Given a noun (which may be singular or already plural), return
    (singular, plural) forms for use in report labels.

    Handles the most common English patterns. Not exhaustive — irregular
    plurals (person/people, child/children) are not handled.
    """
    s = noun.strip()
    lower = s.lower()

    if lower.endswith("ses") or lower.endswith("xes") or lower.endswith("zes"):
        # e.g. "classes" → singular "class", plural "classes"
        return s[:-2], s

    if lower.endswith("ies"):
        # e.g. "categories" → singular "category", plural "categories"
        return s[:-3] + "y", s

    if lower.endswith("s"):
        # Likely already plural — use as-is for plural, strip "s" for singular
        # e.g. "genres" → singular "genre", plural "genres"
        return s[:-1], s

    # Singular noun — pluralise
    if lower.endswith(("x", "z", "ch", "sh")):
        return s, s + "es"
    if lower.endswith("y") and len(s) > 1 and s[-2].lower() not in "aeiou":
        return s, s[:-1] + "ies"
    return s, s + "s"


# ── parse_domain ──────────────────────────────────────────────────────────────

def parse_domain(value: str) -> DomainTemplate:
    """
    Resolve a --domain string to a DomainTemplate.

    Built-in keys: software-tool, philosophy, vehicle, genre, discipline
    Custom: any string starting with "custom:" — remainder becomes the noun.

    Raises:
        ValueError: if value is not a built-in key and does not start with "custom:"
    """
    normalised = value.strip().lower()

    if normalised in DOMAIN_REGISTRY:
        return DOMAIN_REGISTRY[normalised]

    if normalised.startswith("custom:"):
        noun = value[len("custom:"):].strip()
        if not noun:
            raise ValueError(
                "custom: domain requires a noun, e.g. custom:musical instrument"
            )
        label_noun, label_plural = _make_singular_plural(noun)
        return DomainTemplate(
            name=f"custom:{noun}",
            label_noun=label_noun,
            label_plural=label_plural,
            system_prompt_fragment=(
                f"You are a knowledgeable expert on {label_plural} with broad "
                f"understanding of their variations, uses, and distinctions."
            ),
            describe_format_text=(
                f"Four-sentence description template for a {label_noun}:\n\n"
                "  Sentence 1 — What it is or does.\n"
                "  Sentence 2 — What inputs, materials, or subject matter"
                " it involves.\n"
                "  Sentence 3 — What output, effect, or role it produces.\n"
                "  Sentence 4 — Boundary condition (most important): what it "
                "explicitly does NOT do, include, or serve.\n\n"
                "The boundary condition sentence is the most important for "
                "positioning accuracy."
            ),
        )

    raise ValueError(
        f"Unknown domain '{value}'. "
        "Built-in options: concept, software-tool, philosophy, vehicle, "
        "genre, discipline. Custom: custom:<your domain noun>"
    )
