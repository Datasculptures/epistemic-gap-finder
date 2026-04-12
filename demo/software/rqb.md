Reduction Quality Bench (RQB) is a Python CLI tool that evaluates the
fidelity of a dimensionality reduction by running a multi-metric battery
against the original high-dimensional data and its reduced representation,
returning an opinionated quality verdict.

It takes a high-dimensional numpy array and its reduced counterpart as input,
accepting any reduction produced by any method — UMAP, t-SNE, PCA, or other
techniques.

It produces a structured JSON report of trustworthiness, continuity, LCMC,
and related neighbourhood-preservation scores, along with a ranked quality
verdict and human-readable interpretation.

RQB does not perform dimensionality reduction itself, does not visualise the
reduced space, and does not track how reduction quality changes across
different versions of the same dataset over time.
