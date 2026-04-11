Embedding Drift Monitor (EDM) is a Python CLI tool that measures how a
text embedding space has changed between two points in time, using a
six-metric battery to quantify the stability and structural shift of the
semantic landscape.

It takes two sets of embeddings — a baseline and a current snapshot — of
the same or overlapping vocabulary, produced by any sentence-transformer
or word-embedding model.

It produces a structured report of neighbourhood stability, rank correlation,
distance distribution shift, cluster membership stability, global geometry
preservation, and hubness shift, with per-metric interpretations and an
overall drift verdict.

EDM does not generate embeddings, does not visualise the embedding space
interactively, and does not identify which regions of the space are absent
or conceptually underrepresented — it only measures change between two
known states.
