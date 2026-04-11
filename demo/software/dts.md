Dataset Topology Sculptor (DTS) is a Python CLI tool that converts a
dataset into a physically constructable three-dimensional sculpture design,
encoding the topological structure of the data as a network of nodes and
edges constrained to fit within a one-cubic-metre build volume.

It takes tabular datasets, graph structures, or network capture files (PCAP)
as input, builds a nearest-neighbour graph with MST-protected edge pruning,
and applies a grammar-driven spatial layout to position nodes in 3D space.

It produces a build specification — node positions, edge definitions, material
assignments, and colour coding — in formats suitable for physical construction
using wire, rod, and connector components, along with a rendered preview.

DTS does not embed or semantically analyse text, does not evaluate the quality
of its spatial layouts against information-theoretic criteria, and does not
generate candidate structures for regions of the design space that are absent
from the input data.
