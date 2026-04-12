Latent Language Explorer (LLE) is a browser-based interactive visualisation
tool that renders a three-dimensional semantic terrain from a high-dimensional
word embedding model, allowing a user to navigate the topology of language as
a spatial landscape.

It takes a pre-trained GloVe 300-dimensional word embedding model as input,
reduces it to three dimensions using UMAP, and organises the resulting space
into a structured domain taxonomy to provide orientation.

It produces an interactive 3D scatter map in the browser, with a FastAPI
backend serving embedding lookups and an Anthropic API integration that
generates natural-language descriptions of regions the user explores.

LLE does not evaluate the quality of its own dimensionality reduction, does
not track how the embedding space changes over time, and does not identify
regions of the space that are conceptually absent or underrepresented.
