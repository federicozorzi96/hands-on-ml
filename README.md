# Hands-on ML

Practical machine learning exercises using real or simulated data, from understanding methods to deploying models.

## Structure

Each exercise has its own folder containing:

- `starter/` — original instructions, starter code and supporting files.
- `my_solution/` — my implementation and experiments.
- `reference_solution/` — a worked solution for comparison.

Each exercise includes setup and execution instructions. Large datasets and model checkpoints are kept out of Git.

Jupytext notebooks are edited as percent-format `.py` files; install `requirements-dev.txt` and use `make notebook-to-py NOTEBOOK=path/to/file.ipynb` or `make py-to-notebook NOTEBOOK=path/to/file.py` to convert them.
