# Hands-on ML

Practical machine learning exercises using real or simulated data, from understanding methods to deploying models.

This is very much a work in progress. It's on hold for a couple of months while I focus on other projects.

## Study plan

The first two exercises are a [neural network from scratch in NumPy](01_numpy_nn/) and [pedestrian detection with PyTorch](02_pedestrian_detection/).

When I pick this back up, I plan to:

1. **Finish the PyTorch detector.** Compare training with a frozen backbone and partial fine-tuning, evaluate both runs, and review detection errors. Then deploy the trained model as an API and test it.
2. **Build an AI agent.** Implement and test a small agent.
3. **Explore causal machine learning.** Work through Double Machine Learning for estimating treatment effects.
4. **Study power analysis and sample size.** Use a simulated A/B test to see how sample size affects statistical power.

## Structure

Each exercise has its own folder containing:

- `starter/` — original instructions, starter code and supporting files.
- `my_solution/` — my implementation and experiments.
- `reference_solution/` — a worked solution for comparison.

Each exercise includes setup and execution instructions. Large datasets and model checkpoints are kept out of Git.

Jupytext notebooks are edited as percent-format `.py` files; install `requirements-dev.txt` and use `make notebook-to-py NOTEBOOK=path/to/file.ipynb` or `make py-to-notebook NOTEBOOK=path/to/file.py` to convert them.
