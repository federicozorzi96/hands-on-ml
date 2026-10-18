# Neural network from scratch with NumPy

Status: completed by Federico; notes can be added later.

A 2 → 4 → 1 network learns XOR using sigmoid activations, binary cross-entropy and manually implemented backpropagation.

## Files

- `starter/exercise.py`: the guided part of the original exercise, converted to percent-format Python; TODOs are intentionally unfinished.
- `my_solution/nn.py`: code cells from Federico's latest uploaded notebook, before the embedded reference solution. Code, comments and gradient return order are preserved as submitted; notebook outputs and metadata are omitted.
- `reference_solution/train.py`: a separate runnable implementation, a finite-difference gradient check and an experiment comparing descent with the wrong update sign.

The imported original cell text is retained for provenance; this guide and the new reference implementation are in English. The reference code included at the bottom of the original notebooks is deliberately excluded from `starter/` and `my_solution/`.

## Run

From the repository root:

```bash
python -m pip install -r 01_numpy_nn/requirements.txt
python 01_numpy_nn/my_solution/nn.py
python 01_numpy_nn/reference_solution/train.py
```

The reference saves a loss plot and a small results JSON under `01_numpy_nn/outputs/` (ignored by Git). Expected XOR classes: `[0, 1, 1, 0]`.

## Observe

Keep the initial parameters fixed. Compare correct descent with reversed updates, then vary the learning rate. Explain the loss curves before looking at accuracy. The finite-difference check distinguishes an incorrect derivative from a poor optimization setting.

## Provenance

- Starter: `rete_neurale_numpy_guidata.ipynb`, originally created on 1 October 2026.
- Learner implementation: `rete_neurale_numpy_guidata(2).ipynb`, uploaded on 15 October 2026.
- Only Python source is committed, following the repository's notebook convention.
