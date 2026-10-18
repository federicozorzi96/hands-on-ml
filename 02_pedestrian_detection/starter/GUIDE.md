# Do it yourself first

Read the exercise README and `CONCEPTS.md`, copy the Python files from this folder to `my_solution/`, and work there. All data/evaluation/deployment plumbing is supplied. Only `build_model` and `train_step` in `model.py` are unfinished.

## 1. Understand the detector

Draw: image → CNN backbone → feature pyramid → region proposals → classification and box refinement. What parameters belong to the backbone versus the detection heads? Why does the classifier need a background class?

Unlike a fixed-size classification batch, the detector receives a **list** of `(3, height, width)` tensors in `[0, 1]`. Each training target has `boxes` of shape `(N, 4)` in pixel coordinates `(xmin, ymin, xmax, ymax)` and `labels` of shape `(N,)` with integer class IDs. Images can have different numbers of pedestrians.

## 2. Implement `build_model`

- Load the MobileNetV3 Faster R-CNN builder with the supplied pretrained weights enum when `pretrained=True`; explicitly set `weights_backbone=None` for offline reconstruction.
- Use `min_size=image_size`, `max_size=image_size * 2` and zero or one trainable backbone stages depending on `mode`.
- Call the supplied `_frozen_batch_norm` helper so saved and reconstructed model architectures agree.
- For the exercise, replace `model.roi_heads.box_predictor` with `FastRCNNPredictor(in_features, 2)`. Read `in_features` from the old classifier.
- For the original pretrained comparison (`num_classes=91`), keep the original head.
- In frozen mode, set `requires_grad_(False)` on every backbone parameter. The proposal and detection heads still train.

Print the total/trainable parameter counts and inspect one forward pass before launching training. The builder is responsible for constructing the exact same architecture when a checkpoint is loaded without network access.

## 3. Implement `train_step`

In training mode, `model(images, targets)` returns a dictionary of differentiable losses. Sum them. Check that the total is finite, clear previous gradients, backpropagate and update the optimizer. Return a detached Python loss scalar for logging.

Questions: what happens if you omit `zero_grad`? Why must logging detach the loss? Why does inference use both `model.eval()` and `torch.inference_mode()`?

## 4. Run one controlled comparison

Use the same split and seed for frozen versus partial fine-tuning. Keep test data untouched until you choose the checkpoint. Interpret the plots and precision/recall changes; beating the original pretrained model is not a requirement.

## 5. Deploy what you trained

The provided app loads your checkpoint at startup. Follow the exercise README to run it locally, create the Docker bundle and publish a Docker Space. The final check compares the model checkpoint hash and predictions locally versus remotely. Do not replace `my_solution/model.py` with the reference implementation just to make the deployment work: complete your implementation and fix any mismatch.

If stuck, compare one function with `reference_solution/model.py`, then explain the difference in your notes.
