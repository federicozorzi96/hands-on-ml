# Concepts to connect to the code

## From your NumPy network to PyTorch

The mathematics is the same: forward pass, scalar loss, chain rule, parameter update. In PyTorch, a `Tensor` can record the operations used to compute it. Calling `loss.backward()` traverses that computation graph and accumulates derivatives in each trainable parameter's `.grad`.

An `nn.Module` groups computation and registered parameters. The optimizer holds references to those parameters; `optimizer.step()` updates them using the accumulated gradients. Clear gradients before each independent update, because backward accumulates rather than replaces them. A loss dictionary must be summed into a scalar before backpropagation.

`model.train()` and `model.eval()` switch layer behavior, such as dropout or batch normalization. They do not enable or disable gradient recording. `torch.inference_mode()` disables the training bookkeeping during prediction; inference normally uses it together with `eval()`.

## What the CNN learns

A convolution applies small learned filters repeatedly across an image. Shared weights make the same feature detectable at different locations. Early features can describe local edges; deeper layers combine them into more complex patterns. Strides and downsampling trade spatial detail for a larger receptive field. This matters when pedestrians are small or partly hidden.

The **backbone** converts pixels into feature maps. A feature pyramid exposes information at several scales. Faster R-CNN's region proposal network suggests candidate object regions; the ROI heads then classify regions and refine their boxes. This exercise uses a MobileNetV3 backbone to reduce the initial compute budget. You reuse the architecture rather than implement every convolution or proposal operation yourself.

Before coding, find the backbone and prediction head in the printed module tree. Which parameter shapes change when the number of classes changes?

## Detection is more than classification

One image can contain zero, one or several pedestrians. A target therefore contains a variable-length list of boxes, not one class per image. A prediction includes boxes, class IDs and scores. Boxes are expressed in original-image pixels; the model handles its internal resizing.

The training losses include objectness, proposal-box regression, ROI classification and final box regression. A lower total loss is useful for debugging but is not the same as better held-out detection quality. Inspect the individual loss terms from a forward call and explain their purposes.

**IoU** is intersection area divided by union area for two boxes. In the supplied evaluator, a prediction is correct when it matches an as-yet-unmatched ground-truth pedestrian with IoU at least 0.5. Duplicate predictions count as false positives. Changing the confidence threshold changes which predictions are retained, which can change precision and recall without changing the trained model.

## What you actually train

Pretraining supplies feature extractors learned on another dataset. Replacing the classifier/regressor head adapts the output space to background plus pedestrian. In `frozen` mode, the CNN backbone is fixed while the proposal and ROI heads train. In `finetune` mode, a final backbone stage can adapt too.

More trainable parameters can improve adaptation but also increase compute and overfitting. Compare both modes using the same split, initialization seed, epochs and learning rate. Choosing the frozen model is a valid outcome. The supplied frozen normalization helper keeps the architecture consistent when reconstructing a saved checkpoint without downloading pretrained weights.

## What deployment adds

A checkpoint contains model parameters and configuration; an API translates uploaded bytes into a tensor, runs the reconstructed model and serializes predictions. Saving the same weights is not sufficient if preprocessing or architecture changes. The endpoint check compares the checkpoint hash and numerical predictions using the same image.

Local inference time, server inference time and network round-trip time answer different questions. An online URL demonstrates serving; it does not by itself demonstrate edge-device optimization or a complete autonomous-driving system. ONNX export and quantization are possible later exercises.
