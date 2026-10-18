# Pedestrian detection: learn PyTorch, train, deploy

One project, three 90-minute sessions. Learn CNN-based object detection and PyTorch training, then put **your own trained checkpoint online**. Input: an image. Output: pedestrian bounding boxes and confidence scores.

## Start here

Use Python 3.11 or 3.12. From the repository root, create an environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
# Linux/Windows CPU, including Codespaces:
make install-cpu
# Apple Silicon alternative: make install EXERCISE=02_pedestrian_detection
cd 02_pedestrian_detection
cp starter/*.py my_solution/
python my_solution/data.py --download
```

For your Intel Mac, use Codespaces or Colab for this pinned environment. For GPU training, select a GPU runtime in Colab and install the matching PyTorch/Torchvision pair using the [official installer](https://pytorch.org/get-started/locally/). The reproducible baseline here is PyTorch 2.8.0 + Torchvision 0.23.0. Do not install CPU wheels into a GPU environment. The code selects CUDA when available. Training duration depends on hardware; use the small-run command below to establish a working loop first.

`starter/` contains original exercise files, including working data/evaluation/API helpers and **two TODO functions in `model.py`**. Copy them into `my_solution/`, implement them there, and read `starter/GUIDE.md`. `reference_solution/` contains the worked implementation; it is never imported automatically by your solution.

## Model and data

- Real Penn-Fudan pedestrian images; instance masks are converted into detection boxes. Data downloads at runtime and stays outside Git.
- Faster R-CNN with a MobileNetV3 CNN backbone and low-resolution inputs, chosen to keep this first experiment smaller than a ResNet-based detector.
- Fine-tuning from pretrained weights. This is training on your dataset, rather than merely calling an existing model or training an entire detector from random initialization.
- One foreground class, pedestrian (`1`); background is `0`.

## Session 1 — PyTorch and first training

Understand tensors, `Dataset`/`DataLoader`, `nn.Module`, autograd, the optimizer and the detector's loss dictionary. Implement model construction and one optimizer step in `my_solution/model.py`.

```bash
# First end-to-end run, then increase the subset/epochs when it works:
python my_solution/train.py --mode frozen --epochs 1 --max-train 24 --max-val 8 --out runs/first
python my_solution/train.py --mode frozen --epochs 3 --out runs/frozen
```

Observe classification versus box-regression losses in the model's returned dictionary; the supplied loop logs their sum. Predict tensor shapes before running the forward pass. Result: a checkpoint, loss history and a reproducible image split.

## Session 2 — Controlled experiment and evaluation

Keep the split, seed, learning rate and number of epochs fixed. Compare training the detection heads with a frozen backbone against partially fine-tuning the last backbone stage:

```bash
python my_solution/train.py --mode finetune --epochs 3 --out runs/finetune
python my_solution/evaluate.py --checkpoint runs/frozen/best.pt --out runs/frozen_val
python my_solution/evaluate.py --checkpoint runs/finetune/best.pt --out runs/finetune_val
python my_solution/evaluate.py --pretrained-baseline --out runs/pretrained_val
```

Use validation to choose the final checkpoint; only then evaluate that checkpoint on test:

```bash
python my_solution/evaluate.py --checkpoint runs/frozen/best.pt --split test --out runs/final_test
```

Replace `runs/frozen/best.pt` with your validation-selected checkpoint. The COCO baseline retains its original head and filters predictions to the person class. The fine-tuned models have a new two-class head, so the comparison includes adaptation of that head.

Metrics: precision, recall and F1 at score threshold 0.5 and box IoU 0.5, plus warmed-up mean inference time. These are **not COCO mAP**. Inspect five plots (green: truth; red: predictions). On validation, vary `--threshold` between 0.2, 0.5 and 0.8 and explain the precision/recall trade-off. A very small dataset may not show a fine-tuning improvement: explain failures rather than optimizing only the headline number.

Splits are deterministic, disjoint **image-level** partitions, with names and a fingerprint stored in each run. They do not establish independence across scenes or video sequences: this dataset/exercise is an exploratory demo, not an autonomous-driving benchmark. Do not tune using the test set. Augmentation is training-only and transforms image and boxes together.

## Session 3 — Deploy and test the trained model

1. Start your saved checkpoint locally (no replacement by pretrained weights):

```bash
MODEL_PATH=runs/frozen/best.pt python -m uvicorn app:app --app-dir my_solution --host 0.0.0.0 --port 7860
```

2. Open `http://localhost:7860` to upload an image, or `/docs` for the API. In a second terminal, still in this exercise folder:

```bash
python my_solution/check_endpoint.py --url http://localhost:7860 --checkpoint runs/frozen/best.pt --image data/PennFudanPed/PNGImages/FudanPed00001.png
```

3. Package your implementation and selected checkpoint:

```bash
python prepare_deploy.py --implementation my_solution --checkpoint runs/frozen/best.pt
# Optional local Docker check; run from this exercise folder:
docker build -t pedestrian-demo deploy_bundle
docker run --rm -p 7860:7860 pedestrian-demo
```

4. Create a **Docker Space** on [Hugging Face Spaces](https://huggingface.co/new-space), using CPU hardware for the first deployment. Upload the **contents** of `deploy_bundle/` at the Space root: `Dockerfile`, `README.md`, `requirements.txt`, `model.py`, `app.py`, and `artifacts/model.pt`. Keep the YAML header in that README: it selects Docker and port 7860. Large model weights belong in the Space's artifact storage, not in this GitHub code repo. Choose the hardware offered by the host; no paid hardware is needed by the exercise design.

5. Once the Space is running, use its direct app URL (for example `https://YOUR-SPACE.hf.space`, not its repository page) in `check_endpoint.py`. Open it from your phone too. Record checkpoint identity, prediction consistency, invalid-image handling and local/server/network latency. The checker uses a small numerical tolerance: investigate differences across devices rather than silently accepting them.

Success: another person can upload an image to the URL and receive predictions from **your trained model**. The code and Docker bundle are prepared here; creating the host account and publishing the service are part of your exercise.

## Record your results

Use `my_solution/NOTES.md`: hypothesis, dataset/split, seed, model settings, metrics, plots, deployment URL and what surprised you. Generated files go under `runs/`; only commit your notes and code.

## Reference and checks

To run the supplied solution, substitute `reference_solution/` for `my_solution/` in the commands. Download-free pipeline checks:

```bash
python reference_solution/verify_pipeline.py
```

They exercise a real detector training step, save/load, dataset boxes, split separation and API behavior using tiny **synthetic test fixtures**. They are infrastructure checks, not evidence of detection quality on real data. See `VALIDATION.md` for what was actually executed before upload.

Sources: [PyTorch detection tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html), [model API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html), [Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/), [Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker).
