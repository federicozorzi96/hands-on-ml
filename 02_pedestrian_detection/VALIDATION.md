# Validation before upload

Executed on 18 October 2026 with Python 3.12, PyTorch 2.8.0+cpu, Torchvision 0.23.0+cpu and FastAPI 0.146.0.

- All supplied Python files parse, including intentionally unfinished starters.
- The four download-free integration checks pass: disjoint image splits and mask-to-box conversion; a real detector optimizer update and checkpoint round-trip; duplicate-box matching; API prediction parity and invalid/oversized inputs.
- The Penn-Fudan downloader fetched the real dataset: 136 training, 17 validation and 17 test images with seed 42. A real image/mask loaded successfully.
- Both `frozen` and `finetune` training paths ran for one epoch using four real training images and two validation images. This checks the pipeline, not the learning outcome of the full exercise.
- The frozen checkpoint was reconstructed without pretrained downloads and evaluated on all 17 validation images. The original pretrained baseline was also evaluated. The test split was not used.
- A separately running local Uvicorn server served the saved checkpoint. `check_endpoint.py` passed numerical prediction parity, checkpoint-hash equality and invalid-image handling over HTTP. Server inference and network round-trip timings were recorded separately.
- `prepare_deploy.py` generated the Docker/Space bundle containing the selected checkpoint and inference implementation.

The Docker image itself was not built because this environment has no Docker engine. A public Space has not been created: publishing it and testing from another device are the deployment exercise. Small-run metrics are deliberately not advertised as detector performance results.
