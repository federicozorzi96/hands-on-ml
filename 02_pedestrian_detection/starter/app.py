"""Serve the trained checkpoint. Start with MODEL_PATH=... uvicorn app:app."""
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from threading import Lock
import hashlib
import logging
import os
import time

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from torchvision.transforms.functional import pil_to_tensor

from model import load_checkpoint

MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 8_000_000
log = logging.getLogger('uvicorn.error')


def create_app():
    @asynccontextmanager
    async def lifespan(app):
        path = Path(os.environ.get('MODEL_PATH', 'artifacts/model.pt'))
        if not path.is_file():
            raise FileNotFoundError(f'Missing trained checkpoint: {path}. Set MODEL_PATH or prepare the deployment bundle.')
        torch.set_num_threads(int(os.environ.get('TORCH_NUM_THREADS', '2')))
        app.state.model, app.state.config = load_checkpoint(path, 'cpu')
        app.state.version = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        app.state.lock = Lock()
        # Warm up once; startup fails if the checkpoint cannot execute inference.
        with torch.inference_mode():
            app.state.model([torch.zeros(3, 64, 64)])
        log.info('Loaded checkpoint version=%s', app.state.version)
        yield

    app = FastAPI(title='Pedestrian detector', lifespan=lifespan)

    @app.get('/health')
    def health():
        return {'status': 'ready', 'model_version': app.state.version}

    @app.get('/', response_class=HTMLResponse)
    def home():
        return '''<!doctype html><html lang="en"><meta charset="utf-8">
        <title>Pedestrian detector</title><h1>Pedestrian detector</h1>
        <p>Upload an image to receive bounding boxes and confidence scores.</p>
        <form action="/predict" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required><button>Predict</button>
        </form><p><a href="/docs">Interactive API documentation</a></p></html>'''

    @app.post('/predict')
    def predict(file: UploadFile = File(...), threshold: float = Query(.5, ge=0., le=1.)):
        raw = file.file.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, 'Image upload exceeds 8 MiB.')
        try:
            with Image.open(BytesIO(raw)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise HTTPException(413, 'Image exceeds 8 million pixels.')
                source.load()
                image = ImageOps.exif_transpose(source).convert('RGB')
        except Image.DecompressionBombError as exc:
            raise HTTPException(413, 'Image dimensions are too large.') from exc
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(422, 'Upload a valid image.') from exc
        tensor = pil_to_tensor(image).float() / 255
        with app.state.lock, torch.inference_mode():
            start = time.perf_counter()
            result = app.state.model([tensor])[0]
            milliseconds = (time.perf_counter() - start) * 1000
        keep = (result['labels'] == 1) & (result['scores'] >= threshold)
        detections = [{'box_xyxy': box, 'score': score, 'label': 'pedestrian'}
                      for box, score in zip(result['boxes'][keep].tolist(), result['scores'][keep].tolist())]
        log.info('version=%s inference_ms=%.1f detections=%d', app.state.version, milliseconds, len(detections))
        return {'model_version': app.state.version, 'width': image.width, 'height': image.height,
                'inference_ms': milliseconds, 'threshold': threshold, 'detections': detections}

    return app


app = create_app()
