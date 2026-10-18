"""Compare a deployed endpoint against the same local checkpoint and image."""
from pathlib import Path
import argparse
import json
import time
from urllib.parse import urlsplit

import httpx
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor

from model import load_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--out', type=Path, default=Path('runs/endpoint_check.json'))
    args = p.parse_args()
    torch.set_num_threads(2)
    model, _ = load_checkpoint(args.checkpoint)
    with Image.open(args.image) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    tensor = pil_to_tensor(image).float() / 255
    with torch.inference_mode():
        model([tensor])  # warm-up
        start = time.perf_counter()
        prediction = model([tensor])[0]
        local_ms = (time.perf_counter() - start) * 1000
    keep = (prediction['labels'] == 1) & (prediction['scores'] >= args.threshold)
    boxes, scores = prediction['boxes'][keep].tolist(), prediction['scores'][keep].tolist()
    # Local inference should not be sent through a machine's external proxy.
    use_environment = urlsplit(args.url).hostname not in {'localhost', '127.0.0.1', '::1'}
    with httpx.Client(timeout=180., trust_env=use_environment) as client:
        with args.image.open('rb') as file:
            start = time.perf_counter()
            response = client.post(args.url.rstrip('/') + '/predict',
                                   params={'threshold': args.threshold},
                                   files={'file': (args.image.name, file, 'application/octet-stream')})
            remote_roundtrip_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        remote = response.json()
        actual_boxes = [d['box_xyxy'] for d in remote['detections']]
        actual_scores = [d['score'] for d in remote['detections']]
        # Sorting/matching is deliberately strict: differing detections need investigation.
        np.testing.assert_allclose(np.asarray(actual_boxes).reshape(-1, 4),
                                   np.asarray(boxes).reshape(-1, 4), atol=.1, rtol=1e-4)
        np.testing.assert_allclose(actual_scores, scores, atol=1e-4, rtol=1e-4)
        import hashlib
        expected_version = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()[:12]
        if remote['model_version'] != expected_version:
            raise AssertionError('The endpoint is serving a different checkpoint.')
        bad = client.post(args.url.rstrip('/') + '/predict', files={'file': ('bad.txt', b'not an image')})
        if bad.status_code != 422:
            raise AssertionError(f'Expected invalid-image status 422, received {bad.status_code}')
    result = {'prediction_parity': True, 'invalid_image_status': bad.status_code,
              'local_inference_ms': local_ms, 'remote_inference_ms': remote['inference_ms'],
              'remote_roundtrip_ms': remote_roundtrip_ms, 'model_version': remote['model_version']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
