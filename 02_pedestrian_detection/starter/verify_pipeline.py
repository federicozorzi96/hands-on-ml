"""Download-free integration checks; synthetic fixtures are not quality results."""
from io import BytesIO
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from fastapi.testclient import TestClient

from app import create_app
from data import PennFudan, split_names, seed_everything
from evaluate import match_counts
from model import build_model, train_step, save_checkpoint, load_checkpoint


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        seed_everything(42)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        for folder in ['PNGImages', 'PedMasks']:
            (cls.root / folder).mkdir()
        for i in range(12):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            mask = np.zeros((64, 64), dtype=np.uint8)
            rgb[10:45, 20:35] = [190, 100, 50]
            mask[10:45, 20:35] = 1
            Image.fromarray(rgb).save(cls.root / 'PNGImages' / f'fixture{i:02}.png')
            Image.fromarray(mask).save(cls.root / 'PedMasks' / f'fixture{i:02}_mask.png')
        cls.splits = split_names(cls.root)
        cls.dataset = PennFudan(cls.root, cls.splits['train'])
        cls.model = build_model(False, 'frozen', image_size=64)
        cls.model.train()
        before = cls.model.roi_heads.box_predictor.cls_score.weight.detach().clone()
        optimizer = torch.optim.SGD([v for v in cls.model.parameters() if v.requires_grad], lr=.01)
        batch = [cls.dataset[i] for i in range(2)]
        cls.training_loss = train_step(cls.model, [x[0] for x in batch], [x[1] for x in batch], optimizer)
        cls.head_changed = not torch.equal(before, cls.model.roi_heads.box_predictor.cls_score.weight)
        cls.checkpoint = cls.root / 'model.pt'
        save_checkpoint(cls.checkpoint, cls.model, {'mode': 'frozen', 'image_size': 64, 'splits': cls.splits})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_splits_and_mask_boxes(self):
        groups = [set(v) for v in self.splits.values()]
        self.assertEqual(len(set.union(*groups)), 12)
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        _, target = self.dataset[0]
        torch.testing.assert_close(target['boxes'], torch.tensor([[20., 10., 35., 45.]]))

    def test_training_and_checkpoint_roundtrip(self):
        self.assertTrue(np.isfinite(self.training_loss))
        self.assertTrue(self.head_changed)
        loaded, _ = load_checkpoint(self.checkpoint)
        self.model.eval()
        image = self.dataset[0][0]
        with torch.inference_mode():
            before, after = self.model([image])[0], loaded([image])[0]
        torch.testing.assert_close(before['boxes'], after['boxes'])
        torch.testing.assert_close(before['scores'], after['scores'])

    def test_matching_does_not_double_count(self):
        boxes = torch.tensor([[0., 0., 10., 10.], [0., 0., 10., 10.]])
        self.assertEqual(match_counts(boxes, torch.tensor([.9, .8]), boxes[:1]), (1, 1, 0))

    def test_api_parity_and_invalid_inputs(self):
        buffer = BytesIO()
        with Image.open(self.root / 'PNGImages' / self.dataset.names[0]) as image:
            image.save(buffer, format='PNG')
        with patch.dict(os.environ, {'MODEL_PATH': str(self.checkpoint)}):
            with TestClient(create_app()) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                response = client.post('/predict?threshold=0', files={'file': ('image.png', buffer.getvalue(), 'image/png')})
                self.assertEqual(response.status_code, 200)
                loaded, _ = load_checkpoint(self.checkpoint)
                with torch.inference_mode():
                    expected = loaded([self.dataset[0][0]])[0]
                keep = expected['labels'] == 1
                actual = response.json()
                np.testing.assert_allclose([d['box_xyxy'] for d in actual['detections']], expected['boxes'][keep].numpy(), atol=1e-5)
                self.assertEqual(actual['width'], 64)
                self.assertEqual(client.post('/predict', files={'file': ('bad.txt', b'bad')}).status_code, 422)
                self.assertEqual(client.post('/predict?threshold=2', files={'file': ('x.png', buffer.getvalue())}).status_code, 422)
                self.assertEqual(client.post('/predict', files={'file': ('big.png', b'x' * (8 * 1024 * 1024 + 1))}).status_code, 413)


if __name__ == '__main__':
    unittest.main(verbosity=2)
