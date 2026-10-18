"""Penn-Fudan data plumbing, intentionally supplied rather than left as a TODO."""
from pathlib import Path
import argparse
import hashlib
import random
import urllib.request
import zipfile

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor

URL = 'https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip'


def download(destination):
    destination = Path(destination).resolve()
    if (destination / 'PNGImages').is_dir() and (destination / 'PedMasks').is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / 'PennFudanPed.zip'
    urllib.request.urlretrieve(URL, archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            path = (destination.parent / member.filename).resolve()
            if not path.is_relative_to(destination.parent):
                raise ValueError('Archive contains a path outside the destination.')
        z.extractall(destination.parent)
    if not (destination / 'PNGImages').is_dir():
        raise FileNotFoundError(f'Expected extracted data at {destination}')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_names(root, seed=42):
    names = sorted(p.name for p in (Path(root) / 'PNGImages').glob('*.png'))
    if len(names) < 10:
        raise ValueError('Need at least 10 images. Run python my_solution/data.py --download.')
    order = np.random.default_rng(seed).permutation(len(names))
    n_val = n_test = max(1, len(names) // 10)
    groups = {'train': order[:-(n_val + n_test)],
              'val': order[-(n_val + n_test):-n_test], 'test': order[-n_test:]}
    return {key: [names[int(i)] for i in indices] for key, indices in groups.items()}


def fingerprint(splits):
    import json
    return hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest()


class PennFudan(Dataset):
    def __init__(self, root, names, augment=False):
        self.root, self.names, self.augment = Path(root), list(names), augment

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        with Image.open(self.root / 'PNGImages' / name) as raw:
            image = pil_to_tensor(raw.convert('RGB')).float() / 255
        with Image.open(self.root / 'PedMasks' / (Path(name).stem + '_mask.png')) as raw:
            mask = np.array(raw)
        if mask.ndim != 2:
            raise ValueError('Expected a single-channel instance mask.')
        boxes = []
        for object_id in np.unique(mask):
            if object_id == 0:
                continue
            ys, xs = np.where(mask == object_id)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        if self.augment and torch.rand(()) < .5:
            image = image.flip(-1)
            width = image.shape[-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
        target = {'boxes': boxes, 'labels': torch.ones(len(boxes), dtype=torch.int64),
                  'image_id': torch.tensor(index),
                  'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
                  'iscrowd': torch.zeros(len(boxes), dtype=torch.int64)}
        return image, target


def collate(batch):
    return tuple(zip(*batch))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/PennFudanPed'))
    p.add_argument('--download', action='store_true')
    args = p.parse_args()
    if args.download:
        download(args.data)
    splits = split_names(args.data)
    print({key: len(value) for key, value in splits.items()})
    image, target = PennFudan(args.data, splits['train'])[0]
    print('Image:', tuple(image.shape), 'boxes:', tuple(target['boxes'].shape))
