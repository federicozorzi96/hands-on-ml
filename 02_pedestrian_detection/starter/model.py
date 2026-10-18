"""Small Faster R-CNN with a CNN backbone; all inference is checkpoint-backed."""
from pathlib import Path

import torch
from torch import nn
from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn,
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d

ARCHITECTURE = 'fasterrcnn_mobilenet_v3_large_320_fpn'


def _frozen_batch_norm(module):
    """Use the same norm architecture with or without downloaded initialization."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            replacement = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                getattr(replacement, attr).copy_(getattr(child, attr).detach())
            setattr(module, name, replacement)
        else:
            _frozen_batch_norm(child)


def build_model(pretrained=True, mode='frozen', image_size=320, num_classes=2):
    # TODO: follow starter/GUIDE.md and implement this function.
    raise NotImplementedError("Complete build_model in my_solution/model.py")


def train_step(model, images, targets, optimizer):
    # TODO: follow starter/GUIDE.md and implement this function.
    raise NotImplementedError("Complete train_step in my_solution/model.py")


def save_checkpoint(path, model, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'architecture': ARCHITECTURE,
                'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                'config': config}, path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format or architecture.')
    config = checkpoint['config']
    # No pretrained download or substitution when serving the learner's model.
    model = build_model(False, config['mode'], config['image_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model, config
