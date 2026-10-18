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
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT if pretrained else None
    model = fasterrcnn_mobilenet_v3_large_320_fpn(
        weights=weights, weights_backbone=None,
        trainable_backbone_layers=0 if mode == 'frozen' else 1,
        min_size=image_size, max_size=image_size * 2,
    )
    _frozen_batch_norm(model)
    if num_classes == 2:
        # One foreground class (pedestrian=1) and background=0.
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, num_classes)
    elif num_classes != 91:
        raise ValueError('Only the two-class exercise or original COCO baseline is supported.')
    if mode == 'frozen':
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    return model


def train_step(model, images, targets, optimizer):
    losses = model(images, targets)
    total = sum(losses.values())
    if not torch.isfinite(total):
        raise FloatingPointError(f'Non-finite loss: {losses}')
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    return float(total.detach().cpu())


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
