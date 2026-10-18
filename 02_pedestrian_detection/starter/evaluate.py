"""Fixed-threshold detection metrics and example images; no COCO-mAP claim."""
from pathlib import Path
import argparse
import json
import time

import torch
from torchvision.ops import box_iou
from PIL import ImageDraw
from torchvision.transforms.functional import to_pil_image

from data import PennFudan, split_names
from model import build_model, load_checkpoint


def match_counts(boxes, scores, targets, threshold=.5, iou_threshold=.5):
    boxes = boxes[scores >= threshold]
    scores = scores[scores >= threshold]
    boxes = boxes[scores.argsort(descending=True)]
    matched, tp = set(), 0
    overlaps = box_iou(boxes, targets)
    for row in overlaps:
        candidates = [(float(row[j]), j) for j in range(len(targets)) if j not in matched]
        if candidates:
            overlap, j = max(candidates)
            if overlap >= iou_threshold:
                matched.add(j)
                tp += 1
    return tp, len(boxes) - tp, len(targets) - tp


@torch.inference_mode()
def evaluate_model(model, dataset, device='cpu', threshold=.5, out=None):
    model.eval()
    tp = fp = fn = 0
    times = []
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
    for i in range(len(dataset)):
        image, target = dataset[i]
        image_device = image.to(device)
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = model([image_device])[0]
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
        keep = prediction['labels'].cpu() == 1
        boxes, scores = prediction['boxes'].cpu()[keep], prediction['scores'].cpu()[keep]
        a, b, c = match_counts(boxes, scores, target['boxes'], threshold)
        tp += a; fp += b; fn += c
        if out and i < 5:
            picture = to_pil_image(image)
            draw = ImageDraw.Draw(picture)
            for box in target['boxes'].tolist():
                draw.rectangle(box, outline='green', width=2)
            for box in boxes[scores >= threshold].tolist():
                draw.rectangle(box, outline='red', width=2)
            picture.save(Path(out) / f'example_{i}.png')
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    # Exclude the first image as a warm-up when there are multiple images.
    warm_times = times[1:] or times
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / max(precision + recall, 1e-12),
            'tp': tp, 'fp': fp, 'fn': fn, 'score_threshold': threshold,
            'iou_threshold': .5, 'n_images': len(dataset),
            'mean_inference_ms': sum(warm_times) / max(len(warm_times), 1)}


def main():
    p = argparse.ArgumentParser()
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint', type=Path)
    source.add_argument('--pretrained-baseline', action='store_true')
    p.add_argument('--data', type=Path, default=Path('data/PennFudanPed'))
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--device', default='cpu')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    splits = split_names(args.data, args.seed)
    if args.checkpoint:
        model, config = load_checkpoint(args.checkpoint, args.device)
        if config['splits'] != splits:
            raise ValueError('Dataset split differs from checkpoint. Check data and --seed.')
    else:
        model = build_model(True, 'frozen', num_classes=91).to(args.device)
    dataset = PennFudan(args.data, splits[args.split])
    result = evaluate_model(model, dataset, args.device, args.threshold, args.out)
    result.update({'split': args.split, 'seed': args.seed,
                   'checkpoint': str(args.checkpoint), 'pretrained_baseline': args.pretrained_baseline})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
