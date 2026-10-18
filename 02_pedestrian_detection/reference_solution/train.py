"""Run two controlled fine-tuning experiments on the same image split."""
from pathlib import Path
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import PennFudan, split_names, seed_everything, collate, fingerprint
from evaluate import evaluate_model
from model import build_model, train_step, save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/PennFudanPed'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['frozen', 'finetune'], default='frozen')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--image-size', type=int, default=320)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-train', type=int, default=0, help='0 uses the full training split')
    p.add_argument('--max-val', type=int, default=0)
    p.add_argument('--no-pretrained', action='store_true', help='Pipeline debugging only; not the learning experiment')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 64 <= args.image_size <= 640:
        p.error('Use positive epochs/batch-size and image-size between 64 and 640.')
    torch.set_num_threads(2)
    seed_everything(args.seed)
    splits = split_names(args.data, args.seed)
    train_names = splits['train'][:args.max_train] if args.max_train else splits['train']
    val_names = splits['val'][:args.max_val] if args.max_val else splits['val']
    train = PennFudan(args.data, train_names, augment=True)
    val = PennFudan(args.data, val_names)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=collate, generator=torch.Generator().manual_seed(args.seed))
    model = build_model(not args.no_pretrained, args.mode, args.image_size).to(args.device)
    optimizer = torch.optim.SGD([v for v in model.parameters() if v.requires_grad],
                                lr=args.lr, momentum=.9, weight_decay=.0005)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'splits': splits, 'split_fingerprint': fingerprint(splits),
                   'torch_version': str(torch.__version__)})
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    history, best_f1 = [], -1.
    for epoch in range(args.epochs):
        seed_everything(args.seed + epoch)
        model.train()
        losses = []
        for images, targets in loader:
            images = [image.to(args.device) for image in images]
            targets = [{k: v.to(args.device) for k, v in t.items()} for t in targets]
            losses.append(train_step(model, images, targets, optimizer))
        metrics = evaluate_model(model, val, args.device)
        metrics.update({'epoch': epoch + 1, 'train_loss': sum(losses) / len(losses)})
        history.append(metrics)
        print(json.dumps(metrics), flush=True)
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            save_checkpoint(args.out / 'best.pt', model, config)
        save_checkpoint(args.out / 'last.pt', model, config)
        (args.out / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    print('Saved:', args.out / 'best.pt')


if __name__ == '__main__':
    main()
