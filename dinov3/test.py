"""
Test DINOv3 + LoRA for CT COVID-19 multi-view classification.
"""
import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dinov3.build_dino import build_dinov3
from trainer import CTCOVIDTrainer
from datasets.ctcovid import load_task1_index, load_task2_index, filter_by_split, CTCOVIDDataset, collate_ctcovid_val

DATASET_CONFIGS = {
    'CTCOVID_TASK1': {
        'num_classes': 2,
        'img_dim': (256, 256),
        'img_size': 256,
        'num_views': 16,
        'root_path': './data/task_1',
        'index_csv': 'task1_npy_index.csv',
        'warmup_period': 10,
        'view_axis': 'z',
    },
    'CTCOVID_TASK2': {
        'num_classes': 4,
        'img_dim': (256, 256),
        'img_size': 256,
        'num_views': 16,
        'root_path': './data/task_2',
        'index_csv': 'task2_npy_index.csv',
        'warmup_period': 10,
        'view_axis': 'z',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CTCOVID_TASK1', choices=['CTCOVID_TASK1', 'CTCOVID_TASK2'])
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--snapshot', type=str, required=True, help='Path to best_model.pth')
    parser.add_argument('--output', type=str, default='./output/test_log')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dino_size', type=str, default='base')
    parser.add_argument('--use_view_fusion', action='store_true', help='Model was trained with view fusion')
    parser.add_argument('--view_fusion_type', type=str, default='concat', choices=['concat', 'mean'])
    parser.add_argument('--view_axis', type=str, default='z', choices=['z', 'y', 'x', 'all'])
    parser.add_argument('--content_sampling', action='store_true', default=False,
                        help='Use content-based slice sampling (match training config)')
    parser.add_argument('--content_threshold', type=float, default=0.3,
                        help='Std threshold for content filtering (match training)')
    args = parser.parse_args()

    config = DATASET_CONFIGS[args.dataset]
    for k, v in config.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    if args.root_path is None:
        args.root_path = config['root_path']

    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output, exist_ok=True)
    log_file = os.path.join(args.output, 'log.txt')

    import logging
    logging.basicConfig(filename=log_file, level=logging.INFO, format='[%(asctime)s] %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(str(args))

    net = build_dinov3(
        dino_size=args.dino_size,
        num_classes=config['num_classes'],
        img_dim=config['img_dim'],
        use_lora=True,
        lora_rank=16,
        use_view_fusion=getattr(args, 'use_view_fusion', False),
        num_views=config['num_views'],
        view_fusion_type=getattr(args, 'view_fusion_type', 'concat'),
    ).cuda()
    net.load_state_dict(torch.load(args.snapshot, map_location='cuda'))
    net.eval()

    load_index = load_task1_index if args.dataset == 'CTCOVID_TASK1' else load_task2_index
    rows = load_index(args.root_path, args.index_csv)
    rows = filter_by_split(rows, 'val')
    dataset = CTCOVIDDataset(
        root_path=args.root_path,
        rows=rows,
        task='task1' if args.dataset == 'CTCOVID_TASK1' else 'task2',
        num_views=args.num_views,
        img_size=args.img_size,
        augment=False,
        view_axis=getattr(args, 'view_axis', 'z'),
        content_sampling=getattr(args, 'content_sampling', False),
        slice_jitter=0,
        content_threshold=getattr(args, 'content_threshold', 0.3),
        is_train=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_ctcovid_val, num_workers=0)

    trainer = CTCOVIDTrainer(args, net, args.output)
    results = trainer.inference_step(0, loader, 0)
    logging.info(f"Test macro_f1: {results['macro_f1']:.4f}, weighted_f1: {results['weighted_f1']:.4f}, n={results['num_cases']}")

    # Save test preds CSV for error analysis
    csv_path = os.path.join(args.output, 'test_preds.csv')
    if results['num_cases'] > 0:
        preds = results['preds']
        labels = results['labels']
        volume_ids = results.get('volume_ids', [f'idx_{i}' for i in range(len(preds))])
        groups = results.get('groups', [''] * len(preds))
        logits = results.get('logits_list', np.array([]))
        num_classes = config['num_classes']
        rows = []
        for i in range(len(preds)):
            row = {
                'volume_id': volume_ids[i] if i < len(volume_ids) else f'idx_{i}',
                'group': groups[i] if i < len(groups) else '',
                'label': int(labels[i]),
                'pred': int(preds[i]),
                'correct': int(preds[i] == labels[i]),
            }
            if logits.size > 0 and i < len(logits):
                for c in range(num_classes):
                    row[f'logit_{c}'] = f"{float(logits[i, c]):.6f}"
            rows.append(row)
        fieldnames = ['volume_id', 'group', 'label', 'pred', 'correct']
        if rows and 'logit_0' in rows[0]:
            fieldnames += [f'logit_{c}' for c in range(num_classes)]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Saved test preds to {csv_path}")


if __name__ == '__main__':
    main()
