"""
Train DINOv3 + LoRA for CT COVID-19 multi-view classification.
Supports CTCOVID_TASK1 (binary, 4 sources) and CTCOVID_TASK2 (4 classes, gender).
"""
import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from dinov3.build_dino import build_dinov3
from trainer import CTCOVIDTrainer

DATASET_CONFIGS = {
    'CTCOVID_TASK1': {
        'num_classes': 2,
        'img_dim': (256, 256),
        'img_size': 256,
        'num_views': 16,
        'root_path': './data/task_1',
        'index_csv': 'task1_npy_index.csv',
        'task': 'task1',
        'use_lora': True,
        'lora_rank': 16,
        'warmup_period': 10,
        'use_view_fusion': True,
        'view_fusion_type': 'concat',
        'view_axis': 'z',
    },
    'CTCOVID_TASK2': {
        'num_classes': 4,
        'img_dim': (256, 256),
        'img_size': 256,
        'num_views': 16,
        'root_path': './data/task_2',
        'index_csv': 'task2_npy_index.csv',
        'task': 'task2',
        'use_lora': True,
        'lora_rank': 16,
        'warmup_period': 10,
        'use_view_fusion': True,
        'view_fusion_type': 'concat',
        'view_axis': 'z',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, help='root dir for data')
parser.add_argument('--output', type=str, default='./output/DINO')
parser.add_argument('--dataset', type=str, default='CTCOVID_TASK1',
                    choices=['CTCOVID_TASK1', 'CTCOVID_TASK2'])
parser.add_argument('--num_classes', type=int)
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--stop_epoch', type=int, default=160)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--n_gpu', type=int, default=1)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.0005)
parser.add_argument('--img_size', type=int)
parser.add_argument('--num_views', type=int, help='number of axial slices per volume')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--dino_size', type=str, default='base', choices=['base', 'large', 'huge'])
parser.add_argument('--lora_rank', type=int)
parser.add_argument('--warmup_period', type=int)
parser.add_argument('--snapshot', type=str, default=None)
parser.add_argument('--val_interval', type=int, default=10, help='Run validation every N epochs')
parser.add_argument('--use_view_fusion', action='store_true', help='Fuse view features (concat) instead of mean logits')
parser.add_argument('--no_view_fusion', action='store_true', help='Disable view fusion (use mean over logits)')
parser.add_argument('--view_fusion_type', type=str, default='concat', choices=['concat', 'mean'])
parser.add_argument('--view_axis', type=str, default='z',
                    choices=['z', 'y', 'x', 'all'],
                    help='View axis: z=axial, y=coronal, x=sagittal, all=all 3 axes')
parser.add_argument('--augment', action='store_true', default=False,
                    help='Enable intensity augmentation during training (multi-site CT)')
parser.add_argument('--mixup_alpha', type=float, default=0.0,
                    help='Mixup alpha (Beta dist). 0=disabled. Typical: 0.2 or 0.4')
parser.add_argument('--mixup_prob', type=float, default=1.0,
                    help='Per-batch probability of applying mixup. 1.0=always (default)')
parser.add_argument('--content_sampling', action='store_true', default=False,
                    help='Content-based slice sampling (filter background by std)')
parser.add_argument('--slice_jitter', type=int, default=2,
                    help='Max random offset for slice indices when training (default: 2)')
parser.add_argument('--content_threshold', type=float, default=0.3,
                    help='Std threshold ratio for content filtering (default: 0.3)')
parser.add_argument('--val_split', type=str, default='val', choices=['train', 'val'],
                    help='Split for validation: val (default) or train')

args = parser.parse_args()

config = DATASET_CONFIGS[args.dataset]
for key, value in config.items():
    if not hasattr(args, key) or getattr(args, key) is None:
        setattr(args, key, value)
# Handle view fusion flags
if getattr(args, 'no_view_fusion', False):
    args.use_view_fusion = False


def _build_snapshot_path(args):
    """Build snapshot path from args."""
    vf_suffix = '_vf' + getattr(args, 'view_fusion_type', 'concat')[:4] if getattr(args, 'use_view_fusion', False) else ''
    axis_suffix = '_' + getattr(args, 'view_axis', 'z') if getattr(args, 'view_axis', 'z') != 'z' else ''
    args.exp = f"{args.dataset}_{args.img_size}_nv{args.num_views}{axis_suffix}{vf_suffix}"
    p = os.path.join(args.output, args.exp)
    p += f'_dinov3_{args.dino_size}_epo{args.max_epochs}_bs{args.batch_size}'
    if getattr(args, 'augment', False):
        p += '_aug'
    if getattr(args, 'mixup_alpha', 0.0) > 0:
        p += f'_mixup{args.mixup_alpha}'
    if getattr(args, 'mixup_alpha', 0.0) > 0 and getattr(args, 'mixup_prob', 1.0) < 1.0:
        p += f'_mp{args.mixup_prob}'
    if getattr(args, 'content_sampling', False):
        p += '_cs'
    if getattr(args, 'content_sampling', False) and getattr(args, 'slice_jitter', 2) != 2:
        p += f'_sj{args.slice_jitter}'
    if getattr(args, 'val_split', 'val') != 'val':
        p += f'_vs{args.val_split}'
    return p


if __name__ == "__main__":
    snapshot_path = _build_snapshot_path(args)
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(snapshot_path, "log.txt"),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
        force=True,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    logging.info("=" * 60)
    logging.info("CT COVID-19 Training - Starting")
    logging.info("=" * 60)
    logging.info(f"Dataset: {args.dataset} | root_path: {args.root_path} | view_axis: {getattr(args, 'view_axis', 'z')}")
    logging.info(f"Output: {snapshot_path}")

    logging.info("Setting random seed...")
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info(f"Seed: {args.seed} | CUDA available: {torch.cuda.is_available()}")

    logging.info("Building DINOv3 model...")
    net = build_dinov3(
        dino_size=args.dino_size,
        num_classes=config['num_classes'],
        img_dim=config['img_dim'],
        use_lora=config['use_lora'],
        lora_rank=args.lora_rank,
        use_view_fusion=getattr(args, 'use_view_fusion', False),
        num_views=args.num_views,
        view_fusion_type=getattr(args, 'view_fusion_type', 'concat'),
    ).cuda()
    vf = "concat" if getattr(args, 'use_view_fusion', False) else "mean_logits"
    logging.info(f"Model: DINOv3 {args.dino_size} | num_classes: {config['num_classes']} | LoRA rank: {args.lora_rank} | view_fusion: {vf}")

    if args.snapshot:
        logging.info(f"Loading snapshot: {args.snapshot}")
        net.load_state_dict(torch.load(args.snapshot))
        logging.info("Snapshot loaded.")

    config_file = os.path.join(snapshot_path, 'config.txt')
    with open(config_file, 'w') as f:
        for k, v in args.__dict__.items():
            f.write(f'{k}: {v}\n')
    logging.info(f"Config saved to {config_file}")

    logging.info("Creating trainer and loading data...")
    trainer = CTCOVIDTrainer(args, net, snapshot_path)
    logging.info("Starting training loop...")
    trainer.train()
