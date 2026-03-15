"""
CT COVID-19 multi-view classification trainer.
Supports Task 1 (binary, 4 sources) and Task 2 (4 classes, male/female).
"""
import csv
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.ctcovid import (
    CTCOVIDDataset,
    load_task1_index,
    load_task2_index,
    filter_by_split,
    collate_ctcovid_train,
    collate_ctcovid_train_aug,
    collate_ctcovid_val,
    normalize_image,
)
from utils.metrics import compute_macro_f1, compute_weighted_f1


def mixup_data(x, y, alpha):
    """Mixup: x_mixed = lam*x + (1-lam)*x[shuffle], loss = lam*CE + (1-lam)*CE.
    x: (B, N, C, H, W), y: (B,). Returns mixed_x, y_a, y_b, lam."""
    if alpha <= 0:
        return x, y, None, 1.0
    lam = np.random.beta(alpha, alpha)
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


TASK_CONFIG = {
    'CTCOVID_TASK1': {
        'task': 'task1',
        'num_classes': 2,
        'load_index': load_task1_index,
        'filter_key': 'source_label',
        'filter_values': [0, 1, 2, 3],
        'target_names': ['Source0', 'Source1', 'Source2', 'Source3'],
    },
    'CTCOVID_TASK2': {
        'task': 'task2',
        'num_classes': 4,
        'load_index': load_task2_index,
        'filter_key': 'gender_label',
        'filter_values': ['male', 'female'],
        'target_names': ['Male', 'Female'],
    },
}


class CTCOVIDTrainer:
    """CT COVID-19 multi-view classification. Supports Task 1 and Task 2 via config."""

    def __init__(self, args, model, snapshot_path):
        self.args = args
        self.model = model
        self.snapshot_path = snapshot_path

        if not logging.getLogger().handlers:
            logging.basicConfig(
                filename=snapshot_path + "/log.txt",
                level=logging.INFO,
                format='[%(asctime)s.%(msecs)03d] %(message)s',
                datefmt='%H:%M:%S'
            )
            logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
        logging.info("Trainer initialized. Args: " + str(args)[:200] + "...")
        self.writer = SummaryWriter(snapshot_path + '/log')

        if args.n_gpu > 1:
            self.model = nn.DataParallel(model)
        self.model.train()

        self.iter_num = 0
        self.best_epoch = -1
        self.best_results = {}

        self.ce_loss = nn.CrossEntropyLoss()
        self.num_views = getattr(args, 'num_views', 16)
        config = TASK_CONFIG.get(args.dataset, TASK_CONFIG['CTCOVID_TASK1'])
        self.task = config['task']
        self.num_classes = config['num_classes']
        self.load_index = config['load_index']
        self.filter_key = config['filter_key']
        self.filter_values = config['filter_values']
        self.target_names = config['target_names']
        self.val_filter_values = self.filter_values
        self.val_target_names = self.target_names
        self.val_split = getattr(args, 'val_split', 'val')

    def _match_filter(self, row, value):
        """Match row against filter value (int or str)."""
        if isinstance(value, int):
            return int(row.get(self.filter_key, 0)) == value
        return row.get(self.filter_key) == value

    def setup_optimizer_and_scheduler(self):
        base_lr = self.args.base_lr
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=base_lr,
            weight_decay=1e-2
        )
        warmup_sched = LambdaLR(
            optimizer,
            lambda epoch: min(1.0, (epoch + 1) / self.args.warmup_period)
        )
        cos_sched = CosineAnnealingLR(
            optimizer,
            T_max=self.args.max_epochs - self.args.warmup_period,
            eta_min=base_lr * 0.01
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cos_sched],
            milestones=[self.args.warmup_period]
        )
        logging.info(f"Using warmup for {self.args.warmup_period} epochs, then cosine annealing")
        return optimizer, scheduler

    def get_train_loader(self):
        rows = self.load_index(self.args.root_path, self.args.index_csv)
        rows = filter_by_split(rows, 'train')
        content_sampling = getattr(self.args, 'content_sampling', False)
        dataset = CTCOVIDDataset(
            root_path=self.args.root_path,
            rows=rows,
            task=self.task,
            num_views=self.args.num_views,
            img_size=self.args.img_size,
            augment=False,
            view_axis=getattr(self.args, 'view_axis', 'z'),
            content_sampling=content_sampling,
            slice_jitter=getattr(self.args, 'slice_jitter', 2),
            content_threshold=getattr(self.args, 'content_threshold', 0.3),
            is_train=True,
        )
        use_aug = getattr(self.args, 'augment', False)
        collate_fn = collate_ctcovid_train_aug if use_aug else collate_ctcovid_train
        if use_aug:
            logging.info("Training with intensity augmentation enabled")
        if content_sampling:
            jitter = getattr(self.args, 'slice_jitter', 2)
            thresh = getattr(self.args, 'content_threshold', 0.3)
            logging.info(f"Content-based slice sampling (threshold={thresh}, jitter=±{jitter})")
        if getattr(self.args, 'mixup_alpha', 0.0) > 0:
            prob = getattr(self.args, 'mixup_prob', 1.0)
            logging.info(f"Training with mixup (alpha={self.args.mixup_alpha}, prob={prob})")
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def get_valid_loaders(self):
        rows = self.load_index(self.args.root_path, self.args.index_csv)
        rows = filter_by_split(rows, self.val_split)
        loaders = []
        for value in self.val_filter_values:
            subset = [r for r in rows if self._match_filter(r, value)]
            if not subset:
                loaders.append(DataLoader([], batch_size=1))
                continue
            content_sampling = getattr(self.args, 'content_sampling', False)
            ds = CTCOVIDDataset(
                root_path=self.args.root_path,
                rows=subset,
                task=self.task,
                num_views=self.args.num_views,
                img_size=self.args.img_size,
                augment=False,
                view_axis=getattr(self.args, 'view_axis', 'z'),
                content_sampling=content_sampling,
                slice_jitter=0,
                content_threshold=getattr(self.args, 'content_threshold', 0.3),
                is_train=False,
            )
            loaders.append(DataLoader(
                ds, batch_size=1, shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=collate_ctcovid_val,
            ))
        return loaders

    def training_step(self, batch_data, batch_idx):
        data = batch_data['data']  # (B, N, 3, H, W)
        label = batch_data['label']  # (B,)
        B, N = data.shape[0], data.shape[1]
        x = data.reshape(B * N, data.shape[2], data.shape[3], data.shape[4])
        x = normalize_image(x)  # z-score per-channel before DINOv3
        x = torch.from_numpy(x).float().cuda()
        label = torch.from_numpy(label).long().cuda()

        mixup_alpha = getattr(self.args, 'mixup_alpha', 0.0)
        mixup_prob = getattr(self.args, 'mixup_prob', 1.0)
        apply_mixup = mixup_alpha > 0 and np.random.random() < mixup_prob

        if apply_mixup:
            x = x.view(B, N, data.shape[2], data.shape[3], data.shape[4])
            x, y_a, y_b, lam = mixup_data(x, label, mixup_alpha)
            x = x.view(B * N, data.shape[2], data.shape[3], data.shape[4])

        use_view_fusion = getattr(self.args, 'use_view_fusion', False)
        if use_view_fusion and hasattr(self.model, 'view_fusion_head') and self.model.view_fusion_head is not None:
            logits = self.model(x, num_views=N)  # (B, num_classes)
        else:
            logits_per_view = self.model(x)  # (B*N, num_classes)
            logits = logits_per_view.view(B, N, -1).mean(dim=1)  # (B, num_classes)

        if apply_mixup:
            loss = lam * self.ce_loss(logits, y_a) + (1 - lam) * self.ce_loss(logits, y_b)
        else:
            loss = self.ce_loss(logits, label)
        return {'total_loss': loss, 'loss': loss, 'loss_ce': loss}

    def inference_step(self, epoch_num, test_loader, domain_idx):
        preds, labels = [], []
        volume_ids = []
        groups = []
        logits_list = []
        group_name = self.val_target_names[domain_idx] if self.val_target_names and domain_idx < len(self.val_target_names) else (self.target_names[domain_idx] if domain_idx < len(self.target_names) else str(domain_idx))
        desc = group_name
        for batch_data in tqdm(test_loader, desc=desc, leave=False):
            data = batch_data['data']
            label = batch_data['label']
            vol_ids = batch_data.get('volume_id', [])
            B, N = data.shape[0], data.shape[1]
            x = data.reshape(B * N, data.shape[2], data.shape[3], data.shape[4])
            x = normalize_image(x)  # z-score per-channel before DINOv3
            x = torch.from_numpy(x).float().cuda()

            with torch.no_grad():
                use_view_fusion = getattr(self.args, 'use_view_fusion', False)
                if use_view_fusion and hasattr(self.model, 'view_fusion_head') and self.model.view_fusion_head is not None:
                    logits = self.model(x, num_views=N)  # (B, num_classes)
                else:
                    logits = self.model(x)
                    logits = logits.view(B, N, -1).mean(dim=1)
                pred = logits.argmax(dim=1).cpu().numpy()

            preds.extend(pred.tolist())
            labels.extend(label.tolist())
            volume_ids.extend(vol_ids if isinstance(vol_ids, list) else [vol_ids])
            logits_list.append(logits.cpu().numpy())
            # Per-sample group (for mixed loaders like test; for valid loaders all same)
            filter_vals = batch_data.get(self.filter_key, np.full(B, domain_idx))
            if isinstance(filter_vals, np.ndarray):
                filter_vals = filter_vals.tolist()
            for v in (filter_vals if isinstance(filter_vals, list) else [filter_vals]):
                g = self.val_target_names[v] if self.val_target_names and isinstance(v, int) and v < len(self.val_target_names) else (self.target_names[v] if isinstance(v, int) and v < len(self.target_names) else str(v))
                groups.append(g)

        if not preds:
            return {'macro_f1': 0.0, 'weighted_f1': 0.0, 'num_cases': 0, 'preds': [], 'labels': [],
                    'volume_ids': [], 'groups': [], 'logits_list': [], 'group_name': group_name}

        preds_arr = np.array(preds)
        labels_arr = np.array(labels)
        macro_f1 = compute_macro_f1(preds_arr, labels_arr, num_classes=self.num_classes)
        weighted_f1 = compute_weighted_f1(preds_arr, labels_arr, num_classes=self.num_classes)
        logits_arr = np.concatenate(logits_list, axis=0) if logits_list else np.array([])
        return {
            'macro_f1': macro_f1, 'weighted_f1': weighted_f1, 'num_cases': len(preds),
            'preds': preds, 'labels': labels, 'volume_ids': volume_ids, 'groups': groups,
            'logits_list': logits_arr, 'group_name': group_name,
        }

    def _save_best_valid_csv(self, all_results: list, epoch_num: int):
        """Save CSV for error analysis (overwrites each time best is achieved)."""
        csv_path = os.path.join(self.snapshot_path, 'best_valid_preds.csv')
        rows = []
        for r in all_results:
            if r['num_cases'] == 0:
                continue
            preds = r['preds']
            labels = r['labels']
            volume_ids = r.get('volume_ids', [f'idx_{i}' for i in range(len(preds))])
            groups = r.get('groups', [r.get('group_name', '')] * len(preds))
            logits = r.get('logits_list', np.array([]))
            if len(volume_ids) < len(preds):
                volume_ids = volume_ids + [f'idx_{i}' for i in range(len(volume_ids), len(preds))]
            if len(groups) < len(preds):
                groups = groups + [r.get('group_name', '')] * (len(preds) - len(groups))
            for i in range(len(preds)):
                row = {
                    'volume_id': volume_ids[i] if i < len(volume_ids) else f'idx_{i}',
                    'group': groups[i] if i < len(groups) else r.get('group_name', ''),
                    'label': int(labels[i]),
                    'pred': int(preds[i]),
                    'correct': int(preds[i] == labels[i]),
                }
                if logits.size > 0 and i < len(logits):
                    for c in range(logits.shape[1]):
                        row[f'logit_{c}'] = f"{float(logits[i, c]):.6f}"
                rows.append(row)
        if not rows:
            return
        fieldnames = ['volume_id', 'group', 'label', 'pred', 'correct']
        if rows and 'logit_0' in rows[0]:
            fieldnames += [f'logit_{c}' for c in range(self.num_classes)]
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"Saved best valid preds to {csv_path} (epoch {epoch_num + 1}, n={len(rows)})")

    def train(self):
        logging.info("Loading train data...")
        trainloader = self.get_train_loader()
        n_train = len(trainloader.dataset)
        logging.info(f"Train: {n_train} samples, {len(trainloader)} batches (bs={self.args.batch_size})")

        logging.info("Loading validation data...")
        valid_loaders = self.get_valid_loaders()
        n_val_per = [len(ld.dataset) for ld in valid_loaders]
        logging.info(f"Valid: {n_val_per} samples per group ({self.val_target_names}), split={self.val_split}")

        logging.info("Setting up optimizer and scheduler...")
        optimizer, scheduler = self.setup_optimizer_and_scheduler()
        max_epoch = self.args.max_epochs
        stop_epoch = self.args.stop_epoch

        save_interval = getattr(self.args, 'val_interval', 10)
        logging.info(f"{len(trainloader)} iterations per epoch. {len(trainloader) * max_epoch} max iterations")
        logging.info(f"Validation every {save_interval} epochs")

        for epoch_num in range(max_epoch):
            epoch_losses = {'loss': 0.0, 'loss_ce': 0.0}

            for batch_idx, batch_data in tqdm(enumerate(trainloader), total=len(trainloader), leave=True):
                loss_dict = self.training_step(batch_data, batch_idx)
                total_loss = loss_dict['total_loss']

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                epoch_losses['loss'] += loss_dict['loss'].item()
                epoch_losses['loss_ce'] += loss_dict['loss_ce'].item()
                self.iter_num += 1

            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            n = len(trainloader)
            avg_loss = epoch_losses['loss'] / n
            self.writer.add_scalar('epoch/lr', current_lr, epoch_num)
            self.writer.add_scalar('epoch/loss', avg_loss, epoch_num)

            # Validation every save_interval epochs
            val_summary = ""
            if (epoch_num + 1) % save_interval == 0:
                self.model.eval()
                all_results = []
                with torch.no_grad():
                    for domain_idx, loader in enumerate(valid_loaders):
                        results = self.inference_step(epoch_num, loader, domain_idx)
                        all_results.append(results)
                        name = self.val_target_names[domain_idx] if domain_idx < len(self.val_target_names) else f'G{domain_idx}'
                        self.writer.add_scalar(f'Valid/{name}_macro_f1', results['macro_f1'], epoch_num)
                        self.writer.add_scalar(f'Valid/{name}_weighted_f1', results['weighted_f1'], epoch_num)

                valid = [r for r in all_results if r['num_cases'] > 0]
                if valid:
                    avg_f1 = np.mean([r['macro_f1'] for r in valid])
                    avg_weighted = np.mean([r['weighted_f1'] for r in valid])
                    total_n = sum(r['num_cases'] for r in valid)
                    val_summary = f", val_macro_f1={avg_f1:.4f}, val_weighted_f1={avg_weighted:.4f} (n={total_n})"
                    per_src = " | ".join(
                        f"{self.val_target_names[i]}=m:{r['macro_f1']:.3f}/w:{r['weighted_f1']:.3f}"
                        for i, r in enumerate(all_results) if r['num_cases'] > 0
                    )
                    logging.info(f"  -> Valid: {per_src}")

                    if not hasattr(self, 'best_metric'):
                        self.best_metric = 0.0
                    if avg_f1 > self.best_metric:
                        self.best_metric = avg_f1
                        self.best_epoch = epoch_num
                        self.best_results = {
                            'macro_f1': avg_f1, 'weighted_f1': avg_weighted,
                            'per_group': [r['macro_f1'] for r in all_results],
                            'per_group_weighted': [r['weighted_f1'] for r in all_results],
                        }
                        path = os.path.join(self.snapshot_path, 'best_model.pth')
                        torch.save(self.model.state_dict(), path)
                        self._save_best_valid_csv(all_results, epoch_num)
                        val_summary += " [BEST]"

                self.model.train()

            # Single-line log per epoch
            logging.info(f"Epoch {epoch_num + 1}/{max_epoch} | loss={avg_loss:.4f} | lr={current_lr:.6f}{val_summary}")

            if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
                save_path = os.path.join(self.snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
                torch.save(self.model.state_dict(), save_path)
                logging.info(f"save model to {save_path}")
                break

        if self.best_epoch >= 0:
            logging.info("=" * 60)
            logging.info(f"BEST: epoch {self.best_epoch + 1}, macro_f1={self.best_results['macro_f1']:.4f}, weighted_f1={self.best_results['weighted_f1']:.4f}")
            logging.info("=" * 60)
        self.writer.close()
        return "Training Finished!"
