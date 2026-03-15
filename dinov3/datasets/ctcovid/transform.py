"""Collate functions and augmentation for CT COVID multi-view batches."""
import numpy as np
import torch

from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianNoiseTransform,
    GaussianBlurTransform,
)
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform


# ---------------------------------------------------------------------------
# Intensity augmentation pipeline for multi-site CT
# ---------------------------------------------------------------------------

def get_train_transform(patch_size=(256, 256)):
    """Build intensity augmentation pipeline for CT COVID training.

    Focuses on intensity/noise transforms to handle cross-site variation
    (different scanners, protocols, reconstruction kernels). Spatial
    deformation is omitted since classification does not require it.

    Args:
        patch_size: (H, W) of input slices.

    Returns:
        Composed batchgenerators transform operating on {'data': (B, C, H, W)}.
    """
    tr_transforms = [
        # Scanner noise variation between sites
        GaussianNoiseTransform(p_per_sample=0.1),
        # Reconstruction kernel blur
        GaussianBlurTransform(
            blur_sigma=(0.5, 1.0),
            different_sigma_per_channel=True,
            p_per_channel=0.5,
            p_per_sample=0.2,
        ),
        # Window/level variation across institutions
        BrightnessMultiplicativeTransform(
            (0.75, 1.25), p_per_sample=0.15,
        ),
        # Non-linear contrast variation
        GammaTransform(
            gamma_range=(0.7, 1.5),
            invert_image=False,
            per_channel=True,
            retain_stats=True,
            p_per_sample=0.15,
        ),
        # Multi-site scanner resolution differences
        SimulateLowResolutionTransform(
            zoom_range=(0.5, 1),
            per_channel=True,
            p_per_channel=0.5,
            order_downsample=0,
            order_upsample=3,
            p_per_sample=0.25,
        ),
        # Horizontal flip only (CT axial is left-right symmetric, not up-down)
        MirrorTransform(axes=(1,)),
    ]
    return Compose(tr_transforms)


# ---------------------------------------------------------------------------
# Collate functions
# ---------------------------------------------------------------------------

def collate_ctcovid_train(batch):
    """
    Collate batch for training.
    Returns dict with:
    - data: (B, N, 3, H, W) stacked
    - label: (B,) long
    - source_label: (B,) int
    - gender_label: (B,) int
    - volume_id: list of str
    """
    data = np.stack([b['data'] for b in batch], axis=0)
    label = np.array([b['label'] for b in batch], dtype=np.int64)
    source_label = np.array([b['source_label'] for b in batch], dtype=np.int32)
    gender_label = np.array([b['gender_label'] for b in batch], dtype=np.int32)
    volume_id = [b['volume_id'] for b in batch]

    return {
        'data': data,
        'label': label,
        'source_label': source_label,
        'gender_label': gender_label,
        'volume_id': volume_id,
    }


def collate_ctcovid_train_aug(batch):
    """Collate with intensity augmentation for training.

    Stacks batch → reshapes (B, N, 3, H, W) into (B*N, 3, H, W) so each
    slice is treated as an independent sample for batchgenerators, applies
    intensity transforms, then reshapes back.
    """
    data = np.stack([b['data'] for b in batch], axis=0)  # (B, N, 3, H, W)
    label = np.array([b['label'] for b in batch], dtype=np.int64)
    source_label = np.array([b['source_label'] for b in batch], dtype=np.int32)
    gender_label = np.array([b['gender_label'] for b in batch], dtype=np.int32)
    volume_id = [b['volume_id'] for b in batch]

    B, N, C, H, W = data.shape
    # Reshape to (B*N, C, H, W) for batchgenerators
    flat = data.reshape(B * N, C, H, W)

    # Apply augmentation
    tr_transforms = get_train_transform(patch_size=(H, W))
    aug_dict = tr_transforms(**{'data': flat})
    flat_aug = aug_dict['data']

    # Reshape back to (B, N, C, H, W)
    data = flat_aug.reshape(B, N, C, H, W).astype(np.float32)

    return {
        'data': data,
        'label': label,
        'source_label': source_label,
        'gender_label': gender_label,
        'volume_id': volume_id,
    }


def collate_ctcovid_val(batch):
    """Same as train for validation (no augment)."""
    return collate_ctcovid_train(batch)
