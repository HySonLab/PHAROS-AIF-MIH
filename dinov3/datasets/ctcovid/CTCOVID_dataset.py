"""
CT COVID-19 multi-view dataset.
Load 3D .npy volume, sample N slices along chosen axis/axes, return (N, 3, H, W) for DINO.
"""
import os
import numpy as np
from torch.utils.data import Dataset
from .normalize import normalize_ct_to_01, grayscale_to_rgb


# Task 1: binary
CLASS_TO_IDX_TASK1 = {'non-covid': 0, 'covid': 1}

# Task 2: 4 classes
CLASS_TO_IDX_TASK2 = {'A': 0, 'G': 1, 'covid': 2, 'normal': 3}

# Axis mapping: z=axial(0), y=coronal(1), x=sagittal(2)
AXIS_MAP = {'z': 0, 'axial': 0, 'y': 1, 'coronal': 1, 'x': 2, 'sagittal': 2}


def _sample_slice_indices(dim: int, n: int) -> np.ndarray:
    """Uniform sampling of n indices from [0, dim-1]."""
    if n >= dim:
        return np.arange(dim)
    indices = np.linspace(0, dim - 1, n, dtype=int)
    return indices


def _get_slice(vol: np.ndarray, axis: int, idx: int) -> np.ndarray:
    """Extract 2D slice from volume along axis. Returns (H, W)."""
    if axis == 0:
        return vol[idx]  # (H, W)
    elif axis == 1:
        return vol[:, idx, :]  # (D, W)
    else:  # axis == 2
        return vol[:, :, idx]  # (D, H)


def _find_content_range(vol: np.ndarray, axis: int, threshold_ratio: float = 0.3) -> tuple:
    """Find slice range with content (std > threshold). Returns (start, end) exclusive end."""
    dim = vol.shape[axis]
    stds = np.array([np.std(_get_slice(vol, axis, i)) for i in range(dim)])
    max_std = stds.max()
    if max_std <= 0:
        return 0, dim
    threshold = max_std * threshold_ratio
    valid = np.where(stds > threshold)[0]
    if len(valid) == 0:
        return 0, dim
    return int(valid[0]), int(valid[-1]) + 1


def _sample_slice_indices_content(
    start: int, end: int, n: int, jitter: int = 2, training: bool = False
) -> np.ndarray:
    """Uniform sample n indices in [start, end). If training, add random jitter."""
    effective = end - start
    if n >= effective:
        indices = np.arange(start, end)
    else:
        indices = np.linspace(start, end - 1, n, dtype=int)
    if training and jitter > 0:
        offsets = np.random.randint(-jitter, jitter + 1, size=len(indices))
        indices = np.clip(indices + offsets, start, end - 1)
    return indices


def _resize_slice(slc: np.ndarray, target_size: int) -> np.ndarray:
    """Resize slice (3, H, W) to (3, target_size, target_size)."""
    if slc.shape[1] == target_size and slc.shape[2] == target_size:
        return slc
    from scipy.ndimage import zoom
    scale_h = target_size / slc.shape[1]
    scale_w = target_size / slc.shape[2]
    return zoom(slc, (1, scale_h, scale_w), order=1).astype(np.float32)


class CTCOVIDDataset(Dataset):
    """
    Dataset for CT COVID-19. Loads 3D .npy, samples N slices along chosen axis/axes.
    Returns views (N, 3, H, W), label (int), source_label (Task1), gender_label (Task2), volume_id.
    """
    def __init__(
        self,
        root_path: str,
        rows: list,
        task: str,
        num_views: int = 16,
        img_size: int = 256,
        augment: bool = False,
        view_axis: str = "z",
        content_sampling: bool = False,
        slice_jitter: int = 2,
        content_threshold: float = 0.3,
        is_train: bool = False,
    ):
        """
        Args:
            root_path: Path to task folder (e.g. ./data/task_1).
            rows: List of dict from convert_csv (npy_path, class_label, source_label, etc).
            task: 'task1' or 'task2'.
            num_views: Number of slices to sample per volume.
            img_size: Output slice size (H, W).
            augment: Whether to apply augmentation (handled in collate).
            view_axis: 'z'|'y'|'x' (single axis) or 'all' (sample from all 3 axes).
                       z=axial, y=coronal, x=sagittal.
            content_sampling: If True, sample only from content-rich region (std-based).
            slice_jitter: Max random offset for slice indices when training.
            content_threshold: Std threshold ratio (0-1) for content filtering.
            is_train: If True, apply slice jitter augmentation.
        """
        self.root_path = root_path
        self.rows = rows
        self.task = task
        self.num_views = num_views
        self.img_size = img_size
        self.augment = augment
        self.view_axis = view_axis.lower()
        self.content_sampling = content_sampling
        self.slice_jitter = slice_jitter
        self.content_threshold = content_threshold
        self.is_train = is_train

        if task == 'task1':
            self.class_to_idx = CLASS_TO_IDX_TASK1
        else:
            self.class_to_idx = CLASS_TO_IDX_TASK2

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        npy_path = row['npy_path']
        full_path = os.path.join(self.root_path, npy_path)

        vol = np.load(full_path)  # (D, H, W) e.g. (128, 256, 256)
        D, H, W = vol.shape
        dims = [D, H, W]

        slices = []
        if self.view_axis == "all":
            n_per = max(1, self.num_views // 3)
            remainder = self.num_views - 3 * n_per
            counts = [n_per, n_per, n_per]
            for i in range(remainder):
                counts[i] += 1
            for axis, count in zip([0, 1, 2], counts):
                if self.content_sampling:
                    start, end = _find_content_range(vol, axis, self.content_threshold)
                    indices = _sample_slice_indices_content(
                        start, end, count,
                        jitter=self.slice_jitter,
                        training=self.is_train,
                    )
                else:
                    indices = _sample_slice_indices(dims[axis], count)
                for i in indices:
                    slc = _get_slice(vol, axis, i)
                    slc = normalize_ct_to_01(slc)
                    slc = grayscale_to_rgb(slc)  # (3, H, W)
                    slc = _resize_slice(slc, self.img_size)  # (3, img_size, img_size)
                    slices.append(slc)
        else:
            axis = AXIS_MAP.get(self.view_axis, 0)
            if self.content_sampling:
                start, end = _find_content_range(vol, axis, self.content_threshold)
                indices = _sample_slice_indices_content(
                    start, end, self.num_views,
                    jitter=self.slice_jitter,
                    training=self.is_train,
                )
            else:
                indices = _sample_slice_indices(dims[axis], self.num_views)
            for i in indices:
                slc = _get_slice(vol, axis, i)
                slc = normalize_ct_to_01(slc)
                slc = grayscale_to_rgb(slc)  # (3, H, W)
                slc = _resize_slice(slc, self.img_size)  # (3, img_size, img_size)
                slices.append(slc)

        views = np.stack(slices, axis=0).astype(np.float32)  # (N, 3, img_size, img_size)

        class_label = row['class_label']
        label = self.class_to_idx.get(class_label, 0)

        source_label = int(row.get('source_label', 0)) if 'source_label' in row else 0
        gender_label = row.get('gender_label', 'male')
        gender_int = 1 if gender_label == 'female' else 0
        volume_id = row.get('ct_scan_name', str(idx))

        return {
            'data': views,
            'label': label,
            'source_label': source_label,
            'gender_label': gender_int,
            'volume_id': volume_id,
        }
