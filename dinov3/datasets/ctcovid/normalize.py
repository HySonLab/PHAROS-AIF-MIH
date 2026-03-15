"""Normalize CT slice to [0,1] and convert grayscale to 3 channels for DINO."""
import numpy as np


def normalize_ct_to_01(x: np.ndarray) -> np.ndarray:
    """Normalize CT values to [0, 1]."""
    vmin, vmax = x.min(), x.max()
    if vmax > vmin:
        x = (x.astype(np.float32) - vmin) / (vmax - vmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x


def grayscale_to_rgb(x: np.ndarray) -> np.ndarray:
    """Repeat grayscale (H, W) to (3, H, W) for DINO."""
    if x.ndim == 2:
        return np.repeat(x[np.newaxis, :, :], 3, axis=0)
    return x


def normalize_image(img_npy: np.ndarray) -> np.ndarray:
    """Per-sample per-channel z-score normalization for DINOv3 input.

    Args:
        img_npy: (B, C, H, W) float array.

    Returns:
        z-score normalized array of the same shape.
    """
    eps = 1e-6
    if getattr(img_npy, 'dtype', None) is not None and str(img_npy.dtype) != 'float32':
        img_npy = img_npy.astype('float32', copy=False)
    for b in range(img_npy.shape[0]):
        for c in range(img_npy.shape[1]):
            ch = img_npy[b, c]
            m = ch.mean()
            s = ch.std()
            if s < eps:
                img_npy[b, c] = ch - m
            else:
                img_npy[b, c] = (ch - m) / (s + eps)
    return img_npy
