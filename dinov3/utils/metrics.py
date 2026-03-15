import numpy as np
from sklearn.metrics import f1_score


def compute_macro_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Macro F1 for classification. preds, labels: integer class indices."""
    if len(preds) == 0:
        return 0.0
    return float(f1_score(labels, preds, average='macro', zero_division=0))


def compute_weighted_f1(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Weighted F1 for classification. preds, labels: integer class indices."""
    if len(preds) == 0:
        return 0.0
    return float(f1_score(labels, preds, average='weighted', zero_division=0))
