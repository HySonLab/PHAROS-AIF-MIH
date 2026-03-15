"""
Classification head for DINO encoder: GAP + Linear.
Used for CT COVID-19 multi-view classification.
"""
import torch
import torch.nn as nn


class ClassificationDecoder(nn.Module):
    """
    Global average pooling + linear classifier.
    Input: [B, C, h, w] -> Output: [B, n_classes]
    """
    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(emb_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, h, w] from encoder intermediate layers
        Returns:
            logits: [B, n_classes]
        """
        x = self.pool(x)  # [B, C, 1, 1]
        x = x.flatten(1)  # [B, C]
        return self.fc(x)
