"""
View fusion for multi-view CT classification.
Fuses features from N views into a single prediction.
"""
import torch
import torch.nn as nn


class ViewFusionHead(nn.Module):
    """
    Fuse features from N views (B, N, C) -> (B, n_classes).
    Supports: concat (concat + MLP), mean (average then linear).
    """
    def __init__(
        self,
        emb_dim: int,
        num_views: int,
        n_classes: int,
        fusion_type: str = "concat",
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_views = num_views

        if fusion_type == "concat":
            # (B, N, C) -> (B, N*C) -> MLP -> (B, n_classes)
            fused_dim = num_views * emb_dim
            self.fc = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )
        elif fusion_type == "mean":
            # (B, N, C) -> mean(dim=1) -> (B, C) -> Linear
            self.fc = nn.Linear(emb_dim, n_classes)
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C] - features from N views per volume
        Returns:
            logits: [B, n_classes]
        """
        if self.fusion_type == "concat":
            x = x.flatten(1)  # (B, N*C)
        else:  # mean
            x = x.mean(dim=1)  # (B, C)
        return self.fc(x)
