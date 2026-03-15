import math

import torch
import torch.nn as nn

from .lora import LoRA
from .classification_decoder import ClassificationDecoder
from .view_fusion import ViewFusionHead


class DINOEncoderLoRA(nn.Module):
    def __init__(
        self,
        encoder,
        r: int = 3,
        emb_dim: int = 1024,
        n_classes: int = 1000,
        use_lora: bool = False,
        img_dim: tuple[int, int] = (520, 520),
        use_view_fusion: bool = False,
        num_views: int = 16,
        view_fusion_type: str = "concat",
    ):
        """DINOv3 encoder with classification head for downstream tasks.

        Args:
            encoder: The ViT encoder loaded with DINOv3 weights.
            r: LoRA rank. Defaults to 3.
            emb_dim: Encoder embedding dimension. Defaults to 1024.
            n_classes: Number of output classes. Defaults to 1000.
            use_lora: Whether to use LoRA adaptation. Defaults to False.
            img_dim: Input image (height, width). Defaults to (520, 520).
        """
        super().__init__()
        
        # Get patch_size from encoder
        try:
            patch_size = encoder.patch_size
        except AttributeError:
            try:
                patch_size = encoder.patch_embed.patch_size[0] if hasattr(encoder.patch_embed, 'patch_size') else 16
            except:
                patch_size = 16
        
        assert img_dim[0] % patch_size == 0, f"Image height {img_dim[0]} must be divisible by patch_size {patch_size}"
        assert img_dim[1] % patch_size == 0, f"Image width {img_dim[1]} must be divisible by patch_size {patch_size}"
        assert r > 0

        self.emb_dim = emb_dim
        self.img_dim = img_dim
        self.use_lora = use_lora
        self.inter_layers = 1

        self.encoder = encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.decoder = ClassificationDecoder(emb_dim=emb_dim, n_classes=n_classes)
        self.pool = self.decoder.pool

        # Add LoRA layers to all encoder blocks
        if self.use_lora:
            num_blocks = len(self.encoder.blocks)
            self.lora_layers = list(range(num_blocks))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.encoder.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

        self.use_view_fusion = use_view_fusion
        if use_view_fusion:
            self.view_fusion_head = ViewFusionHead(
                emb_dim=emb_dim,
                num_views=num_views,
                n_classes=n_classes,
                fusion_type=view_fusion_type,
            )
        else:
            self.view_fusion_head = None

    def _get_pooled_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract GAP-pooled features (B*N, C) before classification head."""
        feature_list = self.encoder.get_intermediate_layers(
            x, n=self.inter_layers, reshape=True
        )
        feat = feature_list[0]  # [B*N, C, h, w]
        feat = self.pool(feat).flatten(1)  # [B*N, C]
        return feat

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, x: torch.Tensor, num_views: int = None) -> torch.Tensor:
        """
        Args:
            x: [B*N, 3, H, W] - batch of images (N views per volume)
            num_views: If set, use view fusion. Output [B, n_classes].
                      If None, output [B*N, n_classes] (per-view logits).
        """
        # View fusion: fuse features from all views -> single prediction per volume
        if self.view_fusion_head is not None and num_views is not None:
            feat = self._get_pooled_features(x)  # (B*N, C)
            B = feat.shape[0] // num_views
            feat = feat.view(B, num_views, -1)  # (B, N, C)
            return self.view_fusion_head(feat)  # (B, n_classes)

        feature_list = self.encoder.get_intermediate_layers(
            x, n=self.inter_layers, reshape=True
        )
        feat = feature_list[0]  # [B, C, h, w]
        logits = self.decoder(feat)  # [B, n_classes]
        return logits


    def save_parameters(self, filename: str) -> None:
        """Save the LoRA weights and decoder weights to a .pt file

        Args:
            filename (str): Filename of the weights
        """
        w_a, w_b = {}, {}
        if self.use_lora:
            w_a = {f"w_a_{i:03d}": self.w_a[i].weight for i in range(len(self.w_a))}
            w_b = {f"w_b_{i:03d}": self.w_b[i].weight for i in range(len(self.w_a))}

        state = {**w_a, **w_b, **self.decoder.state_dict()}
        if self.view_fusion_head is not None:
            state.update(self.view_fusion_head.state_dict())
        torch.save(state, filename)

    def load_parameters(self, filename: str) -> None:
        """Load the LoRA and decoder weights from a file

        Args:
            filename (str): File name of the weights
        """
        state_dict = torch.load(filename)

        # Load the LoRA parameters
        if self.use_lora:
            for i, w_A_linear in enumerate(self.w_a):
                saved_key = f"w_a_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_A_linear.weight = nn.Parameter(saved_tensor)

            for i, w_B_linear in enumerate(self.w_b):
                saved_key = f"w_b_{i:03d}"
                saved_tensor = state_dict[saved_key]
                w_B_linear.weight = nn.Parameter(saved_tensor)

        # Load decoder parameters
        self.decoder.load_state_dict(
            {k: v for k, v in state_dict.items() if k in self.decoder.state_dict()}
        )
        if self.view_fusion_head is not None:
            vf_state = {k: v for k, v in state_dict.items() if k in self.view_fusion_head.state_dict()}
            if vf_state:
                self.view_fusion_head.load_state_dict(vf_state)
