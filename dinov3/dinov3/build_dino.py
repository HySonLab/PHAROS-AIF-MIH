"""
Build DINOv3 models with LoRA for medical image classification.
"""
import logging
import os
import torch
import torch.nn as nn

from .dino import DINOEncoderLoRA

def build_dinov3(
    dino_size: str = "base",
    num_classes: int = 3,
    img_dim: tuple[int, int] = (512, 512),
    use_lora: bool = True,
    lora_rank: int = 3,
    use_view_fusion: bool = False,
    num_views: int = 16,
    view_fusion_type: str = "concat",
) -> nn.Module:
    """
    Build a DINOv3 model with LoRA for classification.

    Args:
        dino_size: Model size - 'base', 'large', or 'huge'
        num_classes: Number of output classes
        img_dim: Image dimensions (height, width) - must be divisible by 16
        use_lora: Whether to use LoRA adaptation
        lora_rank: LoRA rank parameter

    Returns:
        DINOEncoderLoRA model
    """
    # Note: We'll validate img_dim after loading encoder to get actual patch_size
    
    # Model configurations - using DINOv3 models (official names from GitHub)
    backbones = {
        "base": "dinov3_vitb16",      # ViT-B/16 - best balance for medical images
        "large": "dinov3_vitl16",     # ViT-L/16 - better performance
        "huge": "dinov3_vith16plus",  # ViT-H/16+ - largest, best for classification
    }
    
    # Pretrained checkpoint filenames (matching backbone names)
    checkpoint_files = {
        "base": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "large": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "huge": "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    }
    
    if dino_size not in backbones:
        raise ValueError(f"Invalid dino_size: {dino_size}. Must be one of {list(backbones.keys())}")
    
    # Load from local pretrained folder only (no fallback to torch.hub)
    pretrained_dir = "./pretrained"
    checkpoint_path = os.path.join(pretrained_dir, checkpoint_files[dino_size])
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}\n"
            f"Please download {checkpoint_files[dino_size]} and place it in {pretrained_dir}/\n"
            f"Download URL: https://dinov3.llamameta.net/dinov3_vitb16/{checkpoint_files[dino_size]}"
        )
    
    logging.info(f"Loading {backbones[dino_size]} from local checkpoint: {checkpoint_path}")
    
    # Step 1: Load model structure from local dinov3.models
    logging.info("Loading DINOv3 model structure from local code...")
    from dinov3.models.vision_transformer import DinoVisionTransformer
    
    # Model configurations for local instantiation
    # Note: DINOv3 checkpoints use patch_size=16 (not 14)
    model_configs = {
        "base": {
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "ffn_ratio": 4,
            "patch_size": 16,
        },
        "large": {
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "ffn_ratio": 4,
            "patch_size": 16,
        },
        "huge": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 20,
            "ffn_ratio": 4,
            "patch_size": 16,
        }
    }
    
    if dino_size not in model_configs:
        raise ValueError(f"Unknown model size: {dino_size}")
    
    config = model_configs[dino_size]
    encoder = DinoVisionTransformer(
        patch_size=config["patch_size"],
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        ffn_ratio=config["ffn_ratio"],
        # Additional parameters needed to match checkpoint
        qkv_bias=True,
        layerscale_init=1.0,  # LayerScale initialization
        norm_layer="layernorm",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,  
        mask_k_bias=True,     
        untie_cls_and_patch_norms=False,
        untie_global_and_local_cls_norm=False,
    )
    logging.info("Model structure created successfully")
    
    # Step 2: Load pretrained weights from local file
    logging.info(f"Loading weights from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    # Remove 'module.' prefix if present (from DataParallel)
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    encoder.load_state_dict(state_dict, strict=True)
    
    logging.info(f"Successfully loaded {backbones[dino_size]} from local checkpoint")
    
    # Get embedding dimension and patch size
    try:
        emb_dim = encoder.num_features  # DINOv3 uses num_features
    except AttributeError:
        emb_dim = encoder.blocks[0].attn.qkv.in_features  # Fallback
    
    # Try to get patch_size from encoder
    try:
        patch_size = encoder.patch_size
    except AttributeError:
        # Try to get from patch_embed
        try:
            patch_size = encoder.patch_embed.patch_size[0] if hasattr(encoder.patch_embed, 'patch_size') else 16
        except:
            patch_size = 16  # DINOv3 uses patch_size=16
            logging.warning(f"Could not determine patch_size from encoder, using default: {patch_size}")
    
    logging.info(f"Encoder embedding dimension: {emb_dim}, patch_size: {patch_size}")
    logging.info(f"Image dimensions: {img_dim}")
    
    # Auto-adjust image dimensions to match patch size if needed
    if img_dim[0] % patch_size != 0 or img_dim[1] % patch_size != 0:
        # Round to nearest multiple of patch_size
        h_adjusted = (img_dim[0] // patch_size) * patch_size
        w_adjusted = (img_dim[1] // patch_size) * patch_size
        logging.warning(
            f"Image dimensions {img_dim} adjusted to {h_adjusted}x{w_adjusted} to be divisible by patch_size={patch_size}"
        )
        img_dim = (h_adjusted, w_adjusted)
    
    # Create DINOEncoderLoRA model
    model = DINOEncoderLoRA(
        encoder=encoder,
        r=lora_rank,
        emb_dim=emb_dim,
        n_classes=num_classes,
        use_lora=use_lora,
        img_dim=img_dim,
        use_view_fusion=use_view_fusion,
        num_views=num_views,
        view_fusion_type=view_fusion_type,
    )
    return model
