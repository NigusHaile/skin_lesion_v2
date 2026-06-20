import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import sys as _sys
from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from src.config import CFG


# SimpleCNN  — from-scratch baseline

class SimpleCNN(nn.Module):
    """
    4-block CNN trained entirely from scratch.
    Purpose: establishes the performance floor for the comparison table.

    Architecture: Conv → BN → ReLU → MaxPool (×4) → GlobalAvgPool → FC
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes

        # Each block halves the spatial resolution
        self.block1 = self._conv_block(3,   32)   # 224→112
        self.block2 = self._conv_block(32,  64)   # 112→56
        self.block3 = self._conv_block(64,  128)  # 56→28
        self.block4 = self._conv_block(128, 256)  # 28→14

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Two conv layers + BN + ReLU + MaxPool."""
        return nn.Sequential(
            nn.Conv2d(in_channels,  out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        features = self.global_pool(x).flatten(1)   # (B, 256)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 256-dim embedding (before classifier)."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.global_pool(x).flatten(1)

    def get_gradcam_layer(self) -> nn.Module:
        """Return the last conv layer in block4 for GradCAM hooks."""
        return self.block4[3]


class SimpleCNNShallow(nn.Module):
    """
    2-block variant of SimpleCNN used for the depth ablation study.
    Keeps everything identical to SimpleCNN except only 2 conv blocks.
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes

        self.block1 = SimpleCNN._conv_block(3,  32)   # 224→112
        self.block2 = SimpleCNN._conv_block(32, 64)   # 112→56

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        features = self.global_pool(x).flatten(1)
        return self.classifier(features)


# ResNet50 strong pretrained baseline


class ResNet50(nn.Module):
    """
    ResNet50 pretrained on ImageNet with standard fine-tuning.
    Represents a strong baseline against which EfficientNet is compared.
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes
        dropout = CFG.resnet50.dropout

        # Load backbone without classification head (num_classes=0, global_pool="avg")
        self.backbone = timm.create_model(
            "resnet50",
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features  # 2048

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # (B, 2048)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 2048-dim backbone embedding."""
        return self.backbone(x)

    def get_gradcam_layer(self) -> nn.Module:
        """Return the last residual block group for GradCAM hooks."""
        return self.backbone.layer4


# EfficientNetB3  — primary model with 3-stage transfer learning


class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 pretrained on ImageNet with 3-stage progressive fine-tuning.

    Why EfficientNet-B3?
    - Compound scaling: jointly scales depth + width + resolution
    - 12M parameters: optimal for 10k-image datasets
    - 81.6% top-1 ImageNet accuracy
    - Fits in 12 GB VRAM at batch=32

    3-Stage fine-tuning strategy:
    Stage 1: Freeze backbone → train head only (LR=1e-3, 5 epochs)
             Rationale: head learns domain features from random init quickly
    Stage 2: Unfreeze last 2 blocks (LR=1e-4, 15 epochs)
             Rationale: deep features adapt gradually to dermoscopy
    Stage 3: Unfreeze all (LR=5e-5, 20 epochs)
             Rationale: full fine-tune with small LR avoids catastrophic forgetting
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes
        dropout = CFG.efficientnet.dropout

        # Backbone without head
        self.backbone = timm.create_model(
            CFG.efficientnet.backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        feat_dim = self.backbone.num_features   # 1536 for B3

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # Start with frozen backbone (Stage 1)
        self.freeze_backbone()

    # Freeze / unfreeze methods

    def freeze_backbone(self) -> None:
        """Stage 1: freeze all backbone params, only head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] EfficientNet: backbone FROZEN — trainable params: {n_trainable:,}")

    def unfreeze_top_blocks(self, n_blocks: int = 2) -> None:
        """Stage 2: unfreeze last n blocks of the EfficientNet backbone."""
        blocks = list(self.backbone.blocks.children())
        for block in blocks[-n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True
        # Also unfreeze the final conv head
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] EfficientNet: top {n_blocks} blocks UNFROZEN — "
              f"trainable params: {n_trainable:,}")

    def unfreeze_all(self) -> None:
        """Stage 3: unfreeze entire network for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] EfficientNet: FULLY UNFROZEN — trainable params: {n_trainable:,}")

    # Forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 1536)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1536-dim embedding (before classifier)."""
        return self.backbone(x)

    def get_gradcam_layer(self) -> nn.Module:
        """Return the final conv layer for GradCAM hooks."""
        return self.backbone.conv_head



# LoRALinear  — Low-Rank Adaptation wrapper

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear with a trainable low-rank delta.

    Original LoRA paper (Hu et al., 2021):
        W_effective = W_frozen + B @ A * (alpha / rank)

    where:
        A ∈ R^{rank × d_in}   — init: Kaiming uniform
        B ∈ R^{d_out × rank}  — init: zeros  → delta = 0 at start

    Why LoRA for ViT?
    - ViT-B/16 has 86M parameters — full fine-tuning overfits on 10k images
    - LoRA rank=4 trains only ~0.5% of params
    - Strong regularisation: model stays close to ImageNet prior
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int   = 4,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        d_out, d_in = original_linear.weight.shape
        self.rank   = rank
        self.scale  = alpha / rank    # scaling factor

        # Frozen original weights
        self.weight = original_linear.weight   # kept frozen
        self.bias   = original_linear.bias     # kept frozen

        # Trainable low-rank matrices
        self.lora_A = nn.Parameter(torch.empty(rank, d_in))
        self.lora_B = nn.Parameter(torch.zeros(d_out, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Frozen path
        base_output = F.linear(x, self.weight, self.bias)
        # LoRA delta path: x @ A^T @ B^T * scale
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_output + lora_output * self.scale


# ViTWithLoRA  — Vision Transformer comparison model

class ViTWithLoRA(nn.Module):
    """
    ViT-B/16 pretrained on ImageNet21k with LoRA adapters.

    Architecture:
    - 12 transformer blocks, 12 attention heads, 768-dim embeddings
    - 196 patches (16×16) for 224×224 input
    - [CLS] token used for classification

    LoRA injection: replaces the fused QKV projection in each attention block.
    This injects trainable low-rank matrices while keeping all other weights frozen.
    """

    def __init__(self, lora_rank: int = None) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes
        vit_cfg = CFG.vit

        # Load pretrained ViT backbone (no head)
        self.backbone = timm.create_model(
            vit_cfg.backbone,
            pretrained=True,
            num_classes=0,
        )
        feat_dim = self.backbone.num_features   # 768

        # Freeze all backbone parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Inject LoRA into QKV projections of all 12 attention blocks
        rank = lora_rank if lora_rank is not None else vit_cfg.lora_rank
        n_injected = self._inject_lora(
            rank=rank,
            alpha=vit_cfg.lora_alpha,
            dropout=vit_cfg.lora_dropout,
        )

        # Classification head (fully trainable)
        self.classifier = nn.Sequential(
            nn.Dropout(vit_cfg.dropout),
            nn.Linear(feat_dim, num_classes),
        )

        # Print trainable parameter stats
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[model] ViT+LoRA: injected into {n_injected} blocks | "
              f"trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    def _inject_lora(self, rank: int, alpha: float, dropout: float) -> int:
        """Replace QKV linear layers with LoRALinear wrappers."""
        count = 0
        for block in self.backbone.blocks:
            attn = block.attn
            if hasattr(attn, "qkv"):
                # Replace fused QKV with LoRA-wrapped version
                lora_qkv = LoRALinear(attn.qkv, rank=rank, alpha=alpha, dropout=dropout)
                # Mark LoRA params as trainable
                lora_qkv.lora_A.requires_grad = True
                lora_qkv.lora_B.requires_grad = True
                attn.qkv = lora_qkv
                count += 1
        return count

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)          # (B, 768)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return 768-dim [CLS] token embedding."""
        return self.backbone(x)



# Model factory

def build_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Instantiate and move model to device.

    Args:
        model_name: one of "simple_cnn", "resnet50", "efficientnet", "vit"
        device:     torch.device

    Returns:
        model on the correct device
    """
    name_map = {
        "simple_cnn":         SimpleCNN,
        "simple_cnn_shallow": SimpleCNNShallow,
        "resnet50":           ResNet50,
        "efficientnet":       EfficientNetB3,
        "vit":                ViTWithLoRA,
    }

    if model_name not in name_map:
        raise ValueError(f"Unknown model '{model_name}'. "
                         f"Choose from: {list(name_map.keys())}")

    model = name_map[model_name]().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] Built '{model_name}' | Total params: {total_params:,} | Device: {device}")
    return model


# Aliases for dashboard/app.py compatibility
EfficientNetClassifier = EfficientNetB3
ViTLoRAClassifier      = ViTWithLoRA
ResNet50Classifier     = ResNet50