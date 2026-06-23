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
    Architecture: Conv2d(k=2,same) → ReLU → MaxPool(2), ×4,
    then Dropout(0.3) → Flatten → FC(150) → ReLU → Dropout(0.4) → FC(num_classes).
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes
        img_size    = CFG.data.image_size   # 224

        self.block1 = nn.Sequential(
            nn.Conv2d(3,   16,  kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16,  32,  kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32,  64,  kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64,  128, kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 4 × MaxPool(2) on 224×224 → 14×14; flat = 128 × 14 × 14 = 25,088
        flat_size = 128 * (img_size // 16) ** 2

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(flat_size, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(150, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.classifier(x)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened conv features before the classifier."""
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.flatten(1)

    def get_gradcam_layer(self) -> nn.Module:
        """Return the last conv layer in block4 for GradCAM hooks."""
        return self.block4[0]


class SimpleCNNShallow(nn.Module):
    """
    2-block variant of SimpleCNN used for the depth ablation study.
    """

    def __init__(self) -> None:
        super().__init__()
        num_classes = CFG.data.num_classes
        img_size    = CFG.data.image_size

        self.block1 = nn.Sequential(
            nn.Conv2d(3,  16, kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=2, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        # 2 × MaxPool(2) on 224×224 → 56×56; flat = 32 × 56 × 56 = 100,352
        flat_size = 32 * (img_size // 4) ** 2

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Flatten(),
            nn.Linear(flat_size, 150),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(150, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)


# ResNet50 strong pretrained baseline


class ResNet50(nn.Module):
    """
    ResNet50 pretrained on ImageNet with standard fine-tuning.
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
        model_name: one of "simple_cnn", "resnet50", "vit"
        device:     torch.device

    Returns:
        model on the correct device
    """
    name_map = {
        "simple_cnn":         SimpleCNN,
        "simple_cnn_shallow": SimpleCNNShallow,
        "resnet50":           ResNet50,
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
ViTLoRAClassifier  = ViTWithLoRA
ResNet50Classifier = ResNet50