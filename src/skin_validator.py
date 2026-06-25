"""
src/skin_validator.py
=====================
Skin dermoscopy image gatekeeper — validator class + training entry-point.

Pipeline:
  1. Fine-tuned ResNet50 (HAM10000) extracts 2048-D GAP features.
     Using the fine-tuned backbone is critical — its later layers are specialised
     for skin-lesion textures, so non-dermoscopy images land far from the
     training cluster in feature space.
  2. L2 distance from the training-set centroid: if the image falls outside the
     distribution radius (mean + 3.5 × std, covering ~99.9 % of training images),
     it is rejected as non-dermoscopy.

The centroid + threshold are saved as a small numpy pickle (~16 KB).
The ResNet50 backbone comes from the existing fine-tuned checkpoint
(checkpoints/resnet50/best.pth), so no extra weights are stored.

Run training once from the project root:

    python src/skin_validator.py

The fitted model is saved to checkpoints/skin_validator/skin_validator.pkl
and is picked up automatically by dashboard/app.py at deploy time.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import CFG, get_device, seed_everything

# ── Paths ─────────────────────────────────────────────────────────────────────
VALIDATOR_DIR  = Path(CFG.paths.checkpoints) / "skin_validator"
VALIDATOR_PATH = VALIDATOR_DIR / "skin_validator.pkl"
_RESNET50_CKPT = Path(CFG.paths.checkpoints) / "resnet50" / "best.pth"

# ── Preprocessing ─────────────────────────────────────────────────────────────
_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=CFG.data.imagenet_mean, std=CFG.data.imagenet_std),
    ToTensorV2(),
])


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    arr = np.array(pil_img.convert("RGB"))
    return _transform(image=arr)["image"].unsqueeze(0)


# ── Feature extractor ─────────────────────────────────────────────────────────
class _FineTunedResNet50Extractor(nn.Module):
    """
    ResNet50 backbone from the fine-tuned HAM10000 checkpoint.
    State dict keys start with 'backbone.' — the classifier head is discarded.
    Falls back to ImageNet-pretrained if the checkpoint is missing.
    """

    def __init__(self, ckpt_path: Path = _RESNET50_CKPT) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "resnet50", pretrained=False, num_classes=0, global_pool="avg"
        )
        if ckpt_path.exists():
            sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            backbone_sd = {
                k[len("backbone."):]: v
                for k, v in sd.items()
                if k.startswith("backbone.")
            }
            self.backbone.load_state_dict(backbone_sd, strict=True)
            print(f"[SkinValidator] Fine-tuned backbone loaded from {ckpt_path}")
        else:
            self.backbone = timm.create_model(
                "resnet50", pretrained=True, num_classes=0, global_pool="avg"
            )
            print("[SkinValidator] WARNING: fine-tuned checkpoint not found — "
                  "using ImageNet-pretrained ResNet50")
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)   # [B, 2048]


@torch.no_grad()
def _extract_all_features(
    extractor: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    extractor.eval()
    parts: list[np.ndarray] = []
    total = len(loader)
    for i, batch in enumerate(loader, 1):
        imgs = batch[0].to(device)
        parts.append(extractor(imgs).cpu().numpy())
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] batches processed", end="\r")
    print()
    return np.concatenate(parts, axis=0)


# ── Main class ────────────────────────────────────────────────────────────────
class SkinValidator:
    """
    Fine-tuned ResNet50 centroid-distance skin validator.

    Training (run once after ResNet50 fine-tuning):
        validator = SkinValidator()
        validator.train(train_loader, device, save_dir=VALIDATOR_DIR)

    Inference:
        validator = SkinValidator.load(VALIDATOR_DIR, device)
        is_skin, dist = validator.is_skin(pil_image)
    """

    def __init__(self, threshold_std_multiplier: float = 3.5) -> None:
        self.threshold_std_multiplier = threshold_std_multiplier
        self._extractor: _FineTunedResNet50Extractor | None = None
        self._centroid:  np.ndarray                  | None = None
        self._threshold: float                       | None = None
        self._device:    torch.device                | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        device: torch.device,
        save_dir: str | Path | None = None,
    ) -> "SkinValidator":
        self._device    = device
        self._extractor = _FineTunedResNet50Extractor().to(device)

        print("[SkinValidator] Extracting fine-tuned ResNet50 features …")
        X = _extract_all_features(self._extractor, train_loader, device)
        print(f"[SkinValidator] Feature matrix: {X.shape}")

        self._centroid  = X.mean(axis=0)
        dists           = np.linalg.norm(X - self._centroid, axis=1)
        self._threshold = float(dists.mean() + self.threshold_std_multiplier * dists.std())

        print(f"[SkinValidator] Centroid distance — "
              f"mean={dists.mean():.3f}  std={dists.std():.3f}  "
              f"p99={np.percentile(dists, 99):.3f}  max={dists.max():.3f}")
        print(f"[SkinValidator] Rejection threshold: {self._threshold:.3f} "
              f"(mean + {self.threshold_std_multiplier}×std)")

        if save_dir is not None:
            self.save(save_dir)
        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "skin_validator.pkl"
        payload = {"centroid": self._centroid, "threshold": self._threshold}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)
        size_kb = path.stat().st_size / 1e3
        print(f"[SkinValidator] Saved → {path}  ({size_kb:.0f} KB)")

    @classmethod
    def load(cls, save_dir: str | Path, device: torch.device) -> "SkinValidator":
        path = Path(save_dir) / "skin_validator.pkl"
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj._device    = device
        obj._extractor = _FineTunedResNet50Extractor().to(device)
        obj._centroid  = payload["centroid"]
        obj._threshold = payload["threshold"]
        print(f"[SkinValidator] Loaded — threshold={obj._threshold:.3f}")
        return obj

    # ── Inference ─────────────────────────────────────────────────────────────

    def is_skin(self, pil_img: Image.Image) -> tuple[bool, float]:
        """
        Returns (is_skin, distance_from_centroid).
        Images with distance > threshold are rejected as non-dermoscopy.
        """
        tensor = _pil_to_tensor(pil_img).to(self._device)
        self._extractor.eval()
        with torch.no_grad():
            feat = self._extractor(tensor).cpu().numpy()[0]
        dist = float(np.linalg.norm(feat - self._centroid))
        return dist <= self._threshold, dist


# ── Training entry-point ──────────────────────────────────────────────────────
def main() -> None:
    from src.dataset import build_splits, SkinLesionDataset, get_val_transforms

    seed_everything(CFG.project.random_seed)
    device = get_device(CFG)

    metadata_csv = f"{CFG.paths.data_raw}/HAM10000_metadata.csv"
    train_df, val_df, test_df = build_splits(metadata_csv, CFG.paths.data_raw)

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Use val transforms (no augmentation) so features match inference-time exactly.
    ds = SkinLesionDataset(all_df, transform=get_val_transforms())
    dl = CFG.dataloader
    loader = DataLoader(
        ds,
        batch_size=dl.batch_size,
        shuffle=False,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
    )

    validator = SkinValidator(threshold_std_multiplier=3.5)
    validator.train(loader, device, save_dir=VALIDATOR_DIR)
    print(f"\nDone. Model saved to: {VALIDATOR_DIR}/skin_validator.pkl")


if __name__ == "__main__":
    main()
