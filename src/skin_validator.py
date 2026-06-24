"""
src/skin_validator.py
=====================
Skin dermoscopy image gatekeeper.

Pipeline:
  1. Pretrained ResNet50 (ImageNet) extracts 2048-D global-average-pool features.
  2. PCA reduces to 128-D for a compact, stable representation.
  3. IsolationForest (trained on HAM10000) scores the image as
     in-distribution (dermoscopy) or out-of-distribution (non-skin).

The fitted PCA + IsolationForest are saved together as a single pickle so
the backbone weights never need to be persisted separately — timm always
provides the same ImageNet-pretrained ResNet50.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import timm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import sys as _sys
_SRC_ROOT = Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from src.config import CFG

# ── Paths ─────────────────────────────────────────────────────────────────────
VALIDATOR_DIR  = Path(CFG.paths.checkpoints) / "skin_validator"
VALIDATOR_PATH = VALIDATOR_DIR / "skin_validator.pkl"

# ── Preprocessing (matches ImageNet-pretrained ResNet50) ──────────────────────
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=_MEAN, std=_STD),
    ToTensorV2(),
])


def _pil_to_tensor(pil_img: Image.Image) -> torch.Tensor:
    """PIL → [1, 3, 224, 224] normalised tensor."""
    arr = np.array(pil_img.convert("RGB"))
    return _transform(image=arr)["image"].unsqueeze(0)


# ── Feature extractor ─────────────────────────────────────────────────────────
class _ResNet50Extractor(nn.Module):
    """Frozen ImageNet-pretrained ResNet50 backbone — outputs 2048-D GAP features."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
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
        feats = extractor(imgs).cpu().numpy()
        parts.append(feats)
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] batches processed", end="\r")
    print()
    return np.concatenate(parts, axis=0)


# ── Main class ────────────────────────────────────────────────────────────────
class SkinValidator:
    """
    ResNet50 + PCA + IsolationForest skin dermoscopy gatekeeper.

    Training (run once on HAM10000):
        validator = SkinValidator()
        validator.train(train_loader, device, save_dir=VALIDATOR_DIR)

    Inference (deployed app):
        validator = SkinValidator.load(VALIDATOR_DIR, device)
        is_skin, score = validator.is_skin(pil_image)
    """

    def __init__(
        self,
        pca_components: int   = 128,
        n_estimators:   int   = 300,
        contamination:  float = 0.03,
    ) -> None:
        self.pca_components = pca_components
        self.n_estimators   = n_estimators
        self.contamination  = contamination

        self._extractor: _ResNet50Extractor | None = None
        self._pca:       PCA                | None = None
        self._iforest:   IsolationForest    | None = None
        self._device:    torch.device       | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        device: torch.device,
        save_dir: str | Path | None = None,
    ) -> "SkinValidator":
        self._device    = device
        self._extractor = _ResNet50Extractor().to(device)

        print("[SkinValidator] Step 1/3 — extracting ResNet50 features …")
        X = _extract_all_features(self._extractor, train_loader, device)
        print(f"[SkinValidator] Feature matrix: {X.shape}")

        print(f"[SkinValidator] Step 2/3 — PCA({self.pca_components}) …")
        self._pca = PCA(n_components=self.pca_components, random_state=CFG.project.random_seed)
        X_pca = self._pca.fit_transform(X)
        explained = self._pca.explained_variance_ratio_.sum()
        print(f"[SkinValidator] PCA explains {explained:.1%} of variance")

        print(f"[SkinValidator] Step 3/3 — IsolationForest({self.n_estimators} trees) …")
        self._iforest = IsolationForest(
            n_estimators  = self.n_estimators,
            contamination = self.contamination,
            random_state  = CFG.project.random_seed,
            n_jobs        = -1,
        )
        self._iforest.fit(X_pca)
        print("[SkinValidator] Training complete.")

        if save_dir is not None:
            self.save(save_dir)

        return self

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, save_dir: str | Path) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / "skin_validator.pkl"
        payload = {"pca": self._pca, "iforest": self._iforest}
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1e6
        print(f"[SkinValidator] Saved → {path}  ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, save_dir: str | Path, device: torch.device) -> "SkinValidator":
        path = Path(save_dir) / "skin_validator.pkl"
        with open(path, "rb") as f:
            payload = pickle.load(f)
        obj = cls()
        obj._device    = device
        obj._extractor = _ResNet50Extractor().to(device)
        obj._pca       = payload["pca"]
        obj._iforest   = payload["iforest"]
        print(f"[SkinValidator] Loaded from {path}")
        return obj

    # ── Inference ─────────────────────────────────────────────────────────────

    def is_skin(self, pil_img: Image.Image) -> tuple[bool, float]:
        """
        Args:
            pil_img: raw PIL image (any size, any mode).

        Returns:
            (is_skin, anomaly_score)
            anomaly_score: higher (less negative) = more like a dermoscopy image.
            IsolationForest.score_samples returns negative values; threshold is ~0.
        """
        tensor = _pil_to_tensor(pil_img).to(self._device)
        self._extractor.eval()
        with torch.no_grad():
            feat_2048 = self._extractor(tensor).cpu().numpy()
        feat_pca = self._pca.transform(feat_2048)
        score    = float(self._iforest.score_samples(feat_pca)[0])
        label    = int(self._iforest.predict(feat_pca)[0])   # 1=inlier, -1=outlier
        return label == 1, score
