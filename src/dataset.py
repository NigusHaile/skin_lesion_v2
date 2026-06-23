"""
src/dataset.py
==============
Data pipeline for HAM10000 dermoscopy images: transforms, Dataset, splits,
class-weight/sampler utilities, and DataLoader factory.

Every other module should import constants from here rather than redefining them:
    from src.dataset import CLASS_LABELS, CLASS_NAMES, LABEL_TO_IDX, IDX_TO_NAME
"""

import os
import sys
from pathlib import Path

# Make `src` importable regardless of how this file is invoked.
# When run directly (python src/dataset.py), Python puts src/ on sys.path and
# the `src` package itself is invisible. Inserting the project root fixes this
# without affecting normal `from src.X import Y` usage from train_all.ipynb.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CFG


def load_config(config_path: str = None):
    """Shim for dashboard compatibility — returns the global CFG object."""
    return CFG


#  Class-label constants
# Single source of truth — imported by evaluate, dashboard, gradcam, etc.

CLASS_LABELS: list[str] = CFG.data.class_labels   # short dx codes: ["nv", "mel", ...]
CLASS_NAMES:  list[str] = CFG.data.class_names    # clinical names for plots/dashboard

LABEL_TO_IDX: dict[str, int] = {lbl: i for i, lbl in enumerate(CLASS_LABELS)}
IDX_TO_LABEL: dict[int, str] = {i: lbl for lbl, i in LABEL_TO_IDX.items()}
IDX_TO_NAME:  dict[int, str] = dict(enumerate(CLASS_NAMES))

# Clinical risk used for dashboard colour-coding
RISK_LEVELS: dict[str, str] = dict(zip(CLASS_LABELS, CFG.data.risk_levels))

# Aliases used by dashboard/app.py
LABEL2IDX  = LABEL_TO_IDX
IDX2LABEL  = IDX_TO_LABEL
RISK_LEVEL = RISK_LEVELS

# Consistent colour palette across all plots (one colour per class)
CLASS_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2",
]


# Augmentation pipelines
def get_train_transforms() -> A.Compose:
    """
    Training augmentation pipeline built around clinical domain knowledge:
    - HorizontalFlip / VerticalFlip — dermoscopy has no canonical orientation
    - Rotation                      — lesion angle is diagnostically irrelevant
    - ColorJitter                   — handles lighting/device variability across hospitals
    - ElasticTransform              — simulates soft tissue deformation
    - CoarseDropout                 — forces global context; prevents texture patch over-fitting
    All probabilities and magnitudes are read from CFG.augmentation so they can be
    swept in ablation studies without touching this file.
    """
    aug = CFG.augmentation
    return A.Compose([
        A.Resize(CFG.data.image_size, CFG.data.image_size),
        A.HorizontalFlip(p=aug.horizontal_flip_p),
        A.VerticalFlip(p=aug.vertical_flip_p),
        A.Rotate(limit=aug.rotation_limit_deg, p=aug.rotation_p),
        A.ColorJitter(
            brightness=aug.brightness_limit,
            contrast=aug.contrast_limit,
            saturation=aug.saturation_limit,
            hue=aug.hue_limit,
            p=aug.color_jitter_p,
        ),
        A.ElasticTransform(alpha=aug.elastic_alpha, sigma=aug.elastic_sigma, p=aug.elastic_p),
        A.CoarseDropout(
            num_holes_range=(1, aug.cutout_holes),
            hole_height_range=(1, aug.cutout_hole_height),
            hole_width_range=(1, aug.cutout_hole_width),
            fill=0,
            p=aug.cutout_p,
        ),
        A.Normalize(mean=CFG.data.imagenet_mean, std=CFG.data.imagenet_std),
        ToTensorV2(),  # HWC uint8 → CHW float32, required for pretrained backbones
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test pipeline — resize + ImageNet normalisation only, no augmentation."""
    return A.Compose([
        A.Resize(CFG.data.image_size, CFG.data.image_size),
        A.Normalize(mean=CFG.data.imagenet_mean, std=CFG.data.imagenet_std),
        ToTensorV2(),
    ])


# Dataset

class SkinLesionDataset(Dataset):
    
  #  PyTorch Dataset for HAM10000 dermoscopy images.
    

    def __init__(
        self,
        dataframe:   pd.DataFrame,
        transform:   A.Compose = None,
        return_path: bool = False,
    ) -> None:
        self.df          = dataframe.reset_index(drop=True)
        self.transform   = transform
        self.return_path = return_path
        # Convert string labels → ints once here, not on every __getitem__ call
        self.labels = [LABEL_TO_IDX[dx] for dx in self.df["dx"]]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        label = self.labels[idx]
        # Albumentations expects uint8 HWC numpy; PIL handles EXIF rotation via convert
        image = np.array(Image.open(row["image_path"]).convert("RGB"))
        if self.transform:
            image = self.transform(image=image)["image"]
        if self.return_path:
            return image, label, str(row["image_path"])
        return image, label


#  Split builder

def _find_image_path(image_id: str, image_dir: Path) -> str | None:
    """
    Search both HAM10000 image subdirectories for a given image_id.
    Returns the first match or None if the file is missing.
    HAM10000 ships its images split across two folders; the empty-string fallback
    handles flat layouts used in some Kaggle repacks.
    """
    for folder in ("HAM10000_images_part_1", "HAM10000_images_part_2", ""):
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = image_dir / folder / (image_id + ext)
            if candidate.exists():
                return str(candidate)
    return None


def _log_class_distribution(df: pd.DataFrame, split_name: str) -> None:
    """Print per-class counts and percentages for one split."""
    counts = df["dx"].value_counts()
    print(f"\n  {split_name} distribution:")
    for label in CLASS_LABELS:
        n   = counts.get(label, 0)
        pct = 100 * n / len(df)
        print(f"    {label:6s}: {n:5d}  ({pct:.1f}%)")


def build_splits(metadata_csv: str, image_dir: str) -> tuple[pd.DataFrame, ...]:
    """
    Build stratified 70/15/15 train/val/test splits and persist them as CSVs.

    Stratification on `dx` preserves class distribution in every split —
    essential given the severe imbalance (melanocytic nevi ≈ 67% of samples).
    Saved CSVs let downstream scripts reload splits without re-splitting,
    keeping evaluation fully reproducible.

    Returns:
        (train_df, val_df, test_df)
    """
    os.makedirs(CFG.paths.data_splits, exist_ok=True)
    seed       = CFG.project.random_seed
    image_root = Path(image_dir)

    df = pd.read_csv(metadata_csv)
    df["image_path"] = df["image_id"].apply(lambda img_id: _find_image_path(img_id, image_root))

    n_missing = df["image_path"].isna().sum()
    if n_missing:
        print(f"[dataset] Warning: {n_missing} images not found — dropping")
    df = df.dropna(subset=["image_path"])
    df = df[df["dx"].isin(CLASS_LABELS)].reset_index(drop=True)
    print(f"[dataset] Total valid images: {len(df)}")

    # Two-step stratified split: train | (val + test), then val | test
    val_test_ratio  = 1.0 - CFG.data.train_ratio
    test_of_holdout = 1.0 - CFG.data.val_ratio / val_test_ratio

    train_df, holdout_df = train_test_split(
        df, test_size=val_test_ratio, stratify=df["dx"], random_state=seed,
    )
    val_df, test_df = train_test_split(
        holdout_df, test_size=test_of_holdout, stratify=holdout_df["dx"], random_state=seed,
    )

    for name, split in (("train", train_df), ("val", val_df), ("test", test_df)):
        split.to_csv(f"{CFG.paths.data_splits}/{name}.csv", index=False)
        _log_class_distribution(split, name.capitalize())

    print(f"\n[dataset] Split sizes — Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")
    return train_df, val_df, test_df


# Class-weight / sampler utilities 

def _class_counts(df: pd.DataFrame) -> np.ndarray:
    """Return per-class sample counts as a float32 array aligned to CLASS_LABELS."""
    counts = np.zeros(CFG.data.num_classes, dtype=np.float32)
    for label in df["dx"]:
        counts[LABEL_TO_IDX[label]] += 1
    return counts


def compute_class_weights(train_df: pd.DataFrame) -> torch.Tensor:
    """
    Compute inverse-frequency weights for CrossEntropyLoss.

    Formula: w_c = N / (C × n_c)
    Upweights rare classes (df, vasc) and downweights the dominant class (nv),
    so the loss gradient is not dominated by easy majority-class examples.
    """
    counts  = _class_counts(train_df)
    weights = len(train_df) / (CFG.data.num_classes * counts)

    print("[dataset] Class weights (inverse frequency):")
    for label, n, w in zip(CLASS_LABELS, counts, weights):
        print(f"  {label:6s}: n={int(n):4d}  w={w:.3f}")

    return torch.tensor(weights, dtype=torch.float32)


def get_weighted_sampler(train_df: pd.DataFrame) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler so each batch is approximately class-balanced.
    Works alongside the weighted loss — the sampler balances batch composition
    while the loss weights correct the remaining gradient imbalance.
    """
    counts         = _class_counts(train_df)
    class_weights  = 1.0 / counts
    sample_weights = np.array([class_weights[LABEL_TO_IDX[dx]] for dx in train_df["dx"]])
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )


#  DataLoader factory 
def get_dataloaders(
    train_df: pd.DataFrame,
    val_df:   pd.DataFrame,
    test_df:  pd.DataFrame,
    use_weighted_sampler: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return (train_loader, val_loader, test_loader).

    Train loader uses augmentation and — when use_weighted_sampler=True — a
    WeightedRandomSampler to counteract class imbalance during mini-batch construction.
    Val/test loaders use only resize + normalisation; shuffle is always False so
    evaluation order is deterministic.
    """
    dl = CFG.dataloader
    # Shared kwargs to avoid repeating the same three arguments three times
    _common = dict(num_workers=dl.num_workers, pin_memory=dl.pin_memory)

    train_ds = SkinLesionDataset(train_df, transform=get_train_transforms())
    val_ds   = SkinLesionDataset(val_df,   transform=get_val_transforms())
    test_ds  = SkinLesionDataset(test_df,  transform=get_val_transforms())

    if use_weighted_sampler:
        train_loader = DataLoader(
            train_ds, batch_size=dl.batch_size,
            sampler=get_weighted_sampler(train_df),
            drop_last=True, **_common,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=dl.batch_size,
            shuffle=True, drop_last=True, **_common,
        )

    val_loader  = DataLoader(val_ds,  batch_size=dl.batch_size, shuffle=False, **_common)
    test_loader = DataLoader(test_ds, batch_size=dl.batch_size, shuffle=False, **_common)

    print(f"[dataset] Batches — Train: {len(train_loader)}  "
          f"Val: {len(val_loader)}  Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader