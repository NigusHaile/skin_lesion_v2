"""
scripts/train_skin_validator.py
================================
Train and save the ResNet50 + PCA + IsolationForest skin image validator.
Run once from the project root:

    python scripts/train_skin_validator.py

The fitted model is saved to checkpoints/skin_validator/skin_validator.pkl
and is picked up automatically by dashboard/app.py at deploy time.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG, get_device, seed_everything
from src.dataset import build_splits, get_dataloaders
from src.skin_validator import SkinValidator, VALIDATOR_DIR


def main() -> None:
    seed_everything(CFG.project.random_seed)
    device = get_device(CFG)

    # ── Build data splits (or reload if already done) ─────────────────────────
    metadata_csv = f"{CFG.paths.data_raw}/HAM10000_metadata.csv"
    train_df, val_df, test_df = build_splits(metadata_csv, CFG.paths.data_raw)

    # Use full train + val set for richer coverage of the skin distribution.
    import pandas as pd
    from torch.utils.data import DataLoader
    from src.dataset import SkinLesionDataset, get_val_transforms

    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # IMPORTANT: use val transforms (no augmentation) so features match inference-time
    # features exactly. Augmented images produce a biased centroid.
    ds = SkinLesionDataset(all_df, transform=get_val_transforms())
    dl = CFG.dataloader
    train_loader = DataLoader(
        ds,
        batch_size=dl.batch_size,
        shuffle=False,
        num_workers=dl.num_workers,
        pin_memory=dl.pin_memory,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    validator = SkinValidator(threshold_std_multiplier=3.5)
    validator.train(train_loader, device, save_dir=VALIDATOR_DIR)
    print(f"\nDone. Model saved to: {VALIDATOR_DIR}/skin_validator.pkl")


if __name__ == "__main__":
    main()
