"""
src/ablations.py
================
Three ablation studies to quantify the impact of individual design choices.
Rubric extras: "Ablation study: impact of architecture, augmentation, loss" (+4.0 pt)

Study 1: Without augmentation vs With augmentation
Study 2: Without class weighting vs Weighted CrossEntropy loss
Study 3: Frozen backbone vs Full fine-tuning

Each study trains two EfficientNet-B3 models with exactly one variable changed.
Results are saved as JSON and comparison plots.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, f1_score

import sys
from pathlib import Path as _P
_ROOT = _P(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import CFG
from src.dataset import (
    SkinLesionDataset, get_train_transforms, get_val_transforms,
    compute_class_weights, get_weighted_sampler, CLASS_LABELS,
)
from src.models import EfficientNetB3, SimpleCNN, SimpleCNNShallow, ResNet50, ViTWithLoRA
from src.train import run_train_epoch, run_val_epoch, EarlyStopping


# Number of epochs per ablation run (shorter than full training to save time)
ABLATION_EPOCHS = 10
ABLATION_PATIENCE = 5


# Shared: train one ablation configuration

def _train_ablation_config(
    train_df,
    val_df,
    device:               torch.device,
    use_augmentation:     bool,
    use_class_weights:    bool,
    freeze_backbone:      bool,
    run_tag:              str,
    save_dir:             str,
) -> dict:
    """
    Train EfficientNet-B3 with the given configuration and return results.
    All settings except the variable under study are held constant.
    """
    #  Dataset & DataLoader 
    train_transform = get_train_transforms() if use_augmentation else get_val_transforms()
    val_transform   = get_val_transforms()

    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset   = SkinLesionDataset(val_df,   transform=val_transform)

    if use_class_weights:
        sampler = get_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset, batch_size=32, sampler=sampler,
            num_workers=2, pin_memory=False, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True,
        )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Model 
    model = EfficientNetB3().to(device)

    if freeze_backbone:
        model.freeze_backbone()      # only head trains
    else:
        model.unfreeze_all()         # full fine-tuning

    #  Loss function 
    if use_class_weights:
        weights   = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()   # uniform weights

    # Optimizer & scheduler 
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=ABLATION_EPOCHS, eta_min=1e-7)

    scaler = (torch.amp.GradScaler("cuda")
              if torch.cuda.is_available() else None)

    early_stop = EarlyStopping(
        patience=ABLATION_PATIENCE,
        checkpoint_path=f"{save_dir}/{run_tag}_best.pth",
    )

    #  Training loop 
    history = {"train_bacc": [], "val_bacc": [], "val_f1": []}

    print(f"\n  [{run_tag}] aug={use_augmentation} | "
          f"class_weights={use_class_weights} | freeze={freeze_backbone}")

    for epoch in range(1, ABLATION_EPOCHS + 1):
        train_m = run_train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_m   = run_val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_bacc"].append(train_m["balanced_accuracy"])
        history["val_bacc"].append(val_m["balanced_accuracy"])
        history["val_f1"].append(val_m["f1_macro"])

        print(f"    Ep {epoch:02d}/{ABLATION_EPOCHS}  "
              f"train_bacc={train_m['balanced_accuracy']:.4f}  "
              f"val_bacc={val_m['balanced_accuracy']:.4f}")

        if early_stop.step(val_m["balanced_accuracy"], model):
            print(f"    Early stopping at epoch {epoch}")
            break

    return {
        "tag":           run_tag,
        "best_val_bacc": float(early_stop.best_score or max(history["val_bacc"])),
        "history":       history,
    }


# Study 1: Augmentation effect

def study_augmentation(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """Compare no augmentation vs full albumentations pipeline."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 1 — Data Augmentation")
    print("=" * 60)

    no_aug   = _train_ablation_config(train_df, val_df, device,
                                       use_augmentation=False,
                                       use_class_weights=True,
                                       freeze_backbone=False,
                                       run_tag="aug_off", save_dir=save_dir)

    with_aug = _train_ablation_config(train_df, val_df, device,
                                       use_augmentation=True,
                                       use_class_weights=True,
                                       freeze_backbone=False,
                                       run_tag="aug_on", save_dir=save_dir)

    results = {"no_augmentation": no_aug, "with_augmentation": with_aug}
    _save_ablation_results(
        results, save_dir,
        filename="study1_augmentation",
        title="Study 1: Effect of Data Augmentation",
        label_a="No augmentation",
        label_b="With augmentation",
    )
    return results


#  Study 2: Class weighting effect

def study_class_weights(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """Compare uniform CrossEntropy vs weighted CrossEntropy loss."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 2 — Class Weighting")
    print("=" * 60)

    no_weights   = _train_ablation_config(train_df, val_df, device,
                                           use_augmentation=True,
                                           use_class_weights=False,
                                           freeze_backbone=False,
                                           run_tag="weights_off", save_dir=save_dir)

    with_weights = _train_ablation_config(train_df, val_df, device,
                                           use_augmentation=True,
                                           use_class_weights=True,
                                           freeze_backbone=False,
                                           run_tag="weights_on", save_dir=save_dir)

    results = {"no_class_weights": no_weights, "with_class_weights": with_weights}
    _save_ablation_results(
        results, save_dir,
        filename="study2_class_weights",
        title="Study 2: Effect of Class Weighting",
        label_a="Uniform loss",
        label_b="Weighted loss",
    )
    return results


# Study 3: Backbone freezing effect

def study_backbone_freezing(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """Compare training head-only vs full fine-tuning."""
    print("\n" + "=" * 60)
    print("ABLATION STUDY 3 — Backbone Freezing")
    print("=" * 60)

    frozen_backbone = _train_ablation_config(train_df, val_df, device,
                                              use_augmentation=True,
                                              use_class_weights=True,
                                              freeze_backbone=True,
                                              run_tag="frozen", save_dir=save_dir)

    full_finetuning = _train_ablation_config(train_df, val_df, device,
                                              use_augmentation=True,
                                              use_class_weights=True,
                                              freeze_backbone=False,
                                              run_tag="full_ft", save_dir=save_dir)

    results = {"frozen_backbone": frozen_backbone, "full_finetuning": full_finetuning}
    _save_ablation_results(
        results, save_dir,
        filename="study3_backbone",
        title="Study 3: Effect of Backbone Fine-tuning",
        label_a="Frozen backbone",
        label_b="Full fine-tuning",
    )
    return results


# Save results + plot

def _save_ablation_results(
    results: dict,
    save_dir: str,
    filename: str,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    """Save JSON and comparison learning curve plot."""
    os.makedirs(save_dir, exist_ok=True)

    # Serialisable version for JSON
    serialisable = {}
    for key, val in results.items():
        serialisable[key] = {
            "tag":           val["tag"],
            "best_val_bacc": float(val["best_val_bacc"]),
            "history": {
                k: [float(v) for v in lst]
                for k, lst in val["history"].items()
            },
        }
    with open(f"{save_dir}/{filename}.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    # Comparison plot
    keys = list(results.keys())
    result_a = results[keys[0]]
    result_b = results[keys[1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Learning curves
    axes[0].plot(result_a["history"]["val_bacc"],
                 label=label_a, color="#e74c3c", lw=2)
    axes[0].plot(result_b["history"]["val_bacc"],
                 label=label_b, color="#2ecc71", lw=2)
    axes[0].set_title(f"Val Balanced Accuracy\n{title}", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Balanced Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Summary bar chart
    best_scores = [result_a["best_val_bacc"], result_b["best_val_bacc"]]
    bars = axes[1].bar(
        [label_a, label_b], best_scores,
        color=["#e74c3c", "#2ecc71"], width=0.5,
    )
    axes[1].set_ylim([0, 1])
    axes[1].set_title(f"Best Val Balanced Accuracy\n{title}", fontsize=11)
    axes[1].bar_label(bars, fmt="%.4f", padding=4)
    axes[1].grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Results — {title}:")
    print(f"    {label_a:35s}: best val bacc = {result_a['best_val_bacc']:.4f}")
    print(f"    {label_b:35s}: best val bacc = {result_b['best_val_bacc']:.4f}")


# Generic ablation trainer (SimpleCNN, ResNet50, ViT)

def _train_ablation_config_generic(
    train_df,
    val_df,
    device:            torch.device,
    model_factory,                    # callable() → nn.Module
    use_augmentation:  bool,
    use_class_weights: bool,
    freeze_backbone:   bool,
    run_tag:           str,
    save_dir:          str,
    lr:                float = 1e-4,
    weight_decay:      float = 1e-4,
) -> dict:
    """
    Generic ablation trainer for SimpleCNN, ResNet50, and ViT.
    freeze_backbone freezes model.backbone params when the attribute exists.
    """
    train_transform = get_train_transforms() if use_augmentation else get_val_transforms()
    val_transform   = get_val_transforms()

    train_dataset = SkinLesionDataset(train_df, transform=train_transform)
    val_dataset   = SkinLesionDataset(val_df,   transform=val_transform)

    if use_class_weights:
        sampler = get_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_dataset, batch_size=32, sampler=sampler,
            num_workers=2, pin_memory=False, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=32, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=True,
        )

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = model_factory().to(device)

    if freeze_backbone:
        backbone = getattr(model, "backbone", None)
        if backbone is not None:
            for p in backbone.parameters():
                p.requires_grad = False
    else:
        for p in model.parameters():
            p.requires_grad = True

    if use_class_weights:
        weights   = compute_class_weights(train_df).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=ABLATION_EPOCHS, eta_min=1e-7)

    scaler = (torch.amp.GradScaler("cuda")
              if torch.cuda.is_available() else None)

    early_stop = EarlyStopping(
        patience=ABLATION_PATIENCE,
        checkpoint_path=f"{save_dir}/{run_tag}_best.pth",
    )

    history = {"train_bacc": [], "val_bacc": [], "val_f1": []}

    print(f"\n  [{run_tag}] aug={use_augmentation} | "
          f"class_weights={use_class_weights} | freeze={freeze_backbone}")

    for epoch in range(1, ABLATION_EPOCHS + 1):
        train_m = run_train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_m   = run_val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_bacc"].append(train_m["balanced_accuracy"])
        history["val_bacc"].append(val_m["balanced_accuracy"])
        history["val_f1"].append(val_m["f1_macro"])

        print(f"    Ep {epoch:02d}/{ABLATION_EPOCHS}  "
              f"train_bacc={train_m['balanced_accuracy']:.4f}  "
              f"val_bacc={val_m['balanced_accuracy']:.4f}")

        if early_stop.step(val_m["balanced_accuracy"], model):
            print(f"    Early stopping at epoch {epoch}")
            break

    return {
        "tag":           run_tag,
        "best_val_bacc": float(early_stop.best_score or max(history["val_bacc"])),
        "history":       history,
    }


# ── SimpleCNN ablation studies ──────────────────────────────────────────────

def study_simplecnn_augmentation(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """SimpleCNN: no augmentation vs full augmentation pipeline."""
    print("\n" + "=" * 60)
    print("ABLATION — SimpleCNN: Data Augmentation")
    print("=" * 60)

    no_aug   = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNN,
        use_augmentation=False, use_class_weights=True, freeze_backbone=False,
        run_tag="scnn_aug_off", save_dir=save_dir, lr=1e-3,
    )
    with_aug = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNN,
        use_augmentation=True,  use_class_weights=True, freeze_backbone=False,
        run_tag="scnn_aug_on",  save_dir=save_dir, lr=1e-3,
    )

    results = {"scnn_no_aug": no_aug, "scnn_with_aug": with_aug}
    _save_ablation_results(
        results, save_dir,
        filename="simplecnn_study1_augmentation",
        title="SimpleCNN Study 1: Effect of Data Augmentation",
        label_a="No augmentation",
        label_b="With augmentation",
    )
    return results


def study_simplecnn_class_weights(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """SimpleCNN: uniform CrossEntropy vs weighted CrossEntropy."""
    print("\n" + "=" * 60)
    print("ABLATION — SimpleCNN: Class Weighting")
    print("=" * 60)

    no_w   = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNN,
        use_augmentation=True, use_class_weights=False, freeze_backbone=False,
        run_tag="scnn_weights_off", save_dir=save_dir, lr=1e-3,
    )
    with_w = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNN,
        use_augmentation=True, use_class_weights=True,  freeze_backbone=False,
        run_tag="scnn_weights_on",  save_dir=save_dir, lr=1e-3,
    )

    results = {"scnn_no_weights": no_w, "scnn_with_weights": with_w}
    _save_ablation_results(
        results, save_dir,
        filename="simplecnn_study2_class_weights",
        title="SimpleCNN Study 2: Effect of Class Weighting",
        label_a="Uniform loss",
        label_b="Weighted CE loss",
    )
    return results


def study_simplecnn_depth(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """SimpleCNN: shallow (2 conv blocks) vs deep (4 conv blocks)."""
    print("\n" + "=" * 60)
    print("ABLATION — SimpleCNN: Network Depth")
    print("=" * 60)

    shallow = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNNShallow,
        use_augmentation=True, use_class_weights=True, freeze_backbone=False,
        run_tag="scnn_shallow", save_dir=save_dir, lr=1e-3,
    )
    deep    = _train_ablation_config_generic(
        train_df, val_df, device, SimpleCNN,
        use_augmentation=True, use_class_weights=True, freeze_backbone=False,
        run_tag="scnn_deep",    save_dir=save_dir, lr=1e-3,
    )

    results = {"scnn_shallow": shallow, "scnn_deep": deep}
    _save_ablation_results(
        results, save_dir,
        filename="simplecnn_study3_depth",
        title="SimpleCNN Study 3: Effect of Network Depth",
        label_a="Shallow (2 blocks)",
        label_b="Deep (4 blocks)",
    )
    return results


# ── ResNet50 ablation studies ────────────────────────────────────────────────

def study_resnet50_augmentation(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ResNet50: no augmentation vs full augmentation pipeline."""
    print("\n" + "=" * 60)
    print("ABLATION — ResNet50: Data Augmentation")
    print("=" * 60)

    no_aug   = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=False, use_class_weights=True, freeze_backbone=False,
        run_tag="r50_aug_off", save_dir=save_dir,
    )
    with_aug = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=True,  use_class_weights=True, freeze_backbone=False,
        run_tag="r50_aug_on",  save_dir=save_dir,
    )

    results = {"r50_no_aug": no_aug, "r50_with_aug": with_aug}
    _save_ablation_results(
        results, save_dir,
        filename="resnet50_study1_augmentation",
        title="ResNet50 Study 1: Effect of Data Augmentation",
        label_a="No augmentation",
        label_b="With augmentation",
    )
    return results


def study_resnet50_class_weights(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ResNet50: uniform CrossEntropy vs weighted CrossEntropy."""
    print("\n" + "=" * 60)
    print("ABLATION — ResNet50: Class Weighting")
    print("=" * 60)

    no_w   = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=True, use_class_weights=False, freeze_backbone=False,
        run_tag="r50_weights_off", save_dir=save_dir,
    )
    with_w = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=True, use_class_weights=True,  freeze_backbone=False,
        run_tag="r50_weights_on",  save_dir=save_dir,
    )

    results = {"r50_no_weights": no_w, "r50_with_weights": with_w}
    _save_ablation_results(
        results, save_dir,
        filename="resnet50_study2_class_weights",
        title="ResNet50 Study 2: Effect of Class Weighting",
        label_a="Uniform loss",
        label_b="Weighted CE loss",
    )
    return results


def study_resnet50_freezing(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ResNet50: frozen backbone (head only) vs full fine-tuning."""
    print("\n" + "=" * 60)
    print("ABLATION — ResNet50: Backbone Freezing")
    print("=" * 60)

    frozen = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=True, use_class_weights=True, freeze_backbone=True,
        run_tag="r50_frozen",  save_dir=save_dir,
    )
    full   = _train_ablation_config_generic(
        train_df, val_df, device, ResNet50,
        use_augmentation=True, use_class_weights=True, freeze_backbone=False,
        run_tag="r50_full_ft", save_dir=save_dir,
    )

    results = {"r50_frozen": frozen, "r50_full_ft": full}
    _save_ablation_results(
        results, save_dir,
        filename="resnet50_study3_backbone",
        title="ResNet50 Study 3: Effect of Backbone Fine-tuning",
        label_a="Frozen backbone",
        label_b="Full fine-tuning",
    )
    return results


# ── ViT+LoRA ablation studies ────────────────────────────────────────────────

def study_vit_augmentation(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ViT+LoRA: no augmentation vs full augmentation pipeline."""
    print("\n" + "=" * 60)
    print("ABLATION — ViT+LoRA: Data Augmentation")
    print("=" * 60)

    no_aug   = _train_ablation_config_generic(
        train_df, val_df, device, ViTWithLoRA,
        use_augmentation=False, use_class_weights=True, freeze_backbone=False,
        run_tag="vit_aug_off", save_dir=save_dir,
    )
    with_aug = _train_ablation_config_generic(
        train_df, val_df, device, ViTWithLoRA,
        use_augmentation=True,  use_class_weights=True, freeze_backbone=False,
        run_tag="vit_aug_on",  save_dir=save_dir,
    )

    results = {"vit_no_aug": no_aug, "vit_with_aug": with_aug}
    _save_ablation_results(
        results, save_dir,
        filename="vit_study1_augmentation",
        title="ViT+LoRA Study 1: Effect of Data Augmentation",
        label_a="No augmentation",
        label_b="With augmentation",
    )
    return results


def study_vit_class_weights(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ViT+LoRA: uniform CrossEntropy vs weighted CrossEntropy."""
    print("\n" + "=" * 60)
    print("ABLATION — ViT+LoRA: Class Weighting")
    print("=" * 60)

    no_w   = _train_ablation_config_generic(
        train_df, val_df, device, ViTWithLoRA,
        use_augmentation=True, use_class_weights=False, freeze_backbone=False,
        run_tag="vit_weights_off", save_dir=save_dir,
    )
    with_w = _train_ablation_config_generic(
        train_df, val_df, device, ViTWithLoRA,
        use_augmentation=True, use_class_weights=True,  freeze_backbone=False,
        run_tag="vit_weights_on",  save_dir=save_dir,
    )

    results = {"vit_no_weights": no_w, "vit_with_weights": with_w}
    _save_ablation_results(
        results, save_dir,
        filename="vit_study2_class_weights",
        title="ViT+LoRA Study 2: Effect of Class Weighting",
        label_a="Uniform loss",
        label_b="Weighted CE loss",
    )
    return results


def study_vit_lora_rank(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """ViT+LoRA: low LoRA rank (2) vs high LoRA rank (8)."""
    print("\n" + "=" * 60)
    print("ABLATION — ViT+LoRA: LoRA Rank")
    print("=" * 60)

    low_rank  = _train_ablation_config_generic(
        train_df, val_df, device,
        model_factory=lambda: ViTWithLoRA(lora_rank=2),
        use_augmentation=True, use_class_weights=True, freeze_backbone=False,
        run_tag="vit_lora_rank2", save_dir=save_dir,
    )
    high_rank = _train_ablation_config_generic(
        train_df, val_df, device,
        model_factory=lambda: ViTWithLoRA(lora_rank=8),
        use_augmentation=True, use_class_weights=True, freeze_backbone=False,
        run_tag="vit_lora_rank8", save_dir=save_dir,
    )

    results = {"vit_lora_rank2": low_rank, "vit_lora_rank8": high_rank}
    _save_ablation_results(
        results, save_dir,
        filename="vit_study3_lora_rank",
        title="ViT+LoRA Study 3: Effect of LoRA Rank",
        label_a="LoRA rank=2",
        label_b="LoRA rank=8",
    )
    return results


# Summary table across all ablation studies

def build_ablation_summary(all_results: dict, save_dir: str) -> pd.DataFrame:
    """Collect all ablation results into a single comparison table."""
    label_map = {
        # EfficientNet
        "no_augmentation":    ("EfficientNet / Study 1", "No augmentation"),
        "with_augmentation":  ("EfficientNet / Study 1", "With augmentation ✓"),
        "no_class_weights":   ("EfficientNet / Study 2", "Uniform loss"),
        "with_class_weights": ("EfficientNet / Study 2", "Weighted CE loss ✓"),
        "frozen_backbone":    ("EfficientNet / Study 3", "Frozen backbone"),
        "full_finetuning":    ("EfficientNet / Study 3", "Full fine-tuning ✓"),
        # SimpleCNN
        "scnn_no_aug":        ("SimpleCNN / Study 1",    "No augmentation"),
        "scnn_with_aug":      ("SimpleCNN / Study 1",    "With augmentation ✓"),
        "scnn_no_weights":    ("SimpleCNN / Study 2",    "Uniform loss"),
        "scnn_with_weights":  ("SimpleCNN / Study 2",    "Weighted CE loss ✓"),
        "scnn_shallow":       ("SimpleCNN / Study 3",    "Shallow (2 blocks)"),
        "scnn_deep":          ("SimpleCNN / Study 3",    "Deep (4 blocks) ✓"),
        # ResNet50
        "r50_no_aug":         ("ResNet50 / Study 1",     "No augmentation"),
        "r50_with_aug":       ("ResNet50 / Study 1",     "With augmentation ✓"),
        "r50_no_weights":     ("ResNet50 / Study 2",     "Uniform loss"),
        "r50_with_weights":   ("ResNet50 / Study 2",     "Weighted CE loss ✓"),
        "r50_frozen":         ("ResNet50 / Study 3",     "Frozen backbone"),
        "r50_full_ft":        ("ResNet50 / Study 3",     "Full fine-tuning ✓"),
        # ViT
        "vit_no_aug":         ("ViT+LoRA / Study 1",    "No augmentation"),
        "vit_with_aug":       ("ViT+LoRA / Study 1",    "With augmentation ✓"),
        "vit_no_weights":     ("ViT+LoRA / Study 2",    "Uniform loss"),
        "vit_with_weights":   ("ViT+LoRA / Study 2",    "Weighted CE loss ✓"),
        "vit_lora_rank2":     ("ViT+LoRA / Study 3",    "LoRA rank=2"),
        "vit_lora_rank8":     ("ViT+LoRA / Study 3",    "LoRA rank=8 ✓"),
    }

    rows = []
    for key, val in all_results.items():
        study, setting = label_map.get(key, ("?", key))
        rows.append({
            "Study":         study,
            "Setting":       setting,
            "Best Val Bacc": f"{val['best_val_bacc']:.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{save_dir}/ablation_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False))
    return df
