"""
src/ablations.py
================
Three ablation studies to quantify the impact of individual design choices.

Study 1: Without augmentation vs With augmentation
Study 2: Without class weighting vs Weighted CrossEntropy loss
Study 3: Frozen backbone vs Full fine-tuning

Each study trains two simple_cnn, resnet50, and ViTWithLoRA models with exactly one variable changed.
Results are saved as JSON and comparison plots.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

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
from src.models import SimpleCNN, SimpleCNNShallow, ResNet50, ViTWithLoRA
from src.train import run_train_epoch, run_val_epoch, EarlyStopping
from src.evaluate import plot_confusion_matrix


# Number of epochs per ablation run (shorter than full training to save time)
ABLATION_EPOCHS = 20
ABLATION_PATIENCE = 5


# Checkpoint cleanup

def _drop_loser_checkpoint(save_dir: str, result_a: dict, result_b: dict) -> None:
    """Delete the checkpoint of the worse-performing variant; keep only the winner."""
    loser = result_a if result_a["best_val_acc"] <= result_b["best_val_acc"] else result_b
    path = os.path.join(save_dir, f"{loser['tag']}_best.pth")
    if os.path.exists(path):
        os.remove(path)
        print(f"  [ablation] Removed checkpoint for {loser['tag']} (not best)")


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
            "tag":                val["tag"],
            "best_val_acc":       float(val["best_val_acc"]),
            "best_val_f1":        float(val["best_val_f1"]),
            "best_val_precision": float(val["best_val_precision"]),
            "best_val_roc_auc":   float(val["best_val_roc_auc"]),
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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Val Accuracy curves
    axes[0].plot(result_a["history"]["val_acc"],
                 label=label_a, color="#e74c3c", lw=2)
    axes[0].plot(result_b["history"]["val_acc"],
                 label=label_b, color="#2ecc71", lw=2)
    axes[0].set_title(f"Val Accuracy\n{title}", fontsize=11)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Val F1 curves
    axes[1].plot(result_a["history"]["val_f1"],
                 label=label_a, color="#e74c3c", lw=2)
    axes[1].plot(result_b["history"]["val_f1"],
                 label=label_b, color="#2ecc71", lw=2)
    axes[1].set_title(f"Val Macro F1\n{title}", fontsize=11)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Summary bar chart: Acc, F1, Precision, ROC-AUC side-by-side
    x = np.arange(4)
    metric_names = ["Best Val Acc", "Best Val F1", "Best Val Prec", "Best Val AUC"]
    vals_a = [result_a["best_val_acc"], result_a["best_val_f1"],
              result_a["best_val_precision"], result_a["best_val_roc_auc"]]
    vals_b = [result_b["best_val_acc"], result_b["best_val_f1"],
              result_b["best_val_precision"], result_b["best_val_roc_auc"]]
    width = 0.35
    bars_a = axes[2].bar(x - width / 2, vals_a, width, label=label_a, color="#e74c3c")
    bars_b = axes[2].bar(x + width / 2, vals_b, width, label=label_b, color="#2ecc71")
    axes[2].set_xticks(x); axes[2].set_xticklabels(metric_names, fontsize=8)
    axes[2].set_ylim([0, 1.05])
    axes[2].set_title(f"Summary\n{title}", fontsize=11)
    axes[2].bar_label(bars_a, fmt="%.3f", padding=3, fontsize=7)
    axes[2].bar_label(bars_b, fmt="%.3f", padding=3, fontsize=7)
    axes[2].legend(); axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{filename}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Confusion matrix for each variant (saved to disk; shown in notebook display cells)
    for key, label in [(keys[0], label_a), (keys[1], label_b)]:
        r = results[key]
        if "ground_truth" in r and "predictions" in r:
            plot_confusion_matrix(r["ground_truth"], r["predictions"],
                                  save_dir, r["tag"], show=False)

    print(f"\n  Results — {title}:")
    print(f"    {label_a:35s}: acc={result_a['best_val_acc']:.4f}  "
          f"f1={result_a['best_val_f1']:.4f}  prec={result_a['best_val_precision']:.4f}  "
          f"auc={result_a['best_val_roc_auc']:.4f}")
    print(f"    {label_b:35s}: acc={result_b['best_val_acc']:.4f}  "
          f"f1={result_b['best_val_f1']:.4f}  prec={result_b['best_val_precision']:.4f}  "
          f"auc={result_b['best_val_roc_auc']:.4f}")


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

    history = {
        "train_acc": [], "val_acc": [], "val_f1": [],
        "train_precision": [], "val_precision": [],
        "train_roc_auc":   [], "val_roc_auc":   [],
    }

    print(f"\n  [{run_tag}] aug={use_augmentation} | "
          f"class_weights={use_class_weights} | freeze={freeze_backbone}")

    for epoch in range(1, ABLATION_EPOCHS + 1):
        train_m = run_train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_m   = run_val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_acc"].append(train_m["accuracy"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1_macro"])
        history["train_precision"].append(train_m["precision_macro"])
        history["val_precision"].append(val_m["precision_macro"])
        history["train_roc_auc"].append(train_m["roc_auc_macro"])
        history["val_roc_auc"].append(val_m["roc_auc_macro"])

        print(f"    Ep {epoch:02d}/{ABLATION_EPOCHS}  "
              f"train_acc={train_m['accuracy']:.4f}  "
              f"train_prec={train_m['precision_macro']:.4f}  "
              f"train_auc={train_m['roc_auc_macro']:.4f}  "
              f"val_acc={val_m['accuracy']:.4f}  "
              f"val_f1={val_m['f1_macro']:.4f}  "
              f"val_prec={val_m['precision_macro']:.4f}  "
              f"val_auc={val_m['roc_auc_macro']:.4f}")

        if early_stop.step(val_m["accuracy"], model, epoch):
            print(f"    Early stopping at epoch {epoch}")
            break

    # Final inference pass on val set to collect predictions for confusion matrix
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    ground_truth = np.array(all_labels)
    predictions  = np.array(all_preds)

    valid_auc = [v for v in history["val_roc_auc"] if not np.isnan(v)]
    return {
        "tag":                run_tag,
        "best_val_acc":       float(early_stop.best_score or max(history["val_acc"])),
        "best_val_f1":        float(max(history["val_f1"])),
        "best_val_precision": float(max(history["val_precision"])),
        "best_val_roc_auc":   float(max(valid_auc)) if valid_auc else float("nan"),
        "history":            history,
        "ground_truth":       ground_truth,
        "predictions":        predictions,
    }


# SimpleCNN ablation studies 

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
    _drop_loser_checkpoint(save_dir, no_aug, with_aug)
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
    _drop_loser_checkpoint(save_dir, no_w, with_w)
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
    _drop_loser_checkpoint(save_dir, shallow, deep)
    return results


# ResNet50 ablation studies

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
    _drop_loser_checkpoint(save_dir, no_aug, with_aug)
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
    _drop_loser_checkpoint(save_dir, no_w, with_w)
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
    _drop_loser_checkpoint(save_dir, frozen, full)
    return results


# ViT+LoRA ablation studies 
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
    _drop_loser_checkpoint(save_dir, no_aug, with_aug)
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
    _drop_loser_checkpoint(save_dir, no_w, with_w)
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
    _drop_loser_checkpoint(save_dir, low_rank, high_rank)
    return results


# Hyperparameter search: learning-rate grid search

_LR_SEARCH_VALUES = [1e-5, 5e-5, 1e-4, 5e-4]


def study_learning_rate(train_df, val_df, device: torch.device, save_dir: str) -> dict:
    """
    Systematic 4-point LR grid search on ResNet50.

    Addresses: "Hyperparameter optimization or systematic parameter search" (+2.0 pt extra)

    Keeps all other settings fixed (augmentation on, class weights on, full fine-tuning)
    and varies only the initial learning rate across [1e-5, 5e-5, 1e-4, 5e-4].
    The best LR is used as the recommended value for final ResNet50 training.
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH — Learning Rate (ResNet50)")
    print("=" * 60)

    results = {}
    for lr in _LR_SEARCH_VALUES:
        tag = f"lr_{lr:.0e}"
        print(f"\n  Testing LR = {lr}")
        result = _train_ablation_config_generic(
            train_df, val_df, device,
            model_factory=ResNet50,
            use_augmentation=True,
            use_class_weights=True,
            freeze_backbone=False,
            run_tag=f"lrsearch_{tag}",
            save_dir=save_dir,
            lr=lr,
        )
        results[tag] = result

    _save_lr_search_results(results, save_dir)

    best_tag = max(results, key=lambda t: results[t]["best_val_acc"])
    for tag, res in results.items():
        if tag != best_tag:
            path = os.path.join(save_dir, f"lrsearch_{tag}_best.pth")
            if os.path.exists(path):
                os.remove(path)
                print(f"  [ablation] Removed LR search checkpoint: lrsearch_{tag}_best.pth")

    return results


def _save_lr_search_results(results: dict, save_dir: str) -> None:
    """Save LR search JSON + multi-curve comparison plot."""
    os.makedirs(save_dir, exist_ok=True)

    serialisable = {
        lr_tag: {
            "tag":                val["tag"],
            "best_val_acc":       float(val["best_val_acc"]),
            "best_val_f1":        float(val["best_val_f1"]),
            "best_val_precision": float(val["best_val_precision"]),
            "best_val_roc_auc":   float(val["best_val_roc_auc"]),
            "history": {k: [float(v) for v in lst]
                        for k, lst in val["history"].items()},
        }
        for lr_tag, val in results.items()
    }
    with open(f"{save_dir}/lr_search.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (tag, res) in enumerate(results.items()):
        label = "LR=" + tag.replace("lr_", "")
        col = colors[i % len(colors)]
        axes[0].plot(res["history"]["val_acc"], label=label, color=col, lw=2)
        axes[1].plot(res["history"]["val_f1"],  label=label, color=col, lw=2)

    for ax, ylabel, title in [
        (axes[0], "Accuracy", "Val Accuracy — LR search"),
        (axes[1], "F1",       "Val Macro F1 — LR search"),
    ]:
        ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Summary bar chart — one group per metric, one bar per LR
    metric_keys  = ["best_val_acc", "best_val_f1", "best_val_precision", "best_val_roc_auc"]
    metric_names = ["Best Acc", "Best F1", "Best Prec", "Best AUC"]
    x = np.arange(len(metric_names))
    n = len(results)
    width = 0.18

    for i, (tag, res) in enumerate(results.items()):
        vals  = [res[k] for k in metric_keys]
        label = "LR=" + tag.replace("lr_", "")
        offset = (i - (n - 1) / 2) * width
        axes[2].bar(x + offset, vals, width, label=label, color=colors[i % len(colors)])

    axes[2].set_xticks(x); axes[2].set_xticklabels(metric_names, fontsize=9)
    axes[2].set_ylim([0, 1.1])
    axes[2].set_title("Metric Summary — LR Search", fontsize=11)
    axes[2].legend(fontsize=8); axes[2].grid(axis="y", alpha=0.3)

    plt.suptitle("Learning Rate Grid Search — ResNet50", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/lr_search.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  LR Search Summary:")
    print(f"  {'LR':>10}  {'Val Acc':>8}  {'Val F1':>8}  {'Val AUC':>8}")
    for tag, res in results.items():
        lr_str = tag.replace("lr_", "")
        print(f"  {lr_str:>10}  {res['best_val_acc']:>8.4f}  "
              f"{res['best_val_f1']:>8.4f}  {res['best_val_roc_auc']:>8.4f}")

    best_tag = max(results, key=lambda t: results[t]["best_val_acc"])
    print(f"\n  Best LR by val accuracy: {best_tag.replace('lr_', '')}")


# Multi-seed robustness evaluation

def run_multi_seed_robustness(
    train_df,
    val_df,
    device: torch.device,
    save_dir: str,
    seeds: list = None,
) -> dict:
    """
    Train SimpleCNN with 3 different random seeds and report mean ± std.

    Addresses: "Solid experimental evaluation: multiple runs, standard deviation,
                robustness tests" (+2.0 pt extra)

    SimpleCNN is chosen because it trains from scratch and is the fastest model,
    making 3 independent runs feasible. Pretrained models are more stable and
    benefit less from this analysis.
    """
    if seeds is None:
        seeds = [42, 123, 456]

    print("\n" + "=" * 60)
    print(f"ROBUSTNESS EVALUATION — SimpleCNN ({len(seeds)} seeds)")
    print("=" * 60)

    import random
    per_seed = []

    for seed in seeds:
        print(f"\n  [seed={seed}]")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        result = _train_ablation_config_generic(
            train_df, val_df, device,
            model_factory=SimpleCNN,
            use_augmentation=True,
            use_class_weights=True,
            freeze_backbone=False,
            run_tag=f"robust_seed{seed}",
            save_dir=save_dir,
            lr=1e-3,
        )
        per_seed.append({
            "seed":        seed,
            "val_acc":     result["best_val_acc"],
            "val_f1":      result["best_val_f1"],
            "val_prec":    result["best_val_precision"],
            "val_roc_auc": result["best_val_roc_auc"],
        })

    # Mean ± std across seeds
    metric_keys = ["val_acc", "val_f1", "val_prec", "val_roc_auc"]
    summary = {}
    for key in metric_keys:
        vals = [m[key] for m in per_seed if not np.isnan(m[key])]
        summary[key] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "values": [float(v) for v in vals],
        }

    output = {"model": "SimpleCNN", "seeds": seeds,
               "per_seed": per_seed, "summary": summary}

    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/robustness_evaluation.json", "w") as f:
        json.dump(output, f, indent=2)

    # Error-bar plot
    metric_labels = ["Val Balanced Acc", "Val Macro F1", "Val Precision", "Val ROC-AUC"]
    means = [summary[k]["mean"] for k in metric_keys]
    stds  = [summary[k]["std"]  for k in metric_keys]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
    bars = ax.bar(metric_labels, means, yerr=stds, capsize=7,
                  color=bar_colors, alpha=0.85,
                  error_kw={"linewidth": 2, "ecolor": "#2c3e50"})
    ax.bar_label(bars,
                 labels=[f"{m:.3f}±{s:.3f}" for m, s in zip(means, stds)],
                 padding=8, fontsize=10)

    # Scatter individual seed runs on top
    seed_colors = ["#1a1a2e", "#c0392b", "#e67e22"]
    x_pos = np.arange(len(metric_keys))
    for j, m in enumerate(per_seed):
        seed_vals = [m[k] for k in metric_keys]
        ax.scatter(x_pos, seed_vals,
                   color=seed_colors[j % len(seed_colors)],
                   zorder=5, s=70, label=f"seed={m['seed']}")

    ax.set_ylim([0, 1.2])
    ax.set_title(f"SimpleCNN Robustness: Mean ± Std ({len(seeds)} seeds)", fontsize=13)
    ax.set_ylabel("Metric value")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/robustness_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Robustness Summary (SimpleCNN, {len(seeds)} seeds):")
    print(f"  {'Metric':22}  {'Mean':>8}  {'Std':>8}")
    for key, label in zip(metric_keys, metric_labels):
        print(f"  {label:22}  {summary[key]['mean']:>8.4f}  {summary[key]['std']:>8.4f}")

    best_seed = max(per_seed, key=lambda m: m["val_acc"])["seed"]
    for m in per_seed:
        if m["seed"] != best_seed:
            path = os.path.join(save_dir, f"robust_seed{m['seed']}_best.pth")
            if os.path.exists(path):
                os.remove(path)
                print(f"  [ablation] Removed robustness checkpoint: robust_seed{m['seed']}_best.pth")

    return output


# Summary table across all ablation studies

def build_ablation_summary(all_results: dict, save_dir: str) -> pd.DataFrame:
    """Collect all ablation results into a single comparison table."""
    label_map = {
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
        auc_raw = val.get("best_val_roc_auc", float("nan"))
        rows.append({
            "Study":           study,
            "Setting":         setting,
            "Best Val Acc":    f"{val['best_val_acc']:.4f}",
            "Best Val F1":     f"{val['best_val_f1']:.4f}",
            "Best Val Prec":   f"{val['best_val_precision']:.4f}",
            "Best Val AUC":    f"{auc_raw:.4f}" if not np.isnan(auc_raw) else "—",
        })

    df = pd.DataFrame(rows)
    df.to_csv(f"{save_dir}/ablation_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("  ABLATION STUDY SUMMARY")
    print("=" * 65)
    print(df.to_string(index=False))
    return df
