"""
src/train.py
============
points covered:
  - "Correct training loop, with monitoring of loss and metrics"
  - "Correct use of a validation set in addition to the training set"
  - "Model selection based on validation performance, not last epoch"
  - "Final testing performed only on the best selected model"
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, precision_score, roc_auc_score,
)
from sklearn.preprocessing import label_binarize

import sys as _sys
from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

from src.config import CFG


# Early Stopping


class EarlyStopping:
    """
    Monitor validation performance and stop training when it stagnates.
    Saves the BEST model checkpoint (not last epoch).
    Model selection based on validation performance, not last epoch.
    """

    def __init__(self, patience: int, checkpoint_path: str) -> None:
        self.patience        = patience
        self.checkpoint_path = checkpoint_path
        self.best_score      = None
        self.best_epoch      = None   # epoch number that achieved the best score
        self.wait_count      = 0      # epochs without improvement
        self.should_stop     = False

    def step(self, val_score: float, model: nn.Module, epoch: int) -> bool:
        """
        Call at end of each epoch.
        Returns True when training should stop.
        """
        if self.best_score is None or val_score > self.best_score + 1e-4:
            # Improvement → record epoch, save checkpoint, reset counter
            self.best_score = val_score
            self.best_epoch = epoch
            self.wait_count = 0
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"  ✓ Epoch {epoch}: new best val_balanced_acc = {val_score:.4f} — checkpoint saved")
            return False
        else:
            self.wait_count += 1
            print(f"  No improvement — patience {self.wait_count}/{self.patience}")
            if self.wait_count >= self.patience:
                self.should_stop = True
                return True
            return False



# One epoch: training pass

def run_train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler,                         # torch.amp.GradScaler or None
) -> dict:
    """
    Run one full training epoch.

    Returns dict with: loss, accuracy, balanced_accuracy, f1_macro
    """
    model.train()
    total_loss = 0.0
    all_predictions   = []
    all_ground_truth  = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Reset gradients (set_to_none is more memory-efficient)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision: forward in float16, backward in float32
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            # Gradient clipping prevents exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        predictions = logits.detach().argmax(dim=1).cpu().numpy()
        probs       = torch.softmax(logits.detach().float(), dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_ground_truth.extend(labels.cpu().numpy())
        all_probabilities.extend(probs)

    n_samples    = len(all_ground_truth)
    probabilities = np.array(all_probabilities, dtype=np.float32)
    n_classes    = probabilities.shape[1]
    labels_bin   = label_binarize(all_ground_truth, classes=list(range(n_classes)))
    try:
        roc_auc = roc_auc_score(labels_bin, probabilities,
                                average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "loss":               total_loss / n_samples,
        "accuracy":           np.mean(np.array(all_predictions) == np.array(all_ground_truth)),
        "balanced_accuracy":  balanced_accuracy_score(all_ground_truth, all_predictions),
        "f1_macro":           f1_score(all_ground_truth, all_predictions,
                                       average="macro", zero_division=0),
        "precision_macro":    precision_score(all_ground_truth, all_predictions,
                                              average="macro", zero_division=0),
        "roc_auc_macro":      float(roc_auc),
    }
    return metrics


# One epoch: validation pass (no gradient computation)

@torch.no_grad()
def run_val_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> dict:
    """
    Run evaluation on validation or test set.
    Uses torch.no_grad() to save memory.
    """
    model.eval()
    total_loss = 0.0
    all_predictions   = []
    all_ground_truth  = []
    all_probabilities = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Cast to float32 before loss to prevent float16 overflow (NaN)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            logits = model(images)

        logits_f = logits.float()
        loss = criterion(logits_f, labels)
        total_loss += loss.item() * images.size(0)

        predictions = logits_f.argmax(dim=1).cpu().numpy()
        probs       = torch.softmax(logits_f, dim=1).cpu().numpy()
        all_predictions.extend(predictions)
        all_ground_truth.extend(labels.cpu().numpy())
        all_probabilities.extend(probs)

    n_samples     = len(all_ground_truth)
    probabilities = np.array(all_probabilities, dtype=np.float32)
    n_classes     = probabilities.shape[1]
    labels_bin    = label_binarize(all_ground_truth, classes=list(range(n_classes)))
    try:
        roc_auc = roc_auc_score(labels_bin, probabilities,
                                average="macro", multi_class="ovr")
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "loss":               total_loss / n_samples,
        "accuracy":           np.mean(np.array(all_predictions) == np.array(all_ground_truth)),
        "balanced_accuracy":  balanced_accuracy_score(all_ground_truth, all_predictions),
        "f1_macro":           f1_score(all_ground_truth, all_predictions,
                                       average="macro", zero_division=0),
        "precision_macro":    precision_score(all_ground_truth, all_predictions,
                                              average="macro", zero_division=0),
        "roc_auc_macro":      float(roc_auc),
    }
    return metrics


# Generic training loop (ResNet50, ViT, SimpleCNN)

def train_model(
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    class_weights: torch.Tensor,
    device:        torch.device,
    model_name:    str,
) -> dict:
    """
    Generic training loop for ResNet50, ViT+LoRA, and SimpleCNN.
    Reads hyperparameters from the corresponding section of config.yaml.
    """
    # Map model name to config section
    cfg_map = {
        "resnet50":   CFG.resnet50,
        "vit":        CFG.vit,
        "simple_cnn": CFG.simple_cnn,
    }
    mcfg = cfg_map[model_name]

    checkpoint_dir = f"{CFG.paths.checkpoints}/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=mcfg.lr, weight_decay=mcfg.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=mcfg.epochs,
                                  eta_min=getattr(mcfg, "min_lr", 1e-7))
    early_stop = EarlyStopping(
        patience=mcfg.early_stopping_patience,
        checkpoint_path=f"{checkpoint_dir}/best.pth",
    )

    use_amp = getattr(mcfg, "use_amp", False) and torch.cuda.is_available()
    scaler  = torch.amp.GradScaler("cuda") if use_amp else None

    history = {
        "train_loss":      [], "val_loss":      [],
        "train_bacc":      [], "val_bacc":      [],
        "train_f1":        [], "val_f1":        [],
        "train_precision": [], "val_precision": [],
        "train_roc_auc":   [], "val_roc_auc":   [],
    }

    print(f"\n{'=' * 60}")
    print(f"TRAINING: {model_name.upper()} for {mcfg.epochs} epochs")
    print(f"{'=' * 60}")

    for epoch in range(1, mcfg.epochs + 1):
        t_start = time.time()
        train_metrics = run_train_epoch(
            model, train_loader, criterion, optimizer, device, scaler)
        val_metrics = run_val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t_start
        print(
            f"  Ep {epoch:02d}/{mcfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_bacc={train_metrics['balanced_accuracy']:.4f} "
            f"train_f1={train_metrics['f1_macro']:.4f} "
            f"train_prec={train_metrics['precision_macro']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"val_prec={val_metrics['precision_macro']:.4f} "
            f"val_auc={val_metrics['roc_auc_macro']:.4f} | "
            f"{elapsed:.1f}s"
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_bacc"].append(train_metrics["balanced_accuracy"])
        history["val_bacc"].append(val_metrics["balanced_accuracy"])
        history["train_f1"].append(train_metrics["f1_macro"])
        history["val_f1"].append(val_metrics["f1_macro"])
        history["train_precision"].append(train_metrics["precision_macro"])
        history["val_precision"].append(val_metrics["precision_macro"])
        history["train_roc_auc"].append(train_metrics["roc_auc_macro"])
        history["val_roc_auc"].append(val_metrics["roc_auc_macro"])

        if early_stop.step(val_metrics["balanced_accuracy"], model, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    history["best_epoch"]    = early_stop.best_epoch
    history["best_val_bacc"] = early_stop.best_score

    with open(f"{checkpoint_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"[train] {model_name} done — best val_balanced_acc: {early_stop.best_score:.4f} "
          f"(epoch {early_stop.best_epoch})")
    return history