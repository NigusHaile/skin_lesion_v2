"""
src/gradcam.py
==============
GradCAM explainability for skin lesion models.

  - "Use of advanced techniques: explainability"
  - "In-depth error analysis: GradCAM on failure cases"
  - "Special care in the demo: visualizations, explainability"

Algorithm (same as lab):
  1. Forward hook  → save feature maps of target layer
  2. Backward hook → save gradients of target layer
  3. Global Average Pool of gradients → per-channel weights
  4. Weighted sum of feature maps → raw CAM
  5. ReLU → interpolate to input size → normalise to [0, 1]
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.nn.functional as F
import torch.nn as nn

from src.config import CFG
from src.dataset import CLASS_LABELS, CLASS_NAMES, CLASS_COLORS


# GradCAM

class GradCAM:
    """
    Gradient-weighted Class Activation Map (mirrors course lab structure).

    Hooks are registered on a specific target layer. After compute() the
    caller should call remove_hooks() to avoid memory leaks.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        # Always unwrap torch.compile so hooks fire through the real module.
        self.model        = getattr(model, "_orig_mod", model)
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, input, output) -> None:
        # Save a detached clone for the CAM computation.
        self.activations = output.detach().clone()
        # Register a tensor-level gradient hook — avoids the inplace-ReLU conflict
        # that register_full_backward_hook causes with inplace activations.
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad) -> None:
        self.gradients = grad.detach().clone()

    def compute(
        self,
        image_tensor: torch.Tensor,   # (3, H, W) normalised
        class_idx:    int = None,     # None → use predicted class
    ) -> tuple:
        """
        Compute GradCAM heatmap for a single image.

        Returns:
            cam:        (H, W) float32 array in [0, 1]
            pred_class: int — argmax class index
            confidence: float — softmax probability of predicted class
            probs:      (7,) float32 array — full softmax distribution
        """
        self.activations = None
        self.gradients   = None

        self.model.eval()
        x = image_tensor.unsqueeze(0)

        output     = self.model(x)
        probs      = torch.softmax(output.float(), dim=1)[0]
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()

        target_class = class_idx if class_idx is not None else pred_class

        self.model.zero_grad()
        output[0, target_class].backward()

        if self.gradients is None or self.activations is None:
            blank = np.zeros(
                (CFG.data.image_size, CFG.data.image_size), dtype=np.float32
            )
            return blank, pred_class, confidence, probs.cpu().detach().numpy()

        # Global average pool gradients → per-channel importance weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=(CFG.data.image_size, CFG.data.image_size),
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, pred_class, confidence, probs.cpu().detach().numpy()

    def remove_hooks(self) -> None:
        self._fwd_handle.remove()


# Image utilities

def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor to a HWC uint8 numpy array."""
    mean = np.array(CFG.data.imagenet_mean, dtype=np.float32)
    std  = np.array(CFG.data.imagenet_std,  dtype=np.float32)

    img = tensor.clone().cpu().float().numpy()
    img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_cam(
    image_np: np.ndarray,
    cam:      np.ndarray,
    alpha:    float = 0.50,
) -> np.ndarray:
    """
    Blend a turbo-colourmap GradCAM heatmap over the original image.
    Uses 'turbo' which gives cleaner, more readable activation maps than jet.
    """
    heatmap       = plt.cm.turbo(cam)[:, :, :3]
    heatmap_float = (heatmap * 255).astype(np.float32)
    image_float   = image_np.astype(np.float32)
    blended       = alpha * heatmap_float + (1 - alpha) * image_float
    return np.clip(blended, 0, 255).astype(np.uint8)


# Batch visualisation

def generate_gradcam_examples(
    model:        nn.Module,
    target_layer: nn.Module,
    dataset,
    device:       torch.device,
    save_dir:     str,
    model_name:   str,
    n_correct:    int = 8,
    n_wrong:      int = 8,
) -> None:
    """
    Generate GradCAM grids for the most confident correct predictions
    and n_wrong incorrect predictions.

    Correct predictions are sorted by confidence (highest first) so the
    most impressive, certain results appear at the top.
    """
    os.makedirs(save_dir, exist_ok=True)
    gradcam = GradCAM(model, target_layer)
    model.eval()

    # Collect more candidates than needed so we can sort by confidence
    correct_pool, wrong_examples = [], []
    max_scan = min(len(dataset), n_correct * 20 + n_wrong * 10)

    for idx in range(max_scan):
        if len(wrong_examples) >= n_wrong and len(correct_pool) >= n_correct * 5:
            break

        item = dataset[idx]
        image_tensor, true_label = item[:2]

        image_tensor = image_tensor.to(device)
        cam, pred_class, confidence, probs = gradcam.compute(image_tensor)
        image_display = tensor_to_display_image(image_tensor.cpu())
        cam_overlay   = overlay_cam(image_display, cam)

        entry = {
            "image":      image_display,
            "cam":        cam,
            "overlay":    cam_overlay,
            "true_label": true_label,
            "pred_class": pred_class,
            "confidence": confidence,
            "probs":      probs,
        }

        if pred_class == true_label:
            correct_pool.append(entry)
        elif len(wrong_examples) < n_wrong:
            wrong_examples.append(entry)

    gradcam.remove_hooks()

    # Sort correct by confidence descending → take top n_correct
    correct_pool.sort(key=lambda e: e["confidence"], reverse=True)
    correct_examples = correct_pool[:n_correct]

    _save_gradcam_grid(
        correct_examples,
        save_path=f"{save_dir}/{model_name}_gradcam_correct.png",
        title=f"{model_name} — GradCAM: Top-{len(correct_examples)} Most Confident Correct Predictions",
        is_correct_grid=True,
    )
    _save_gradcam_grid(
        wrong_examples,
        save_path=f"{save_dir}/{model_name}_gradcam_incorrect.png",
        title=f"{model_name} — GradCAM: Incorrect Predictions (Error Analysis)",
        is_correct_grid=False,
    )

    print(f"[gradcam] Saved {len(correct_examples)} correct + "
          f"{len(wrong_examples)} incorrect examples.")

    # Also generate the per-class showcase
    generate_gradcam_class_showcase(
        correct_pool,
        save_dir=save_dir,
        model_name=model_name,
    )


def _save_gradcam_grid(
    examples:        list,
    save_path:       str,
    title:           str,
    is_correct_grid: bool = True,
) -> None:
    """
    Save a 4-column grid per example:
      Col 0 — Original image
      Col 1 — Raw GradCAM heatmap (turbo) with colorbar
      Col 2 — GradCAM overlay on image
      Col 3 — Per-class probability bar chart

    Examples are sorted by confidence before rendering.
    """
    if not examples:
        return

    examples = sorted(examples, key=lambda e: e["confidence"], reverse=True)
    n = len(examples)

    fig = plt.figure(figsize=(18, 4.2 * n), facecolor="#1a1a2e")

    # Outer title with summary stats
    mean_conf = np.mean([e["confidence"] for e in examples])
    plt.suptitle(
        f"{title}\n"
        f"Mean confidence: {mean_conf:.1%}",
        fontsize=13, color="white", y=1.002, fontweight="bold",
    )

    for row, ex in enumerate(examples):
        true_name = CLASS_NAMES[ex["true_label"]]
        pred_name = CLASS_NAMES[ex["pred_class"]]
        is_correct = ex["true_label"] == ex["pred_class"]
        accent     = "#2ecc71" if is_correct else "#e74c3c"
        probs      = ex["probs"]  # (7,)

        gs = gridspec.GridSpec(
            n, 4,
            figure=fig,
            hspace=0.08, wspace=0.05,
            left=0.02, right=0.98, top=0.96, bottom=0.02,
        )

        # Original image
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(ex["image"])
        ax0.set_title(
            f"True: {true_name}",
            fontsize=9.5, color="white", pad=4, fontweight="bold",
        )
        ax0.axis("off")
        for spine in ax0.spines.values():
            spine.set_edgecolor(accent); spine.set_linewidth(2.5)
        ax0.set_facecolor("#1a1a2e")

        # Raw GradCAM heatmap with colorbar 
        ax1 = fig.add_subplot(gs[row, 1])
        im = ax1.imshow(ex["cam"], cmap="turbo", vmin=0, vmax=1)
        ax1.set_title("Activation map", fontsize=9.5, color="white", pad=4)
        ax1.axis("off")
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.04)
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 0.5, 1])
        cb.ax.yaxis.set_tick_params(color="white", labelsize=7)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")
        cb.outline.set_edgecolor("white")

        # Overlay 
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(ex["overlay"])
        ax2.set_title(
            f"Pred: {pred_name}  {ex['confidence']:.1%}",
            fontsize=9.5, color=accent, pad=4, fontweight="bold",
        )
        ax2.axis("off")

        # Per-class probability bar chart 
        ax3 = fig.add_subplot(gs[row, 3])
        ax3.set_facecolor("#12122a")

        bar_colors = [
            CLASS_COLORS[i] if i == ex["pred_class"] else "#4a4a6a"
            for i in range(len(probs))
        ]
        bars = ax3.barh(
            range(len(probs)), probs,
            color=bar_colors, edgecolor="none", height=0.6,
        )
        # Highlight true label with a white outline
        bars[ex["true_label"]].set_edgecolor("white")
        bars[ex["true_label"]].set_linewidth(1.8)

        ax3.set_yticks(range(len(CLASS_LABELS)))
        ax3.set_yticklabels(CLASS_LABELS, fontsize=8, color="white")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("Probability", fontsize=8, color="#aaaacc")
        ax3.set_title("Class probabilities", fontsize=9.5, color="white", pad=4)
        ax3.tick_params(axis="x", colors="#aaaacc", labelsize=7)
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.spines["left"].set_color("#333355")
        ax3.spines["bottom"].set_color("#333355")
        ax3.invert_yaxis()

        # Value labels on bars
        for i, (bar, prob) in enumerate(zip(bars, probs)):
            if prob > 0.03:
                ax3.text(
                    min(prob + 0.01, 0.97), i,
                    f"{prob:.2f}", va="center", ha="left",
                    fontsize=7, color="white",
                )

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [gradcam] Saved: {os.path.basename(save_path)}")


def generate_gradcam_class_showcase(
    correct_pool: list,
    save_dir:     str,
    model_name:   str,
) -> None:
    """
    Compact showcase grid: for each class pick the single most confident
    correct prediction and render original + overlay side by side.
    Shows the model confidently and correctly identifying every lesion type.
    """
    if not correct_pool:
        return

    # Best (highest confidence) correct example per class
    best_per_class: dict[int, dict] = {}
    for ex in correct_pool:
        cls = ex["true_label"]
        if cls not in best_per_class or ex["confidence"] > best_per_class[cls]["confidence"]:
            best_per_class[cls] = ex

    classes    = sorted(best_per_class.keys())
    n_classes  = len(classes)
    if n_classes == 0:
        return

    # 2 columns per class: original | overlay
    fig, axes = plt.subplots(
        n_classes, 2,
        figsize=(8, 3.6 * n_classes),
        facecolor="#1a1a2e",
    )
    if n_classes == 1:
        axes = [axes]

    fig.suptitle(
        f"{model_name} — Best Correct Prediction per Class\n"
        f"(GradCAM highlights discriminative lesion regions)",
        fontsize=12, color="white", fontweight="bold", y=1.002,
    )

    for row, cls in enumerate(classes):
        ex        = best_per_class[cls]
        cls_name  = CLASS_NAMES[cls]
        cls_color = CLASS_COLORS[cls]

        axes[row][0].imshow(ex["image"])
        axes[row][0].set_title(
            f"{CLASS_LABELS[cls]} — {cls_name}",
            fontsize=10, color=cls_color, fontweight="bold", pad=5,
        )
        axes[row][0].axis("off")

        axes[row][1].imshow(ex["overlay"])
        axes[row][1].set_title(
            f"Confidence: {ex['confidence']:.1%}",
            fontsize=10, color="#2ecc71", fontweight="bold", pad=5,
        )
        axes[row][1].axis("off")

        for col in range(2):
            for spine in axes[row][col].spines.values():
                spine.set_edgecolor(cls_color)
                spine.set_linewidth(2)

    plt.tight_layout(pad=0.5)
    save_path = f"{save_dir}/{model_name}_gradcam_class_showcase.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [gradcam] Class showcase saved: {os.path.basename(save_path)}")


# Single-image GradCAM (Streamlit dashboard)

def run_gradcam_single(
    model:        nn.Module,
    target_layer: nn.Module,
    image_tensor: torch.Tensor,
    device:       torch.device,
    alpha:        float = 0.50,
) -> tuple:
    """
    Compute GradCAM for one image and return display-ready arrays.

    Returns:
        image_display: (H, W, 3) uint8 — denormalised original
        cam_overlay:   (H, W, 3) uint8 — GradCAM blended overlay
        pred_class:    int
        confidence:    float
        probs:         (7,) float32
    """
    gradcam      = GradCAM(model, target_layer)
    image_tensor = image_tensor.to(device)

    cam, pred_class, confidence, probs = gradcam.compute(image_tensor)
    image_display = tensor_to_display_image(image_tensor.cpu())
    cam_overlay   = overlay_cam(image_display, cam, alpha=alpha)

    gradcam.remove_hooks()
    return image_display, cam_overlay, pred_class, confidence, probs


# Aliases for dashboard/app.py compatibility
gradcam_single_image = run_gradcam_single
denormalize          = tensor_to_display_image
overlay_cam_on_image = overlay_cam
