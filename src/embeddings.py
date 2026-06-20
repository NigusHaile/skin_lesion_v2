"""
src/embeddings.py
=================
Feature extraction, dimensionality reduction (PCA → t-SNE), and visualisation
for trained model backbones.

Pipeline:
    features  = extract_features(model, loader)   # penultimate-layer activations
    pca_result = apply_pca(features)              # denoise + speed up t-SNE
    tsne_2d    = apply_tsne(pca_result)           # 2-D projection for scatter plots
"""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

from src.config import CFG
from src.dataset import CLASS_LABELS, CLASS_NAMES, CLASS_COLORS


# Feature extraction
@torch.no_grad()
def extract_features(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract penultimate-layer embeddings via model.get_features().

    AMP is enabled when CUDA is available to match training precision and
    avoid unnecessary dtype mismatches. Outputs are cast to float32 before
    returning so downstream sklearn code always receives a consistent dtype.

    Returns:
        features: (N, D) float32 array of backbone activations
        labels:   (N,)   int array of ground-truth class indices
    """
    model.eval()
    all_features, all_labels = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=torch.cuda.is_available()):
            feats = model.get_features(images)
        all_features.append(feats.float().cpu().numpy())
        all_labels.extend(labels.numpy())

    features = np.vstack(all_features)   # (N, D)
    labels   = np.array(all_labels)      # (N,)
    print(f"[embeddings] {model_name}: extracted {features.shape[0]} "
          f"features of dimension {features.shape[1]}")
    return features, labels


# Dimensionality reduction
def _sanitise_features(features: np.ndarray) -> np.ndarray:
    """
    Replace NaN/Inf values with 0 before scaling.

    These arise when a backbone layer outputs all-zero activations (dead
    neurons after ReLU), causing StandardScaler to divide by zero and
    produce NaN.  Logging the count helps diagnose backbone training issues.
    """
    n_bad = np.sum(~np.isfinite(features))
    if n_bad > 0:
        print(f"[embeddings] Warning: {n_bad} non-finite values replaced with 0. "
              f"Check backbone for dead neurons or exploding activations.")
        features = np.where(np.isfinite(features), features, 0.0)
    return features


def apply_pca(features: np.ndarray, n_components: int = 50) -> tuple:
    """
    StandardScaler → PCA, following the course lab pipeline.

    Reducing to 50 dims before t-SNE removes noise and cuts t-SNE runtime
    by ~30× (t-SNE is O(N² log N) in feature dimensionality).

    Robustness guards:
    - Non-finite values are replaced before scaling (see _sanitise_features).
    - Zero-variance columns (constant features) are dropped; StandardScaler
      would produce NaN for them after division by zero std.
    - n_components is clamped to what the sanitised matrix can support.

    Returns:
        pca_result: (N, n_components) — input for t-SNE
        pca_2d:     (N, 2)            — first two PCs for quick scatter plot
        pca_obj:    fitted PCA        — used for explained-variance plot
    """
    features = _sanitise_features(features)

    scaler          = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Drop constant-variance dimensions; they produce NaN after standardisation
    zero_var = np.where(scaler.var_ == 0)[0]
    if len(zero_var):
        print(f"[embeddings] Dropping {len(zero_var)} constant-variance dimensions before PCA.")
        features_scaled = np.delete(features_scaled, zero_var, axis=1)

    # Guard: n_components cannot exceed min(n_samples, n_features)
    n_components = min(n_components, *features_scaled.shape)

    pca        = PCA(n_components=n_components, random_state=CFG.project.random_seed)
    pca_result = pca.fit_transform(features_scaled)
    pca_2d     = pca_result[:, :2]

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[embeddings] PCA: top {n_components} components explain {explained:.1f}% variance")
    return pca_result, pca_2d, pca


def apply_tsne(pca_result: np.ndarray) -> np.ndarray:
    """
    t-SNE on PCA-reduced features.

    Parameters are read from CFG.embeddings so they can be swept in ablations
    without modifying this file.  perplexity=30 is a reliable default for
    datasets of ~1 500 samples; max_iter=1000 gives consistent convergence.

    Note: sklearn renamed n_iter → max_iter in v1.5. We detect which parameter
    name is accepted at runtime so the code works on both old and new installs.
    """
    import sklearn
    emb        = CFG.embeddings
    n_iter_val = emb.tsne_n_iter
    print(f"[embeddings] Running t-SNE "
          f"(perplexity={emb.tsne_perplexity}, max_iter={n_iter_val}) ...")

    # max_iter is the canonical name since sklearn 1.5; fall back to n_iter for < 1.5
    iter_kwarg = (
        "max_iter" if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 5)
        else "n_iter"
    )
    tsne = TSNE(
        n_components=2,
        perplexity=emb.tsne_perplexity,
        random_state=emb.tsne_random_state,
        n_jobs=-1,
        **{iter_kwarg: n_iter_val},
    )
    return tsne.fit_transform(pca_result)


# ── Visualisation helpers 
def _scatter_ax(ax, embedding, labels, title, xlabel, ylabel) -> None:
    """Shared scatter-plot logic used by both PCA and t-SNE static plots."""
    for idx, (label, name, color) in enumerate(zip(CLASS_LABELS, CLASS_NAMES, CLASS_COLORS)):
        mask = labels == idx
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=color, label=f"{label} ({mask.sum()})",
            alpha=0.65, s=12, edgecolors="none",
        )
    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.grid(alpha=0.3)


def plot_pca_explained_variance(pca_obj: PCA, save_dir: str, model_name: str) -> None:
    """
    Scree plot: per-component and cumulative explained variance.
    The 90 % threshold line helps identify the minimum useful component count.
    """
    var_ratio  = pca_obj.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)
    x          = range(1, len(var_ratio) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, var_ratio * 100, alpha=0.7, color="#3498db", label="Individual")

    ax2 = ax.twinx()
    ax2.plot(x, cumulative * 100, "r-o", markersize=3, label="Cumulative")
    ax2.axhline(90, color="gray", linestyle="--", alpha=0.5, label="90% threshold")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)", color="#3498db")
    ax2.set_ylabel("Cumulative (%)", color="red")
    ax.set_title(f"{model_name} — PCA Explained Variance")

    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels_ = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax2.legend(handles, labels_, loc="center right", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_pca_variance.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_embedding_scatter(
    embedding:  np.ndarray,
    labels:     np.ndarray,
    save_dir:   str,
    model_name: str,
    viz_type:   str,          # "PCA" or "t-SNE"
) -> None:
    """Save a static matplotlib scatter plot coloured by class."""
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_ax(
        ax, embedding, labels,
        title=f"{model_name} — {viz_type} Embedding",
        xlabel=f"{viz_type} dim 1",
        ylabel=f"{viz_type} dim 2",
    )
    plt.tight_layout()
    plt.savefig(f"{save_dir}/{model_name}_{viz_type.lower()}_scatter.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[embeddings] {viz_type} scatter saved.")


def build_plotly_scatter(
    embedding:  np.ndarray,
    labels:     np.ndarray,
    model_name: str,
    viz_type:   str = "t-SNE",
) -> go.Figure:
    """
    Interactive Plotly scatter for the Streamlit dashboard.
    Hover tooltip shows class name and sample index.
    """
    fig = go.Figure()
    for idx, (label, name, color) in enumerate(zip(CLASS_LABELS, CLASS_NAMES, CLASS_COLORS)):
        mask    = labels == idx
        indices = np.where(mask)[0]
        fig.add_trace(go.Scatter(
            x=embedding[mask, 0],
            y=embedding[mask, 1],
            mode="markers",
            name=f"{label} ({mask.sum()})",
            marker=dict(color=color, size=5, opacity=0.7),
            text=[f"<b>{name}</b><br>idx: {i}" for i in indices],
            hovertemplate="%{text}<extra></extra>",
        ))
    fig.update_layout(
        title=f"{model_name} — Interactive {viz_type} Embedding",
        xaxis_title=f"{viz_type} dim 1",
        yaxis_title=f"{viz_type} dim 2",
        height=580,
        hovermode="closest",
        legend=dict(title="Class", font=dict(size=10)),
    )
    return fig


# ── Full pipeline ─────────────────────────────────────────────────────────────

def run_embedding_analysis(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    save_dir:   str,
    model_name: str,
) -> dict:
    """
    End-to-end embedding pipeline: extract → PCA → t-SNE → plots → Plotly HTML.

    Saved artefacts (all under save_dir):
        {model_name}_labels.npy          — ground-truth labels for dashboard
        {model_name}_pca2d.npy           — (N, 2) PCA projection
        {model_name}_tsne2d.npy          — (N, 2) t-SNE projection
        {model_name}_pca_variance.png    — scree plot
        {model_name}_pca_scatter.png     — static PCA scatter
        {model_name}_t-sne_scatter.png   — static t-SNE scatter
        {model_name}_tsne_interactive.html — Plotly hover scatter

    Returns a dict with all arrays and the Plotly figure for dashboard reuse.
    """
    os.makedirs(save_dir, exist_ok=True)
    emb_cfg = CFG.embeddings

    features, labels = extract_features(model, loader, device, model_name)
    pca_result, pca_2d, pca_obj = apply_pca(features, n_components=emb_cfg.pca_n_components)
    tsne_2d = apply_tsne(pca_result)

    # Persist arrays so the dashboard can reload without re-running the pipeline
    np.save(f"{save_dir}/{model_name}_labels.npy", labels)
    np.save(f"{save_dir}/{model_name}_pca2d.npy",  pca_2d)
    np.save(f"{save_dir}/{model_name}_tsne2d.npy", tsne_2d)

    plot_pca_explained_variance(pca_obj, save_dir, model_name)
    plot_embedding_scatter(pca_2d,  labels, save_dir, model_name, viz_type="PCA")
    plot_embedding_scatter(tsne_2d, labels, save_dir, model_name, viz_type="t-SNE")

    plotly_fig = build_plotly_scatter(tsne_2d, labels, model_name)
    plotly_fig.write_html(f"{save_dir}/{model_name}_tsne_interactive.html")
    print(f"[embeddings] Interactive t-SNE HTML saved.")

    return {
        "features":   features,
        "labels":     labels,
        "pca_2d":     pca_2d,
        "tsne_2d":    tsne_2d,
        "plotly_fig": plotly_fig,
    }