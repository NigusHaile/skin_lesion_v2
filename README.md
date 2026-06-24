# DermiAI — Skin Lesion Diagnosis Platform

> Deep learning classification of dermoscopy images across 7 lesion types,
> with explainability (GradCAM), embedding analysis (PCA / t-SNE), ablation studies,
> and an interactive Streamlit dashboard.

**Live demo:** [skinlesionv2-5imnxbm4nqszvdayq6hcu2.streamlit.app](https://skinlesionv2-5imnxbm4nqszvdayq6hcu2.streamlit.app/)

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models](#models)
- [Metrics](#metrics)
- [Ablation Studies](#ablation-studies)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Dashboard](#dashboard)
- [Results](#results)

---

## Overview

DermiAI trains and compares three deep learning architectures on the
**HAM10000** dermoscopy dataset. The pipeline covers the full ML lifecycle:

| Stage | What happens |
|---|---|
| **Data** | Stratified 70/15/15 split · inverse-frequency class weights · heavy augmentation |
| **Training** | Per-model early stopping · best-checkpoint selection · AMP mixed precision |
| **Evaluation** | Balanced Accuracy · Precision · Macro F1 · ROC-AUC · confusion matrix · PR curves |
| **Explainability** | GradCAM heatmaps on correct *and* failure cases · per-class showcase grid |
| **Ablations** | 3 controlled studies per model (augmentation / class weighting / architecture) |
| **Embeddings** | PCA + t-SNE on test-set features for all models |
| **Dashboard** | Streamlit app — single diagnosis · batch prediction · ablation viewer · GradCAM |

---

## Project Structure

```
skin_lesion_v2/
├── configs/
│   └── config.yaml              # single source of truth for all hyperparameters
├── data/
│   ├── raw/                     # HAM10000 images + metadata CSV
│   └── splits/                  # train / val / test CSVs (auto-generated)
├── checkpoints/                 # best model weights per model
├── results/
│   ├── ablations/               # per-study JSON + PNG plots
│   ├── embeddings/              # PCA / t-SNE arrays + interactive HTML
│   ├── gradcam/                 # GradCAM grids (correct + incorrect + showcase)
│   └── plots/                   # training history curves
├── src/
│   ├── config.py                # YAML loader → dot-accessible CFG singleton
│   ├── dataset.py               # SkinLesionDataset, augmentation, class weights
│   ├── models.py                # SimpleCNN, ResNet50, ViTWithLoRA
│   ├── train.py                 # run_train_epoch, run_val_epoch, EarlyStopping
│   ├── evaluate.py              # evaluate_model, plot_training_history, build_comparison_table
│   ├── ablations.py             # all ablation study functions + summary table
│   ├── gradcam.py               # GradCAM class + batch/single visualisation
│   └── embeddings.py            # PCA + t-SNE feature extraction
├── dashboard/
│   └── app.py                   # Streamlit dashboard (6 pages)
├── notebooks/
│   └── train_all.ipynb          # end-to-end training & evaluation notebook
├── requirements.txt
└── README.md
```

---

## Dataset

**HAM10000** — Human Against Machine with 10000 training images
([Kaggle](https://www.kaggle.com/code/shashwatwork/skin-cancer-analyzer-streamlit-app/input))

| Code | Class | Risk |
|---|---|---|
| `nv` | Melanocytic nevi | Low |
| `mel` | Melanoma | **High** |
| `bkl` | Benign keratosis | Low |
| `bcc` | Basal cell carcinoma | Medium |
| `akiec` | Actinic keratosis | Medium |
| `vasc` | Vascular lesions | Low |
| `df` | Dermatofibroma | Low |

**10,015 images** — heavily imbalanced (`nv` = 67 %, `df` = 1.2 %).

### Handling class imbalance

Two complementary strategies are applied simultaneously:

1. **Weighted sampler** — over-samples minority classes during training so each
   batch has a balanced class distribution.
2. **Weighted CrossEntropy** — loss weight per class = `1 / class_frequency`,
   penalising errors on rare classes more heavily.

### Data splits

| Split | Size | Ratio |
|---|---|---|
| Train | 7,010 | 70 % |
| Val | 1,502 | 15 % |
| Test | 1,503 | 15 % |

Splits are stratified by class and saved to `data/splits/` on first run.

### Augmentation (training only)

Applied via [albumentations](https://albumentations.ai/):

- Horizontal & vertical flip (p = 0.5 each)
- Random rotation ±30° (p = 0.5)
- Color jitter — brightness, contrast, saturation, hue (p = 0.5)

---

## Models

### SimpleCNN — from-scratch baseline

4-block CNN built entirely from scratch. Establishes the performance floor.

```
Conv(3→32) → Conv(32→32) → MaxPool   ×
Conv(32→64) → Conv(64→64) → MaxPool  ×
Conv(64→128) → Conv(128→128) → MaxPool ×
Conv(128→256) → Conv(256→256) → MaxPool ×
GlobalAvgPool → Dropout(0.5) → FC(256→128) → Dropout(0.25) → FC(128→7)
```

### ResNet50 — pretrained baseline

ImageNet-pretrained ResNet50 from `timm` with the final FC replaced:

```
ResNet50 backbone (pretrained) → AdaptiveAvgPool → Dropout(0.3) → FC(2048→7)
```

### ViT-B/16 + LoRA — transformer model

Vision Transformer with **Low-Rank Adaptation (LoRA)** applied to all
attention projection matrices. Only LoRA parameters and the classification
head are trainable — the base ViT weights remain frozen.

```
LoRA: W' = W + (α/r) × B × A    where rank r=4, α=16
Trainable params: ~0.5 % of total ViT-B/16 parameters
```

---

## Metrics

Every epoch (train + val) tracks the following. All metrics are computed
over the **full epoch** (not per batch), ensuring stable estimates.

| Metric | Description |
|---|---|
| **Balanced Accuracy** | Mean recall per class — primary training signal & early stopping criterion |
| **Macro F1** | Unweighted mean F1 across all 7 classes |
| **Macro Precision** | Unweighted mean precision across all 7 classes |
| **ROC-AUC (macro, OvR)** | One-vs-Rest AUC averaged across classes — requires softmax probabilities |
| **Loss** | Weighted CrossEntropy |

Final test evaluation additionally reports:
- Weighted F1
- Per-class Precision / F1 / Recall / AUC
- Normalised confusion matrix
- Per-class ROC curves
- Precision-Recall curves
- Error analysis (top-5 confused pairs + confidence distribution)

---

## Ablation Studies

Three controlled studies per model — **one variable changed at a time**,
all else held constant (20 epochs, patience-5 early stopping).

| Model | Study 1 | Study 2 | Study 3 |
|---|---|---|---|
| **SimpleCNN** | Augmentation on/off | Class weighting on/off | 2 blocks ↔ 4 blocks |
| **ResNet50** | Augmentation on/off | Class weighting on/off | Frozen ↔ full fine-tuning |
| **ViT+LoRA** | Augmentation on/off | Class weighting on/off | LoRA rank=2 ↔ rank=8 |

Each study saves:
- JSON with `best_val_acc`, `best_val_f1`, `best_val_precision`, `best_val_roc_auc` per condition
- Per-epoch history for all metrics
- Comparison plot: Val Acc curves · Val F1 curves · 4-metric grouped bar chart

---

## Installation

### 1. Clone & create environment

```bash
git clone <repo-url>
cd skin_lesion_v2

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download HAM10000

Download from the link in the Dataset section above, or from the [https://www.kaggle.com/code/shashwatwork/skin-cancer-analyzer-streamlit-app/input].

Place files as follows:

```
data/raw/
├── HAM10000_metadata.csv
├── HAM10000_images_part1/   (or flat — any image layout works)
└── HAM10000_images_part2/
```

> The dataset loader searches recursively for image files matching the
> `image_id` field in the metadata CSV, so nested folder structures are fine.

---

## Quick Start

### Option A — Jupyter notebook (recommended)

```bash
jupyter notebook notebooks/train_all.ipynb
```

Run all cells in order. The notebook trains all three models, runs ablations,
generates GradCAM visualisations, and builds the comparison table.

To train only one model, set `SELECTED_MODEL` in the **Model Registry** cell:

```python
SELECTED_MODEL = "resnet50"   # "all" | "resnet50" | "vit" | "simple_cnn"
RUN_ABLATIONS  = False        # skip ablations if needed
```

### Option B — Import as library

```python
from src.config import CFG, get_device, seed_everything
from src.dataset import build_splits, get_dataloaders, compute_class_weights
from src.models import build_model
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_history

device = get_device(CFG)
seed_everything(CFG.project.random_seed)

train_df, val_df, test_df = build_splits(
    f"{CFG.paths.data_raw}/HAM10000_metadata.csv",
    CFG.paths.data_raw,
)
train_loader, val_loader, test_loader = get_dataloaders(
    train_df, val_df, test_df, use_weighted_sampler=True)
class_weights = compute_class_weights(train_df)

model   = build_model("resnet50", device)
history = train_model(model, train_loader, val_loader, class_weights, device, model_name="resnet50")

plot_training_history(history, "results/plots", "resnet50")
results = evaluate_model(model, test_loader, device, "results", "resnet50")
```

---

## Configuration

All hyperparameters live in **`configs/config.yaml`** and are loaded once
at import time as the module-level singleton `CFG`. No arguments to pass around.

### Key sections

```yaml
resnet50:
  epochs: 25
  lr: 1.0e-4

vit:
  epochs: 30
  lora_rank: 4
  lora_alpha: 16

simple_cnn:
  epochs: 30
  lr: 1.0e-3
```

Edit `config.yaml` to change epochs, learning rates, batch size, augmentation
parameters, or paths — no code changes required.

---

## Dashboard

**Live demo:** [skinlesionv2-5imnxbm4nqszvdayq6hcu2.streamlit.app](https://skinlesionv2-5imnxbm4nqszvdayq6hcu2.streamlit.app/)

```bash
streamlit run dashboard/app.py
```

Six pages:

| Page | Description |
|---|---|
| **Single Diagnosis** | Upload a dermoscopy image → prediction + confidence + top-3 differential |
| **Batch Prediction** | Upload many images → CSV export with all class probabilities |
| **Ablation Studies** | Interactive Plotly charts: bar charts + learning curves per study per model |
| **GradCAM Explainability** | Upload image → overlay heatmap showing which regions drove the prediction |
| **Model Comparison** | Test-set metrics table + bar charts + confusion matrices + ROC/PR curves |
| **Embedding Explorer** | PCA scatter or interactive t-SNE coloured by ground-truth class |

The dashboard auto-selects the best checkpoint per model (comparing base
training vs. all ablation checkpoints by val balanced accuracy).

### Skin image validation

Every uploaded image passes two automatic checks before inference:
1. **Minimum resolution** — must be at least 64 × 64 px
2. **Skin-tone pixel proportion** — Kovač et al. (2003) YCrCb range; images
   with < 20 % skin-coloured pixels are rejected with an explanation

---

## Results

> Results below are from a single training run on an RTX 5070 Ti Laptop GPU.

### Model comparison — test set

| Model | Balanced Acc | Macro F1 | Macro Precision | ROC-AUC |
|---|---|---|---|---|
| Simple CNN | 0.5560 | 0.2950 | 0.3541 | 0.8377 |
| ResNet50 | 0.7607 | 0.5698 | 0.5175 | 0.9185 |
| **ViT-B/16 + LoRA** | **0.7899** | **0.6701** | **0.6208** | **0.9504** |

GradCAM analysis confirms the model focuses on **lesion borders and texture**
rather than image artefacts (hair, ruler marks) for high-confidence predictions.

---

## Explainability — GradCAM

Three output grids are generated for each model:

| File | Contents |
|---|---|
| `{model}_gradcam_correct.png` | Top-N most confident correct predictions — 4 panels each (original · heatmap · overlay · class probability bar) |
| `{model}_gradcam_incorrect.png` | Failure cases with the same 4-panel layout for error analysis |
| `{model}_gradcam_class_showcase.png` | Best correct prediction per lesion class — one row per class |

---

## Citation / Acknowledgements

- **Dataset**: Tschandl, P., Rosendahl, C., Kittler, H. (2018). *The HAM10000 dataset*.
  Scientific Data, 5, 180161. [doi:10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)
- **LoRA**: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*.
- **GradCAM**: Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks*.
- Backbone weights provided by [timm](https://github.com/huggingface/pytorch-image-models).

---

*UNIMIB · Advanced Deep Learning 2026*
