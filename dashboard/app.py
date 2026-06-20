"""
dashboard/app.py  ·  DermiAI  ·  Skin Lesion Analysis Platform
Run:  streamlit run dashboard/app.py
"""
import sys
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import CFG, get_device
from src.dataset import (CLASS_LABELS, CLASS_NAMES, LABEL_TO_IDX, IDX_TO_LABEL,
                          RISK_LEVELS, CLASS_COLORS)
from src.models import build_model
from src.gradcam import run_gradcam_single

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="DermiAI · Skin Lesion Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ── TOKENS ─────────────────────────────────────── */
:root {
  --teal-700: #0f766e; --teal-600: #0d9488; --teal-500: #14b8a6;
  --teal-400: #2dd4bf; --teal-100: #ccfbf1; --teal-50: #f0fdfa;
  --navy:  #0f172a;  --navy-2: #1e293b;
  --gray-900: #111827; --gray-700: #374151; --gray-500: #6b7280;
  --gray-300: #d1d5db; --gray-200: #e5e7eb; --gray-100: #f3f4f6;
  --white: #ffffff;
  --red: #ef4444; --red-bg: #fef2f2;
  --amber: #f59e0b; --amber-bg: #fffbeb;
  --green: #10b981; --green-bg: #ecfdf5;
  --r: 12px; --r-lg: 16px;
  --sh: 0 1px 3px rgba(0,0,0,.07), 0 4px 12px rgba(0,0,0,.05);
  --sh-lg: 0 8px 24px rgba(0,0,0,.1), 0 2px 6px rgba(0,0,0,.06);
}

/* ── GLOBAL ──────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', system-ui, sans-serif !important;
  color: var(--gray-900) !important;
}
.stApp { background: #f1f5f9 !important; }
.block-container { padding: 2rem 2.5rem 5rem !important; max-width: 1380px; margin: 0 auto; }
#MainMenu, footer { visibility: hidden !important; }
[data-testid="stHeader"] { background: transparent !important; border-bottom: none !important; }
[data-testid="collapsedControl"] {
  display: flex !important; visibility: visible !important;
  pointer-events: auto !important; z-index: 100000 !important;
}

/* ── SIDEBAR ─────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--navy) 0%, var(--navy-2) 100%) !important;
  border-right: none !important;
  box-shadow: 4px 0 20px rgba(0,0,0,.25) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] hr {
  border-color: rgba(255,255,255,.08) !important;
  margin: 0.5rem 1rem !important;
}

/* ── SIDEBAR NAV BUTTONS ─────────────────────────── */
/* Strip default Streamlit button chrome inside the sidebar */
[data-testid="stSidebar"] .stButton { margin: 1px 0 !important; }
[data-testid="stSidebar"] .stButton > button {
  width: 100% !important;
  text-align: left !important;
  background: transparent !important;
  color: rgba(226,232,240,.8) !important;
  border: none !important;
  border-left: 3px solid transparent !important;
  border-radius: 8px !important;
  padding: 0.62rem 1rem !important;
  font-size: 0.875rem !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  transition: background .14s, color .14s, border-color .14s !important;
  line-height: 1.4 !important;
  white-space: nowrap !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(20,184,166,.15) !important;
  color: #ffffff !important;
  border-left-color: rgba(20,184,166,.5) !important;
}
/* Active page button — Streamlit sets data-testid="baseButton-primary" */
[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
  background: rgba(13,148,136,.3) !important;
  color: #ffffff !important;
  font-weight: 600 !important;
  border-left: 3px solid var(--teal-500) !important;
}
[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
  background: rgba(13,148,136,.45) !important;
}

/* ── SIDEBAR BRAND ───────────────────────────────── */
.sb-brand {
  padding: 1.4rem 1.25rem 1.1rem;
  border-bottom: 1px solid rgba(255,255,255,.07);
  margin-bottom: .5rem;
}
.sb-logo-row { display: flex; align-items: center; gap: .65rem; margin-bottom: .3rem; }
.sb-mark {
  width: 40px; height: 40px; border-radius: 11px;
  background: linear-gradient(135deg,#0d9488,#14b8a6);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem; box-shadow: 0 4px 14px rgba(13,148,136,.4);
  flex-shrink: 0;
}
.sb-name {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 1.4rem !important; font-weight: 700 !important;
  color: #f1f5f9 !important; letter-spacing: -.02em;
}
.sb-name em { color: var(--teal-400) !important; font-style: normal; }
.sb-tagline {
  font-size: 0.63rem !important; color: rgba(148,163,184,.7) !important;
  letter-spacing: .15em; text-transform: uppercase; padding-left: 48px;
}

/* ── SIDEBAR SECTION LABEL ───────────────────────── */
.sb-sect {
  font-size: 0.6rem !important; font-weight: 700 !important;
  letter-spacing: .15em; text-transform: uppercase;
  color: rgba(148,163,184,.55) !important;
  padding: 0 1rem; margin: .8rem 0 .3rem;
}

/* ── SIDEBAR MODEL CHIPS ─────────────────────────── */
.sb-chips { padding: 0 .75rem; display: flex; flex-direction: column; gap: .28rem; }
.sb-chip {
  display: flex; align-items: center; gap: .5rem;
  padding: .38rem .7rem;
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 7px;
}
.sb-dot { width: 6px; height: 6px; border-radius: 99px; flex-shrink: 0; }
.sb-chip-name { font-size: 0.74rem !important; color: rgba(203,213,225,.75) !important; font-weight: 500 !important; }
.sb-badge {
  margin-left: auto; font-size: 0.58rem !important; font-weight: 700 !important;
  padding: 2px 7px; border-radius: 99px;
  background: rgba(20,184,166,.18); color: var(--teal-400) !important;
  letter-spacing: .04em;
}

/* ── SIDEBAR STAT CARD ───────────────────────────── */
.sb-stat {
  margin: 0 .75rem .4rem; padding: .85rem 1rem;
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.07);
  border-radius: var(--r);
}
.sb-stat-head {
  font-size: 0.6rem !important; font-weight: 700 !important;
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--teal-400) !important; margin-bottom: .5rem;
}
.sb-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: .18rem 0; border-bottom: 1px solid rgba(255,255,255,.05);
}
.sb-row:last-child { border-bottom: none; }
.sb-key { font-size: 0.71rem !important; color: rgba(148,163,184,.5) !important; }
.sb-val { font-size: 0.71rem !important; color: rgba(226,232,240,.85) !important; font-weight: 600 !important; }

/* ── SIDEBAR FOOTER ──────────────────────────────── */
.sb-foot {
  padding: .75rem 1.25rem 1rem;
  border-top: 1px solid rgba(255,255,255,.06); margin-top: .5rem;
}
.sb-foot p { font-size: 0.63rem !important; color: rgba(148,163,184,.35) !important; text-align: center; margin: 0; }

/* ── PAGE HEADER BANNER ──────────────────────────── */
.ph-banner {
  background: linear-gradient(135deg, var(--navy) 0%, #1e3a5f 60%, #0f4c75 100%);
  border-radius: var(--r-lg); padding: 1.75rem 2rem;
  margin-bottom: 1.75rem; position: relative; overflow: hidden;
  box-shadow: var(--sh-lg);
}
.ph-banner::before {
  content: ''; position: absolute; right: -30px; top: -30px;
  width: 160px; height: 160px; border-radius: 99px;
  background: rgba(20,184,166,.12);
}
.ph-banner::after {
  content: ''; position: absolute; right: 80px; bottom: -40px;
  width: 100px; height: 100px; border-radius: 99px;
  background: rgba(20,184,166,.07);
}
.ph-icon-big {
  font-size: 2.5rem; margin-bottom: .5rem; display: block;
  filter: drop-shadow(0 2px 8px rgba(20,184,166,.5));
}
.ph-title {
  font-family: 'Playfair Display', Georgia, serif !important;
  font-size: 1.8rem !important; font-weight: 700 !important;
  color: #f1f5f9 !important; margin: 0; line-height: 1.15;
}
.ph-sub { font-size: 0.88rem !important; color: rgba(203,213,225,.75) !important; margin: .35rem 0 0; }

/* ── CARDS ───────────────────────────────────────── */
.card {
  background: var(--white); border-radius: var(--r-lg);
  border: 1px solid var(--gray-200); box-shadow: var(--sh);
  padding: 1.5rem; margin-bottom: 1rem;
}
.card-sm { padding: .9rem 1.1rem; }

/* ── STAT TILES ──────────────────────────────────── */
.tile {
  background: var(--white); border: 1px solid var(--gray-200);
  border-radius: var(--r-lg); padding: 1.25rem 1rem;
  text-align: center; box-shadow: var(--sh);
  transition: box-shadow .18s, transform .18s;
}
.tile:hover { box-shadow: var(--sh-lg); transform: translateY(-2px); }
.tile-val {
  font-family: 'Playfair Display', Georgia, serif;
  font-size: 2rem; font-weight: 700; color: var(--teal-600); line-height: 1;
}
.tile-lbl { font-size: 0.66rem; color: var(--gray-500); text-transform: uppercase; letter-spacing: .08em; margin-top: 5px; }

/* ── PREDICTION CARD ─────────────────────────────── */
.pred-card {
  border-radius: var(--r-lg); padding: 1.3rem 1.5rem;
  margin: .5rem 0 .8rem; border-left: 4px solid; position: relative; overflow: hidden;
}
.pred-card::after {
  content: ''; position: absolute; right: -16px; top: -16px;
  width: 72px; height: 72px; border-radius: 99px; background: currentColor; opacity: .05;
}
.pred-card.high   { background: var(--red-bg);   border-color: var(--red); }
.pred-card.medium { background: var(--amber-bg); border-color: var(--amber); }
.pred-card.low    { background: var(--green-bg); border-color: var(--green); }
.pred-name { font-family: 'Playfair Display', Georgia, serif; font-size: 1.45rem; font-weight: 700; margin: 0 0 3px; }
.pred-code { font-size: 0.72rem; font-family: monospace; color: var(--gray-500); }
.pred-badge {
  display: inline-flex; align-items: center; gap: 4px;
  font-size: 0.64rem; font-weight: 700; letter-spacing: .06em;
  text-transform: uppercase; padding: 3px 10px; border-radius: 99px; margin-top: 9px;
}
.badge-high   { background: var(--red);   color: #fff; }
.badge-medium { background: var(--amber); color: #fff; }
.badge-low    { background: var(--green); color: #fff; }

/* ── CONFIDENCE STRIP ────────────────────────────── */
.conf-strip {
  display: flex; align-items: center; gap: 1rem;
  padding: .8rem 1.1rem; background: var(--white);
  border: 1px solid var(--gray-200); border-radius: var(--r);
  margin-bottom: .75rem; box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.conf-bar-wrap { flex: 1; }
.conf-bar-track { height: 6px; border-radius: 99px; background: var(--gray-200); overflow: hidden; margin-top: 5px; }
.conf-bar-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg,var(--teal-600),var(--teal-400)); }
.conf-pct { font-family: 'Playfair Display', serif; font-size: 1.7rem; font-weight: 700; color: var(--teal-600); white-space: nowrap; }

/* ── RANK ROWS ───────────────────────────────────── */
.rank-row { display: flex; align-items: center; gap: .7rem; padding: .55rem 0; border-bottom: 1px solid var(--gray-100); }
.rank-row:last-child { border-bottom: none; }
.rank-medal { font-size: 1.1rem; flex-shrink: 0; }
.rank-name  { font-size: .875rem; font-weight: 500; flex: 1; }
.rank-code  { font-size: .68rem; color: var(--gray-400); background: var(--gray-100); border-radius: 4px; padding: 2px 6px; font-family: monospace; }
.rank-pct   { font-size: .875rem; font-weight: 700; color: var(--teal-600); }
.rank-bar   { height: 3px; border-radius: 99px; background: var(--gray-200); overflow: hidden; margin-top: 3px; }
.rank-fill  { height: 100%; border-radius: 99px; background: linear-gradient(90deg,var(--teal-600),var(--teal-400)); }

/* ── LABEL (UPPERCASE SECTION HEADING) ───────────── */
.lbl {
  font-size: .66rem; font-weight: 700; letter-spacing: .13em;
  text-transform: uppercase; color: var(--gray-400); margin: 0 0 .5rem; display: block;
}

/* ── EMPTY STATE ─────────────────────────────────── */
.empty { text-align: center; padding: 3.5rem 2rem; }
.empty-icon  { font-size: 2.8rem; margin-bottom: .7rem; }
.empty-title { font-family: 'Playfair Display', serif; font-size: 1.2rem; font-weight: 700; color: var(--gray-700); margin-bottom: .4rem; }
.empty-body  { font-size: .84rem; color: var(--gray-500); line-height: 1.65; max-width: 340px; margin: 0 auto; }

/* ── FILE UPLOADER ───────────────────────────────── */
[data-testid="stFileUploader"] {
  border: 2px dashed #2dd4bf !important;
  border-radius: var(--r-lg) !important;
  background: var(--teal-50) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--teal-500) !important; }

/* ── WIDGETS ─────────────────────────────────────── */
div[data-testid="stSelectbox"] > div { border-radius: var(--r) !important; }
div[data-testid="stSelectbox"] > div:focus-within {
  border-color: var(--teal-500) !important;
  box-shadow: 0 0 0 3px rgba(20,184,166,.15) !important;
}
.stProgress > div > div > div { background: linear-gradient(90deg,var(--teal-600),var(--teal-400)) !important; border-radius: 99px !important; }
.stProgress > div > div { border-radius: 99px !important; background: var(--teal-100) !important; }

/* ── TABS ────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent; border-bottom: 2px solid var(--gray-200); }
.stTabs [data-baseweb="tab"] {
  border-radius: var(--r) var(--r) 0 0; padding: .55rem 1.25rem;
  font-weight: 500; font-size: .85rem; color: var(--gray-500) !important;
}
.stTabs [aria-selected="true"] {
  background: var(--white) !important; color: var(--teal-700) !important;
  font-weight: 600 !important; border-bottom: 2.5px solid var(--teal-600) !important;
}

/* ── MISC ────────────────────────────────────────── */
.stSpinner > div { border-top-color: var(--teal-500) !important; }
.stDataFrame { border-radius: var(--r) !important; }
.stAlert { border-radius: var(--r) !important; }
.streamlit-expanderHeader { background: var(--gray-100) !important; border-radius: var(--r) !important; font-weight: 500 !important; }
.stDownloadButton button {
  background: linear-gradient(135deg,var(--teal-700),var(--teal-500)) !important;
  color: #fff !important; border: none !important; border-radius: var(--r) !important;
  font-weight: 600 !important; padding: .55rem 1.5rem !important;
  box-shadow: 0 2px 8px rgba(13,148,136,.35) !important;
}
hr { border-color: var(--gray-200) !important; }
.info-box {
  background: var(--teal-50); border: 1px solid var(--teal-100); border-radius: var(--r);
  padding: .75rem 1rem; font-size: .81rem; color: var(--teal-700);
  display: flex; align-items: flex-start; gap: .5rem; margin-top: .75rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────
RISK_CSS   = {"High": "high",   "Medium": "medium",   "Low": "low"}
BADGE_CSS  = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}
RISK_ICON  = {"High": "⚠️",  "Medium": "⚡",  "Low": "✅"}
MODEL_OPTS   = ["EfficientNet-B3 (recommended)", "ViT-B/16 + LoRA", "ResNet50", "Simple CNN"]
GRADCAM_OPTS = ["EfficientNet-B3", "ResNet50", "Simple CNN"]   # CNN-based (ViT has no conv layer)
TEAL_SEQ     = ["#0d9488", "#14b8a6", "#2dd4bf", "#5eead4"]

EMBED_KEYS = {"EfficientNet-B3": "efficientnet", "ViT+LoRA": "vit",
              "ResNet50": "resnet50", "Simple CNN": "simple_cnn"}

# ── Helpers ────────────────────────────────────────────────────
def page_banner(icon: str, title: str, sub: str = "") -> None:
    sub_html = f'<p class="ph-sub">{sub}</p>' if sub else ""
    st.markdown(f"""
    <div class="ph-banner">
      <span class="ph-icon-big">{icon}</span>
      <h1 class="ph-title">{title}</h1>
      {sub_html}
    </div>""", unsafe_allow_html=True)

def empty_card(icon: str, title: str, body: str) -> None:
    st.markdown(f"""
    <div class="card">
      <div class="empty">
        <div class="empty-icon">{icon}</div>
        <div class="empty-title">{title}</div>
        <div class="empty-body">{body}</div>
      </div>
    </div>""", unsafe_allow_html=True)

def stat_tiles(items: list) -> None:
    """items = [(value, label, color)] — renders a row of stat tiles."""
    cols = st.columns(len(items), gap="medium")
    for col, (val, lbl, clr) in zip(cols, items):
        col.markdown(
            f'<div class="tile"><div class="tile-val" style="color:{clr};">{val}</div>'
            f'<div class="tile-lbl">{lbl}</div></div>',
            unsafe_allow_html=True)

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    img = np.array(pil_image.convert("RGB"))
    t   = A.Compose([
        A.Resize(CFG.data.image_size, CFG.data.image_size),
        A.Normalize(mean=CFG.data.imagenet_mean, std=CFG.data.imagenet_std),
        ToTensorV2(),
    ])
    return t(image=img)["image"]

def _validate_skin_image(pil_img: Image.Image) -> tuple:
    """
    Returns (is_valid: bool, reason: str).

    Two checks:
    1. Minimum resolution — dermoscopy images are never thumbnails.
    2. Skin-tone pixel proportion via YCrCb colour space.
       The Kovac et al. (2003) skin range (Cr 133-173, Cb 77-127) covers
       all skin tones well enough for close-up dermoscopy shots.
       Dermoscopy images: typically 60-95 % skin pixels.
       Non-skin photos (houses, landscapes, food …): typically < 20 %.
    """
    img_np = np.array(pil_img.convert("RGB"))
    h, w   = img_np.shape[:2]

    if h < 64 or w < 64:
        return False, f"Image resolution ({w}×{h}) is too small for analysis. Please upload a proper dermoscopy photo."

    img_ycrcb = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    Cr = img_ycrcb[:, :, 1].astype(np.int32)
    Cb = img_ycrcb[:, :, 2].astype(np.int32)
    skin_mask  = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127))
    skin_ratio = float(skin_mask.mean())

    if skin_ratio < 0.20:
        pct = f"{skin_ratio:.0%}"
        return False, (
            f"This image does not appear to contain skin tissue "
            f"({pct} skin-coloured pixels detected). "
            "Please upload a close-up dermoscopy photograph of a skin lesion."
        )

    return True, ""


def _skin_error_ui(reason: str) -> None:
    """Render a styled rejection card."""
    st.markdown(f"""
    <div style="background:#fef2f2;border:1.5px solid #fca5a5;border-radius:12px;
                padding:1.25rem 1.5rem;display:flex;gap:1rem;align-items:flex-start;">
      <span style="font-size:1.8rem;line-height:1;">🚫</span>
      <div>
        <div style="font-weight:700;color:#b91c1c;font-size:0.95rem;margin-bottom:.35rem;">
          Not a skin lesion image
        </div>
        <div style="color:#7f1d1d;font-size:0.85rem;line-height:1.55;">{reason}</div>
        <div style="margin-top:.7rem;font-size:0.78rem;color:#991b1b;">
          ⚕️ DermiAI is trained exclusively on dermoscopy images of skin lesions.
          Submitting unrelated images produces meaningless results.
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def _strip(sd: dict) -> dict:
    return {k.removeprefix("_orig_mod."): v for k, v in sd.items()} \
           if any(k.startswith("_orig_mod.") for k in sd) else sd

# ── Model loaders ──────────────────────────────────────────────
@st.cache_resource
def _load(model_key: str, ckpt_rel: str):
    dev = get_device(CFG)
    m   = build_model(model_key, dev)
    p   = Path(CFG.paths.checkpoints) / ckpt_rel
    if p.exists():
        m.load_state_dict(_strip(torch.load(str(p), map_location=dev, weights_only=True)))
        m.eval()
    else:
        st.warning(f"Checkpoint not found: {p}")
    return m, dev

def load_efficientnet(): return _load("efficientnet", "efficientnet/final_best.pth")
def load_vit():          return _load("vit",          "vit/best.pth")
def load_resnet():       return _load("resnet50",     "resnet50/best.pth")
def load_simple_cnn():   return _load("simple_cnn",   "simple_cnn/best.pth")

def pick_model(choice: str):
    if "EfficientNet" in choice: return load_efficientnet()
    if "ViT"          in choice: return load_vit()
    if "Simple CNN"   in choice: return load_simple_cnn()
    return load_resnet()

def pick_gradcam_model(choice: str):
    """GradCAM only works on CNN-based models that expose get_gradcam_layer()."""
    if "EfficientNet" in choice: return load_efficientnet()
    if "ResNet"       in choice: return load_resnet()
    return load_simple_cnn()

# ── Sidebar ────────────────────────────────────────────────────
PAGE_OPTIONS = [
    "🩺  Single Diagnosis",       # [0] → render_single_diagnosis
    "📦  Batch Prediction",        # [1] → render_batch
    "🔬  Ablation Studies",        # [2] → render_ablations
    "🌡️  GradCAM Explainability", # [3] → render_gradcam
    "📊  Model Comparison",        # [4] → render_comparison
    "🔵  Embedding Explorer",      # [5] → render_embeddings
]

# ── Session-state navigation (reliable across all Streamlit versions) ──
if "page" not in st.session_state:
    st.session_state.page = PAGE_OPTIONS[0]

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sb-brand">
      <div class="sb-logo-row">
        <div class="sb-mark">🩺</div>
        <div class="sb-name">Dermi<em>AI</em></div>
      </div>
      <div class="sb-tagline">Skin Lesion Analysis Platform</div>
    </div>""", unsafe_allow_html=True)

    # Navigation — button per page (session_state tracks active page)
    st.markdown('<p class="sb-sect">Navigation</p>', unsafe_allow_html=True)
    for _opt in PAGE_OPTIONS:
        _is_active = st.session_state.page == _opt
        if st.button(_opt, key=f"nav_{_opt}", use_container_width=True,
                     type="primary" if _is_active else "secondary"):
            st.session_state.page = _opt
            st.rerun()
    selected_page = st.session_state.page

    st.divider()

    # Model chips
    st.markdown('<p class="sb-sect">Models</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-chips">
      <div class="sb-chip"><div class="sb-dot" style="background:#14b8a6;"></div>
        <span class="sb-chip-name">EfficientNet-B3</span><span class="sb-badge">Primary</span></div>
      <div class="sb-chip"><div class="sb-dot" style="background:#818cf8;"></div>
        <span class="sb-chip-name">ViT-B/16 + LoRA</span><span class="sb-badge">Transformer</span></div>
      <div class="sb-chip"><div class="sb-dot" style="background:#60a5fa;"></div>
        <span class="sb-chip-name">ResNet50</span><span class="sb-badge">Baseline</span></div>
      <div class="sb-chip"><div class="sb-dot" style="background:#9ca3af;"></div>
        <span class="sb-chip-name">Simple CNN</span><span class="sb-badge">Scratch</span></div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    # Dataset stats
    st.markdown('<p class="sb-sect">Dataset</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sb-stat">
      <div class="sb-stat-head">HAM10000</div>
      <div class="sb-row"><span class="sb-key">Total images</span><span class="sb-val">10,015</span></div>
      <div class="sb-row"><span class="sb-key">Classes</span><span class="sb-val">7 lesion types</span></div>
      <div class="sb-row"><span class="sb-key">Split</span><span class="sb-val">70 / 15 / 15 %</span></div>
      <div class="sb-row"><span class="sb-key">Explainability</span><span class="sb-val">GradCAM</span></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-foot">
      <p>UNIMIB · Advanced Deep Learning 2026</p>
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — Single Diagnosis
# ═══════════════════════════════════════════════════════════════
def render_single_diagnosis():
    page_banner("🩺", "Single Diagnosis",
                "Upload a dermoscopy image for AI-assisted lesion classification")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.markdown('<span class="lbl">Dermoscopy Image</span>', unsafe_allow_html=True)
        uploaded = st.file_uploader("img", type=["jpg","jpeg","png"],
                                    label_visibility="collapsed")
        st.markdown('<span class="lbl" style="margin-top:.8rem;display:block;">Model</span>',
                    unsafe_allow_html=True)
        model_choice = st.selectbox("mdl", MODEL_OPTS, label_visibility="collapsed")
        if uploaded:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption="Uploaded image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        if not uploaded:
            empty_card("🔬", "Ready for analysis",
                       "Upload a dermoscopy image on the left. DermiAI will return a "
                       "diagnosis, confidence score, and top-3 differential with risk level.")
            return

        valid, reason = _validate_skin_image(pil_img)
        if not valid:
            _skin_error_ui(reason)
            return

        with st.spinner("Running inference…"):
            tensor = preprocess(pil_img)
            model, device = pick_model(model_choice)
            with torch.no_grad():
                probs = torch.softmax(
                    model(tensor.unsqueeze(0).to(device)).float(), dim=1)[0].cpu().numpy()
            pred_idx = int(probs.argmax())

        lbl  = CLASS_LABELS[pred_idx]
        name = CLASS_NAMES[pred_idx]
        risk = RISK_LEVELS[lbl]
        conf = float(probs[pred_idx])

        st.markdown(f"""
        <div class="pred-card {RISK_CSS.get(risk,'low')}">
          <div class="pred-name">{name}</div>
          <div class="pred-code">Code: {lbl}</div>
          <span class="pred-badge {BADGE_CSS.get(risk,'badge-low')}">{RISK_ICON.get(risk,'')} {risk} Risk</span>
        </div>""", unsafe_allow_html=True)

        fill = int(conf * 100)
        st.markdown(f"""
        <div class="conf-strip">
          <div class="conf-bar-wrap">
            <span class="lbl" style="margin:0;">Confidence</span>
            <div class="conf-bar-track"><div class="conf-bar-fill" style="width:{fill}%"></div></div>
          </div>
          <div class="conf-pct">{conf:.0%}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.markdown('<span class="lbl">Top-3 Differential</span>', unsafe_allow_html=True)
        for i, idx in enumerate(np.argsort(probs)[::-1][:3]):
            pct  = float(probs[idx])
            fill = int(pct * 100)
            medal = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
            st.markdown(f"""
            <div class="rank-row">
              <span class="rank-medal">{medal}</span>
              <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:.5rem;flex-wrap:wrap;">
                  <span class="rank-name">{CLASS_NAMES[idx]}</span>
                  <span class="rank-code">{CLASS_LABELS[idx]}</span>
                  <span class="rank-pct" style="margin-left:auto;">{pct:.1%}</span>
                </div>
                <div class="rank-bar"><div class="rank-fill" style="width:{fill}%"></div></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
          ⚕️ <span><b>Research use only.</b> Not a substitute for clinical diagnosis by a qualified dermatologist.</span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — GradCAM Explainability
# ═══════════════════════════════════════════════════════════════
def render_gradcam():
    page_banner("🌡️", "GradCAM Explainability",
                "Visualise which image regions drove the model's prediction")

    ctl_a, ctl_b, ctl_c = st.columns([2, 1, 1])
    with ctl_a:
        uploaded = st.file_uploader("gcam-img", type=["jpg","jpeg","png"],
                                    label_visibility="collapsed")
    with ctl_b:
        st.markdown('<span class="lbl">Model</span>', unsafe_allow_html=True)
        gcam_choice = st.selectbox("gcam-mdl", GRADCAM_OPTS, label_visibility="collapsed")
    with ctl_c:
        st.markdown('<span class="lbl">Heatmap opacity</span>', unsafe_allow_html=True)
        alpha = st.slider("alpha", 0.1, 0.9, 0.45, 0.05, label_visibility="collapsed")

    if not uploaded:
        empty_card("🌡️", "No image uploaded",
                   "Upload a dermoscopy image above to overlay a gradient heatmap showing "
                   "which regions drove the model's classification. "
                   "ViT is excluded here — GradCAM requires convolutional feature maps.")
        return

    pil_img = Image.open(uploaded).convert("RGB")

    valid, reason = _validate_skin_image(pil_img)
    if not valid:
        _skin_error_ui(reason)
        return

    tensor = preprocess(pil_img)

    with st.spinner("Computing GradCAM…"):
        model, device = pick_gradcam_model(gcam_choice)
        gm = getattr(model, "_orig_mod", model)
        img_disp, cam_overlay, pred_idx, conf, probs = run_gradcam_single(
            gm, gm.get_gradcam_layer(), tensor, device, alpha=alpha)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.markdown('<span class="lbl">Original Image</span>', unsafe_allow_html=True)
        st.image(img_disp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.markdown('<span class="lbl">GradCAM Overlay</span>', unsafe_allow_html=True)
        st.image(cam_overlay, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    lbl  = CLASS_LABELS[pred_idx]
    name = CLASS_NAMES[pred_idx]
    risk = RISK_LEVELS[lbl]
    st.markdown(f"""
    <div class="card card-sm">
      <span class="lbl">Prediction</span>
      <div class="pred-card {RISK_CSS.get(risk,'low')}" style="margin:.4rem 0 .6rem;">
        <div class="pred-name" style="font-size:1.1rem;">{name}</div>
        <div class="pred-code">{lbl}</div>
        <span class="pred-badge {BADGE_CSS.get(risk,'badge-low')}">{RISK_ICON.get(risk,'')} {risk} Risk</span>
      </div>
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <span class="lbl" style="margin:0;">Confidence</span>
        <span style="font-family:'Playfair Display',serif;font-size:1.5rem;font-weight:700;color:#0d9488;">{conf:.1%}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("ℹ️ How to interpret GradCAM"):
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Colour scale**\n- 🔴 Red/yellow = high importance\n- 🔵 Blue/green = low importance")
        with cb:
            st.markdown("**Clinical check**\n- ✅ Heat on lesion body/border = reliable\n- ⚠️ Heat on hair or background = artefact bias")


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — Model Comparison
# ═══════════════════════════════════════════════════════════════
def render_comparison():
    page_banner("📊", "Model Comparison",
                "All four architectures benchmarked on the held-out test set")

    res = Path(CFG.paths.results)
    csv_path = res / "model_comparison.csv"

    if not csv_path.exists():
        st.warning("No results found. Run `python train_all.py` first.")
        st.markdown("**Demo values (placeholder):**")
        st.dataframe(pd.DataFrame({
            "Model":           ["Simple CNN","ResNet50","EfficientNet-B3","ViT + LoRA"],
            "Balanced Acc":    ["0.612","0.761","0.872","0.843"],
            "Macro F1":        ["0.589","0.748","0.858","0.831"],
            "ROC-AUC (macro)": ["0.891","0.938","0.972","0.965"],
        }), use_container_width=True)
        return

    df   = pd.read_csv(csv_path)
    best = df.loc[df["ROC-AUC (macro)"].astype(float).idxmax()]

    stat_tiles([
        (f"{float(best['ROC-AUC (macro)']):.3f}", "Best ROC-AUC",      "#0d9488"),
        (f"{df['Balanced Acc'].astype(float).max():.3f}", "Best Balanced Acc", "#0d9488"),
        (f"{df['Macro F1'].astype(float).max():.3f}",     "Best Macro F1",     "#0d9488"),
        (str(len(df)),                                     "Models Tested",     "#6b7280"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["📋 Metrics Table", "📈 Bar Charts",
                    "🗂️ Confusion Matrices", "📉 Training Curves",
                    "📐 ROC Curves", "📉 PR Curves"])

    with tabs[0]:
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.dataframe(
            df.style
              .highlight_max(subset=["Balanced Acc","Macro F1","ROC-AUC (macro)"],
                             color="#ccfbf1")
              .format(precision=4),
            use_container_width=True, height=260)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[1]:
        for col, metric in zip(st.columns(3, gap="medium"),
                               ["Balanced Acc","Macro F1","ROC-AUC (macro)"]):
            with col:
                fig = px.bar(df, x="Model", y=df[metric].astype(float),
                             title=metric, text_auto=".3f", color="Model",
                             color_discrete_sequence=TEAL_SEQ, height=320)
                fig.update_layout(showlegend=False, yaxis_range=[0,1],
                                  plot_bgcolor="white", paper_bgcolor="white",
                                  font_family="Inter", title_font_size=12,
                                  margin=dict(t=40,b=50,l=16,r=16))
                fig.update_traces(marker_line_width=0)
                st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        model_names = ["simple_cnn","resnet50","vit","efficientnet"]
        disp_names  = ["Simple CNN","ResNet50","ViT+LoRA","EfficientNet-B3"]
        cols = st.columns(4, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / f"{mname}_confusion_matrix.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(f"{dname}: not yet trained")

    with tabs[3]:
        cols = st.columns(4, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / "plots" / f"{mname}_training_history.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info(f"{dname}: not yet trained")

    with tabs[4]:
        cols = st.columns(4, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / f"{mname}_roc_curves.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    with tabs[5]:
        cols = st.columns(4, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / f"{mname}_pr_curves.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — Embedding Explorer
# ═══════════════════════════════════════════════════════════════
def render_embeddings():
    page_banner("🔵", "Embedding Explorer",
                "Each dot = one test image · colour = ground-truth class label")

    ca, cb, _ = st.columns([1,1,2])
    with ca:
        mc = st.selectbox("Embedding model", ["EfficientNet-B3","ViT+LoRA","ResNet50"])
    with cb:
        vt = st.radio("Projection", ["t-SNE","PCA"], horizontal=True)

    mkey    = EMBED_KEYS[mc]
    emb_dir = Path(CFG.paths.results) / "embeddings"

    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)

    if vt == "t-SNE":
        hp = emb_dir / f"{mkey}_tsne_interactive.html"
        if hp.exists():
            with open(hp) as f:
                st.components.v1.html(f.read(), height=650, scrolling=True)
        else:
            st.info("Interactive t-SNE not found. Run `python train_all.py` to generate it.")
    else:
        pp = emb_dir / f"{mkey}_pca2d.npy"
        lp = emb_dir / f"{mkey}_labels.npy"
        if pp.exists() and lp.exists():
            emb = np.load(str(pp))
            lbs = np.load(str(lp))
            fig = go.Figure()
            for i, (lb, nm, col) in enumerate(zip(CLASS_LABELS, CLASS_NAMES, CLASS_COLORS)):
                mask = lbs == i
                fig.add_trace(go.Scatter(
                    x=emb[mask,0], y=emb[mask,1], mode="markers",
                    name=f"{lb} ({mask.sum()})",
                    marker=dict(color=col, size=5, opacity=.85,
                                line=dict(width=.4, color="white"))))
            fig.update_layout(
                title=f"{mc} — PCA Embedding", height=620,
                plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
                legend=dict(title="Class", font_size=11, bgcolor="rgba(255,255,255,.9)",
                            bordercolor="#e5e7eb", borderwidth=1),
                xaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False),
                yaxis=dict(showgrid=True, gridcolor="#f3f4f6", zeroline=False))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("PCA data not found. Run `python train_all.py` first.")

    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ PCA vs t-SNE"):
        ea, eb = st.columns(2)
        with ea:
            st.markdown("""**PCA** (linear)\n- Preserves global distances\n- Fast & deterministic""")
        with eb:
            st.markdown("""**t-SNE** (non-linear)\n- Reveals local clusters\n- Best for class separation""")


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — Batch Prediction
# ═══════════════════════════════════════════════════════════════
def render_batch():
    page_banner("📦", "Batch Prediction",
                "Process multiple images at once · export results as CSV")

    ca, cb = st.columns([2,1])
    with ca:
        st.markdown('<span class="lbl">Upload Images</span>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader("batch", type=["jpg","jpeg","png"],
                                          accept_multiple_files=True,
                                          label_visibility="collapsed")
    with cb:
        st.markdown('<span class="lbl">Model</span>', unsafe_allow_html=True)
        model_choice = st.selectbox("bmdl", MODEL_OPTS, label_visibility="collapsed")

    if not uploaded_files:
        empty_card("📦", "Ready for batch processing",
                   "Upload one or more dermoscopy images above. Results include predictions, "
                   "confidence scores, risk levels, and per-class probabilities — all exportable as CSV.")
        return

    model, device = pick_model(model_choice)
    model.eval()
    rows     = []
    skipped  = []
    pb       = st.progress(0)
    stxt     = st.empty()

    for i, f in enumerate(uploaded_files):
        stxt.caption(f"Processing {i+1}/{len(uploaded_files)}: {f.name}")
        img = Image.open(f).convert("RGB")

        valid, reason = _validate_skin_image(img)
        if not valid:
            skipped.append((f.name, reason))
            pb.progress((i + 1) / len(uploaded_files))
            continue

        tensor = preprocess(img)
        with torch.no_grad():
            probs = torch.softmax(
                model(tensor.unsqueeze(0).to(device)).float(), dim=1)[0].cpu().numpy()
        pi  = int(probs.argmax())
        row = {"filename": f.name,
               "predicted_label": CLASS_LABELS[pi],
               "predicted_class": CLASS_NAMES[pi],
               "confidence":      f"{probs[pi]:.4f}",
               "risk_level":      RISK_LEVELS[CLASS_LABELS[pi]]}
        for j, lb in enumerate(CLASS_LABELS):
            row[f"prob_{lb}"] = f"{probs[j]:.4f}"
        rows.append(row)
        pb.progress((i + 1) / len(uploaded_files))

    if skipped:
        with st.expander(f"⚠️ {len(skipped)} file(s) skipped — not skin lesion images",
                         expanded=True):
            for fname, rsn in skipped:
                st.markdown(
                    f'<div style="padding:.35rem 0;border-bottom:1px solid #fee2e2;">'
                    f'<b style="color:#b91c1c;">{fname}</b>'
                    f'<span style="color:#7f1d1d;font-size:.83rem;margin-left:.6rem;">{rsn}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

    if not rows:
        st.error("No valid skin lesion images were found in the upload. "
                 "Please upload close-up dermoscopy photographs.")
        return

    stxt.success(f"✅ Done — {len(rows)} image(s) processed"
                 + (f", {len(skipped)} skipped" if skipped else ""))
    rdf = pd.DataFrame(rows)

    hn = (rdf["risk_level"] == "High").sum()
    mn = (rdf["risk_level"] == "Medium").sum()
    ln = (rdf["risk_level"] == "Low").sum()
    st.markdown("<br>", unsafe_allow_html=True)
    stat_tiles([
        (len(rdf), "Processed",   "#0d9488"),
        (hn,       "High Risk",   "#ef4444"),
        (mn,       "Medium Risk", "#f59e0b"),
        (ln,       "Low Risk",    "#10b981"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    t1, t2 = st.tabs(["📋 Results Table", "📊 Distribution"])

    with t1:
        def highlight_risk(v):
            return ("background:#fef2f2;color:#b91c1c" if v == "High" else
                    "background:#fffbeb;color:#92400e" if v == "Medium" else
                    "background:#ecfdf5;color:#065f46")
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        st.dataframe(
            rdf[["filename","predicted_label","predicted_class","confidence","risk_level"]]
               .style.map(highlight_risk, subset=["risk_level"]),
            use_container_width=True, height=420)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("📥 Download full CSV",
                           rdf.to_csv(index=False).encode(),
                           "dermai_predictions.csv", "text/csv")

    with t2:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            cc  = rdf["predicted_class"].value_counts()
            fig = px.pie(names=cc.index, values=cc.values,
                         title="Class distribution",
                         color_discrete_sequence=CLASS_COLORS, hole=0.4)
            fig.update_layout(font_family="Inter", paper_bgcolor="white",
                              margin=dict(t=44,b=16,l=16,r=16))
            fig.update_traces(textinfo="percent+label", textfont_size=11)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            rc  = rdf["risk_level"].value_counts()
            fig = px.bar(x=rc.index, y=rc.values, color=rc.index, text_auto=True,
                         color_discrete_map={"High":"#ef4444","Medium":"#f59e0b","Low":"#10b981"},
                         title="Risk breakdown",
                         labels={"x":"Risk Level","y":"Count"})
            fig.update_layout(showlegend=False, font_family="Inter",
                              paper_bgcolor="white", plot_bgcolor="white",
                              margin=dict(t=44,b=40,l=16,r=16))
            fig.update_traces(marker_line_width=0, textfont_size=12)
            st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 6 — Ablation Studies
# ═══════════════════════════════════════════════════════════════
_ABL_PLOTS = {
    "EfficientNet-B3": [
        ("study1_augmentation.png",  "Study 1 — Data Augmentation"),
        ("study2_class_weights.png", "Study 2 — Class Weighting"),
        ("study3_backbone.png",      "Study 3 — Backbone Freezing"),
    ],
    "SimpleCNN": [
        ("simplecnn_study1_augmentation.png",  "Study 1 — Data Augmentation"),
        ("simplecnn_study2_class_weights.png", "Study 2 — Class Weighting"),
        ("simplecnn_study3_depth.png",         "Study 3 — Network Depth"),
    ],
    "ResNet50": [
        ("resnet50_study1_augmentation.png",  "Study 1 — Data Augmentation"),
        ("resnet50_study2_class_weights.png", "Study 2 — Class Weighting"),
        ("resnet50_study3_backbone.png",      "Study 3 — Backbone Freezing"),
    ],
    "ViT+LoRA": [
        ("vit_study1_augmentation.png",  "Study 1 — Data Augmentation"),
        ("vit_study2_class_weights.png", "Study 2 — Class Weighting"),
        ("vit_study3_lora_rank.png",     "Study 3 — LoRA Rank"),
    ],
}

def render_ablations():
    page_banner("🔬", "Ablation Studies",
                "One variable changed at a time — quantifying each design choice")

    abl_dir  = Path(CFG.paths.results) / "ablations"
    csv_path = abl_dir / "ablation_summary.csv"

    if not csv_path.exists():
        empty_card("🔬", "No ablation results yet",
                   "Run `python train_all.py` to generate ablation results. "
                   "They will appear here automatically once training completes.")
        return

    df = pd.read_csv(csv_path)
    df["Best Val Bacc"] = df["Best Val Bacc"].astype(float)

    # Summary stats row
    models_present = df["Study"].str.split(" / ", n=1, expand=True)[0].unique()
    tile_data = []
    for mn in models_present:
        peak = df[df["Study"].str.startswith(mn)]["Best Val Bacc"].max()
        tile_data.append((f"{peak:.3f}", mn, "#0d9488"))
    stat_tiles(tile_data)
    st.markdown("<br>", unsafe_allow_html=True)

    # Summary table
    st.markdown("### Summary Table")
    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
    st.dataframe(
        df.style.highlight_max(subset=["Best Val Bacc"], color="#ccfbf1").format(precision=4),
        use_container_width=True,
        height=min(100 + len(df) * 36, 700))
    st.markdown('</div>', unsafe_allow_html=True)

    # Per-model tabs with plots
    st.markdown("### Learning Curves & Bar Charts")
    model_tabs = st.tabs(list(_ABL_PLOTS.keys()))
    for tab, (model_name, studies) in zip(model_tabs, _ABL_PLOTS.items()):
        with tab:
            found = [(abl_dir / fn, caption)
                     for fn, caption in studies
                     if (abl_dir / fn).exists()]
            if not found:
                st.info(f"No plots yet for {model_name}. "
                        "Run the training pipeline to generate them.")
                continue
            for path, caption in found:
                st.markdown(f'<div class="card card-sm">', unsafe_allow_html=True)
                st.image(str(path), caption=caption, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("ℹ️ Study design"):
        st.markdown("""
| Model | Study 1 | Study 2 | Study 3 |
|:---|:---|:---|:---|
| **EfficientNet-B3** | Augmentation on/off | Class weighting on/off | Frozen ↔ full fine-tuning |
| **SimpleCNN**       | Augmentation on/off | Class weighting on/off | 2 blocks ↔ 4 blocks |
| **ResNet50**        | Augmentation on/off | Class weighting on/off | Frozen ↔ full fine-tuning |
| **ViT+LoRA**        | Augmentation on/off | Class weighting on/off | LoRA rank=2 ↔ rank=8 |

All other hyperparameters are held constant within each study.
Each condition trains for up to **10 epochs** with patience-5 early stopping.
        """)


# ═══════════════════════════════════════════════════════════════
# Dispatcher
# ═══════════════════════════════════════════════════════════════
PAGE_HANDLERS = {
    PAGE_OPTIONS[0]: render_single_diagnosis,
    PAGE_OPTIONS[1]: render_batch,
    PAGE_OPTIONS[2]: render_ablations,
    PAGE_OPTIONS[3]: render_gradcam,
    PAGE_OPTIONS[4]: render_comparison,
    PAGE_OPTIONS[5]: render_embeddings,

}

PAGE_HANDLERS[selected_page]()
