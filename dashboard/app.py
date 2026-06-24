"""
dashboard/app.py  ·  DermiAI  ·  Skin Lesion Analysis Platform
Run:  streamlit run dashboard/app.py
"""
import sys
import json
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
from src.skin_validator import SkinValidator, VALIDATOR_DIR

# ── Page config ────────────────────────────────────────────────
st.set_page_config(
    page_title="MedSkin AI · Skin Lesion Analysis",
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

# ── Top-right toolbar — injected into parent document via iframe ─
# st.markdown() uses React dangerouslySetInnerHTML which never executes
# <script> tags. st.components.v1.html() runs in a real iframe so JS
# executes reliably; we push everything into window.parent.document.
st.components.v1.html(r"""
<script>
(function () {
  'use strict';
  var P = window.parent, D = P.document;

  /* ── Idempotent: remove previous render on Streamlit rerun ─── */
  ['_trbar', '_trstyle', '_trscript'].forEach(function (id) {
    var el = D.getElementById(id); if (el) el.remove();
  });
  if (P._trCloseKb) D.removeEventListener('click', P._trCloseKb);

  /* ── CSS → parent <head> ─────────────────────────────────────── */
  var sty = D.createElement('style');
  sty.id  = '_trstyle';
  sty.textContent = [
    '#_trbar{position:fixed;top:10px;right:14px;z-index:2147483647;pointer-events:none}',
    '#_trbar>*{pointer-events:auto}',
    '#_kbwrap{position:relative;display:inline-block}',
    '#_kbbtn{position:relative;width:36px;height:36px;background:rgba(15,23,42,.88);color:#e2e8f0;',
    'border:1px solid rgba(255,255,255,.16);border-radius:10px;font-size:22px;line-height:1;',
    'cursor:pointer;display:flex;align-items:center;justify-content:center;',
    'backdrop-filter:blur(10px);padding:0;font-family:system-ui,sans-serif;',
    'transition:background .15s,border-color .15s}',
    '#_kbbtn:hover{background:rgba(30,41,59,.96);border-color:rgba(255,255,255,.26)}',
    '#_kbbtn._kbrec{border-color:#ef4444!important;background:rgba(239,68,68,.22)!important}',
    '#_recdot{display:none;position:absolute;top:5px;right:5px;width:7px;height:7px;',
    'border-radius:50%;background:#ef4444;animation:_blink 1s step-start infinite}',
    '@keyframes _blink{50%{opacity:0}}',
    '#_kbpanel{display:none;position:absolute;right:0;top:44px;width:256px;',
    'background:#1e293b;border:1px solid rgba(255,255,255,.1);border-radius:13px;',
    'box-shadow:0 18px 52px rgba(0,0,0,.6),0 2px 8px rgba(0,0,0,.3);overflow:hidden;z-index:2147483647}',
    '#_kbpanel._kbopen{display:block}',
    '._kbptitle{font-size:10px;font-weight:700;letter-spacing:.15em;text-transform:uppercase;',
    'color:rgba(100,116,139,.85);padding:12px 14px 8px;border-bottom:1px solid rgba(255,255,255,.07);',
    'font-family:Inter,system-ui,sans-serif}',
    '._kbpsect{font-size:9.5px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;',
    'color:rgba(100,116,139,.6);padding:10px 14px 4px;font-family:Inter,system-ui,sans-serif}',
    '._kbprec{display:flex;align-items:center;gap:8px;padding:6px 14px 10px;',
    'border-bottom:1px solid rgba(255,255,255,.06)}',
    '#_recbtn{flex:1;display:flex;align-items:center;justify-content:center;gap:7px;',
    'background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.12);border-radius:8px;',
    'color:rgba(203,213,225,.9);font-size:12px;font-weight:600;padding:7px 10px;cursor:pointer;',
    'font-family:Inter,system-ui,sans-serif;transition:background .14s,border-color .14s,color .14s}',
    '#_recbtn:hover{background:rgba(255,255,255,.13);color:#fff}',
    '#_recbtn._recon{background:rgba(239,68,68,.2);border-color:#ef4444;color:#fca5a5}',
    '#_recbtn._recon #_recicon{animation:_blink 1s step-start infinite}',
    '#_rectimer{display:none;background:#ef4444;color:#fff;font-size:11px;font-weight:700;',
    'font-family:monospace;padding:4px 8px;border-radius:6px;letter-spacing:.05em;white-space:nowrap}',
    '._kbprow{display:flex;align-items:center;justify-content:space-between;padding:6px 14px}',
    '._kbplbl{font-size:12px;color:rgba(203,213,225,.9);font-weight:500;font-family:Inter,system-ui,sans-serif}',
    '._kbpclr{width:32px;height:24px;border:1px solid rgba(255,255,255,.18);',
    'border-radius:5px;background:none;cursor:pointer;padding:1px}',
    '._kbppresets{display:flex;flex-wrap:wrap;gap:5px;padding:6px 14px 10px}',
    '._kbppre{background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);',
    'border-radius:6px;color:rgba(203,213,225,.85);font-size:11px;cursor:pointer;padding:4px 8px;',
    'transition:background .12s;font-family:Inter,system-ui,sans-serif}',
    '._kbppre:hover{background:rgba(255,255,255,.14);color:#fff}',
    '._kbpfoot{border-top:1px solid rgba(255,255,255,.07);padding:8px 14px}',
    '._kbpreset{width:100%;background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);',
    'border-radius:7px;color:rgba(148,163,184,.75);font-size:11px;cursor:pointer;padding:6px;',
    'transition:background .12s,color .12s;font-family:Inter,system-ui,sans-serif}',
    '._kbpreset:hover{background:rgba(255,255,255,.1);color:#e2e8f0}',
  ].join('');
  D.head.appendChild(sty);

  /* ── HTML → parent <body> ────────────────────────────────────── */
  var bar = D.createElement('div');
  bar.id  = '_trbar';
  bar.innerHTML =
    '<div id="_kbwrap">'
    + '<button id="_kbbtn" onclick="trToggleKb(event)" title="Settings &amp; Recorder" aria-label="Settings">'
    + '&#x22EE;<span id="_recdot"></span></button>'
    + '<div id="_kbpanel">'
    + '<div class="_kbptitle">System Settings</div>'
    + '<div class="_kbpsect">Screen Recording</div>'
    + '<div class="_kbprec">'
    +   '<button id="_recbtn" onclick="trToggleRec()">'
    +     '<span id="_recicon">&#x23FA;</span>'
    +     '<span id="_reclabel">Start Recording</span>'
    +   '</button>'
    +   '<span id="_rectimer">00:00</span>'
    + '</div>'
    + '<div class="_kbpsect" style="margin-top:4px">Appearance</div>'
    + '<div class="_kbprow"><label class="_kbplbl">App background</label>'
    +   '<input type="color" id="_setbg" class="_kbpclr" value="#f1f5f9"'
    +   ' oninput="trApply(\'app-bg\',this.value)"></div>'
    + '<div class="_kbprow"><label class="_kbplbl">Sidebar color</label>'
    +   '<input type="color" id="_setsb" class="_kbpclr" value="#0f172a"'
    +   ' oninput="trApply(\'sb-bg\',this.value)"></div>'
    + '<div class="_kbprow"><label class="_kbplbl">Accent color</label>'
    +   '<input type="color" id="_setac" class="_kbpclr" value="#14b8a6"'
    +   ' oninput="trApply(\'accent\',this.value)"></div>'
    + '<div class="_kbprow"><label class="_kbplbl">Card background</label>'
    +   '<input type="color" id="_setcard" class="_kbpclr" value="#ffffff"'
    +   ' oninput="trApply(\'card-bg\',this.value)"></div>'
    + '<div class="_kbprow"><label class="_kbplbl">Text color</label>'
    +   '<input type="color" id="_settxt" class="_kbpclr" value="#111827"'
    +   ' oninput="trApply(\'text\',this.value)"></div>'
    + '<div class="_kbpsect" style="margin-top:4px">Theme Presets</div>'
    + '<div class="_kbppresets">'
    +   '<button class="_kbppre" onclick="trPreset(\'teal\')">&#x1F7E2; Teal</button>'
    +   '<button class="_kbppre" onclick="trPreset(\'dark\')">&#x1F311; Dark</button>'
    +   '<button class="_kbppre" onclick="trPreset(\'purple\')">&#x1F7E3; Purple</button>'
    +   '<button class="_kbppre" onclick="trPreset(\'rose\')">&#x1F339; Rose</button>'
    +   '<button class="_kbppre" onclick="trPreset(\'ocean\')">&#x1F30A; Ocean</button>'
    + '</div>'
    + '<div class="_kbpfoot">'
    +   '<button class="_kbpreset" onclick="trReset()">&#x21BA; Reset to defaults</button>'
    + '</div>'
    + '</div></div>';
  D.body.appendChild(bar);

  /* ── Parent-context script ───────────────────────────────────────
     Injecting a <script> into the parent document means all functions
     (trToggleKb, trToggleRec, trApply, …) live on the parent window,
     so onclick="trToggleKb(event)" on the parent's buttons works.    */
  if (!P._trS) P._trS = { mr:null, stream:null, chunks:[], ti:null, secs:0, timerText:'00:00' };

  var sc = D.createElement('script');
  sc.id  = '_trscript';
  sc.textContent = '(function(){'
  /* state */
  + 'var S=window._trS;'
  /* panel toggle */
  + 'if(window._trCloseKb)document.removeEventListener("click",window._trCloseKb);'
  + 'window.trToggleKb=function(e){if(e&&e.stopPropagation)e.stopPropagation();'
  +   'var p=document.getElementById("_kbpanel");if(p)p.classList.toggle("_kbopen");};'
  + 'window._trCloseKb=function(e){'
  +   'var w=document.getElementById("_kbwrap");'
  +   'if(w&&!w.contains(e.target)){var p=document.getElementById("_kbpanel");if(p)p.classList.remove("_kbopen");}};'
  + 'document.addEventListener("click",window._trCloseKb);'
  /* screen recorder */
  + 'window.trToggleRec=async function(){'
  +   'var btn=document.getElementById("_recbtn"),icon=document.getElementById("_recicon"),'
  +       'lbl=document.getElementById("_reclabel"),tmr=document.getElementById("_rectimer"),'
  +       'dot=document.getElementById("_recdot"),kbb=document.getElementById("_kbbtn");'
  +   'if(!S.mr||S.mr.state==="inactive"){'
  +     'try{S.stream=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:30},audio:true});}'
  +     'catch(e){return;}'
  +     'var mime="video/webm;codecs=vp9,opus";'
  +     'if(!MediaRecorder.isTypeSupported(mime))mime="video/webm";'
  +     'S.mr=new MediaRecorder(S.stream,{mimeType:mime});S.chunks=[];'
  +     'S.mr.ondataavailable=function(e){if(e.data.size)S.chunks.push(e.data);};'
  +     'S.mr.onstop=function(){'
  +       'var blob=new Blob(S.chunks,{type:"video/webm"});'
  +       'var a=document.createElement("a");'
  +       'a.href=URL.createObjectURL(blob);'
  +       'a.download="DermiAI-"+new Date().toISOString().slice(0,19).replace(/[T:]/g,"-")+".webm";'
  +       'document.body.appendChild(a);a.click();document.body.removeChild(a);'
  +       'clearInterval(S.ti);S.mr=null;'
  +       'var t=document.getElementById("_rectimer"),d=document.getElementById("_recdot"),'
  +           'b=document.getElementById("_recbtn"),k=document.getElementById("_kbbtn"),'
  +           'i=document.getElementById("_recicon"),l=document.getElementById("_reclabel");'
  +       'if(t)t.style.display="none";if(d)d.style.display="none";'
  +       'if(b)b.classList.remove("_recon");if(k)k.classList.remove("_kbrec");'
  +       'if(i)i.textContent="⏺";if(l)l.textContent="Start Recording";'
  +     '};'
  +     'S.stream.getVideoTracks()[0].onended=function(){if(S.mr&&S.mr.state!=="inactive")S.mr.stop();};'
  +     'S.mr.start(200);S.secs=0;'
  +     'if(tmr)tmr.style.display="inline";if(dot)dot.style.display="block";'
  +     'if(btn)btn.classList.add("_recon");if(kbb)kbb.classList.add("_kbrec");'
  +     'if(icon)icon.textContent="⏹";if(lbl)lbl.textContent="Stop Recording";'
  +     'S.ti=setInterval(function(){'
  +       'S.secs++;'
  +       'var m=String(Math.floor(S.secs/60)).padStart(2,"0"),s=String(S.secs%60).padStart(2,"0");'
  +       'S.timerText=m+":"+s;'
  +       'var t=document.getElementById("_rectimer");if(t)t.textContent=S.timerText;'
  +     '},1000);'
  +   '}else{S.mr.stop();S.stream.getTracks().forEach(function(t){t.stop();});}'
  + '};'
  /* colour helpers */
  + 'var LS="dermai_v1_settings";'
  + 'function gS(){try{return JSON.parse(localStorage.getItem(LS))||{};}catch(e){return {};}}'
  + 'function sS(o){localStorage.setItem(LS,JSON.stringify(o));}'
  + 'function hx(r,g,b){return "#"+[r,g,b].map(function(v){return Math.round(Math.max(0,Math.min(255,v))).toString(16).padStart(2,"0");}).join("");}'
  + 'function dk(h){var r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return hx(r*.75,g*.75,b*.75);}'
  + 'function lk(h){var r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16);return hx(r*.9+30,g*.9+30,b*.9+30);}'
  + 'function iCSS(id,css){var el=document.getElementById(id);if(!el){el=document.createElement("style");el.id=id;document.head.appendChild(el);}el.textContent=css;}'
  + 'function aK(k,v){'
  +   'if(k==="app-bg")iCSS("_cs_appbg",".stApp{background:"+v+"!important}");'
  +   'else if(k==="sb-bg")iCSS("_cs_sbbg","[data-testid=\'stSidebar\']{background:"+v+"!important}[data-testid=\'stSidebar\']>div:first-child{background:"+v+"!important}");'
  +   'else if(k==="accent"){var d=dk(v),l=lk(v);'
  +     'iCSS("_cs_accent","[data-testid=\'stSidebar\'] button[data-testid=\'baseButton-primary\']{border-left-color:"+v+"!important}"'
  +     '+"[data-testid=\'stSidebar\'] .sb-mark{background:linear-gradient(135deg,"+d+","+v+")!important}"'
  +     '+".conf-bar-fill,.rank-fill{background:linear-gradient(90deg,"+d+","+v+")!important}"'
  +     '+".stProgress>div>div>div{background:linear-gradient(90deg,"+d+","+v+")!important}"'
  +     '+".tile-val,.conf-pct,.rank-pct{color:"+d+"!important}"'
  +     '+":root{--teal-500:"+v+";--teal-600:"+d+";--teal-400:"+l+"}");}'
  +   'else if(k==="card-bg")iCSS("_cs_cardbg",".card,.tile{background:"+v+"!important}");'
  +   'else if(k==="text")iCSS("_cs_text",".stApp{color:"+v+"!important}.rank-name,.pred-name{color:"+v+"!important}");'
  + '}'
  + 'window.trApply=function(k,v){var o=gS();o[k]=v;sS(o);aK(k,v);};'
  /* presets */
  + 'var PRE={"teal":{"app-bg":"#f1f5f9","sb-bg":"#0f172a","accent":"#14b8a6","card-bg":"#ffffff","text":"#111827"},'
  +   '"dark":{"app-bg":"#0f172a","sb-bg":"#020617","accent":"#14b8a6","card-bg":"#1e293b","text":"#e2e8f0"},'
  +   '"purple":{"app-bg":"#faf5ff","sb-bg":"#2e1065","accent":"#8b5cf6","card-bg":"#ffffff","text":"#111827"},'
  +   '"rose":{"app-bg":"#fff1f2","sb-bg":"#4c0519","accent":"#f43f5e","card-bg":"#ffffff","text":"#111827"},'
  +   '"ocean":{"app-bg":"#f0f9ff","sb-bg":"#0c1a2e","accent":"#0ea5e9","card-bg":"#ffffff","text":"#111827"}};'
  + 'var CID={"app-bg":"_setbg","sb-bg":"_setsb","accent":"_setac","card-bg":"_setcard","text":"_settxt"};'
  + 'window.trPreset=function(n){'
  +   'var p=PRE[n];if(!p)return;var o=gS();Object.assign(o,p);sS(o);'
  +   'Object.entries(p).forEach(function(kv){aK(kv[0],kv[1]);});'
  +   'Object.entries(CID).forEach(function(kv){var el=document.getElementById(kv[1]);if(el&&p[kv[0]])el.value=p[kv[0]];});};'
  + 'window.trReset=function(){localStorage.removeItem(LS);window.trPreset("teal");};'
  /* restore saved settings */
  + '(function(){var o=gS();if(!Object.keys(o).length)return;'
  +   'Object.entries(o).forEach(function(kv){aK(kv[0],kv[1]);});'
  +   'Object.entries(CID).forEach(function(kv){var el=document.getElementById(kv[1]);if(el&&o[kv[0]])el.value=o[kv[0]];});})();'
  /* sync recording UI if already active */
  + 'if(S.mr&&S.mr.state==="recording"){'
  +   'var b=document.getElementById("_recbtn"),d=document.getElementById("_recdot"),'
  +       'k=document.getElementById("_kbbtn"),i=document.getElementById("_recicon"),'
  +       'l=document.getElementById("_reclabel"),t=document.getElementById("_rectimer");'
  +   'if(b)b.classList.add("_recon");if(d)d.style.display="block";'
  +   'if(k)k.classList.add("_kbrec");if(i)i.textContent="⏹";'
  +   'if(l)l.textContent="Stop Recording";'
  +   'if(t){t.style.display="inline";t.textContent=S.timerText||"00:00";}}'
  + '})();';

  D.head.appendChild(sc);
})();
</script>
""", height=0, scrolling=False)

# ── (dead code sentinel — the old st.markdown block below is gone) ─
if False:
  pass

# ── Constants ──────────────────────────────────────────────────
RISK_CSS   = {"High": "high",   "Medium": "medium",   "Low": "low"}
BADGE_CSS  = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}
RISK_ICON  = {"High": "⚠️",  "Medium": "⚡",  "Low": "✅"}
MODEL_OPTS   = ["ViT-B/16 + LoRA", "ResNet50", "Simple CNN"]
GRADCAM_OPTS = ["ResNet50", "Simple CNN"]   # CNN-based (ViT has no conv layer)
TEAL_SEQ     = ["#0d9488", "#14b8a6", "#2dd4bf", "#5eead4"]

EMBED_KEYS = {"ViT+LoRA": "vit", "ResNet50": "resnet50", "Simple CNN": "simple_cnn"}

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

@st.cache_resource(show_spinner="Loading skin validator …")
def _load_skin_validator():
    """Load the ML skin validator from disk (cached across Streamlit reruns)."""
    dev = get_device(CFG)
    validator_pkl = VALIDATOR_DIR / "skin_validator.pkl"
    if not validator_pkl.exists():
        return None   # fall back to YCrCb heuristic
    try:
        return SkinValidator.load(VALIDATOR_DIR, dev)
    except Exception:
        return None


def _validate_skin_image(pil_img: Image.Image) -> tuple:
    """
    Returns (is_valid: bool, reason: str).

    Two-stage check:
    1. Minimum resolution — dermoscopy images are never thumbnails.
    2. ML skin validator (ResNet50 + IsolationForest trained on HAM10000).
       Falls back to YCrCb heuristic if the validator model is not available.
    """
    img_np = np.array(pil_img.convert("RGB"))
    h, w   = img_np.shape[:2]

    if h < 64 or w < 64:
        return False, (
            f"Image resolution ({w}×{h}) is too small for analysis. "
            "Please upload a proper dermoscopy photo."
        )

    # ── Stage 1: YCrCb skin-pixel check (Kovač et al. 2003) ─────────────────
    # Fast colour-space gate: rejects landscapes, food, objects, etc.
    img_ycrcb  = cv2.cvtColor(img_np, cv2.COLOR_RGB2YCrCb)
    Cr         = img_ycrcb[:, :, 1].astype(np.int32)
    Cb         = img_ycrcb[:, :, 2].astype(np.int32)
    skin_mask  = ((Cr >= 133) & (Cr <= 173) & (Cb >= 77) & (Cb <= 127))
    skin_ratio = float(skin_mask.mean())

    if skin_ratio < 0.20:
        pct = f"{skin_ratio:.0%}"
        return False, (
            f"This image does not appear to contain skin tissue "
            f"({pct} skin-coloured pixels). "
            "Please upload a close-up dermoscopy photograph of a skin lesion."
        )

    # ── Stage 2: ML distribution check (ResNet50 centroid distance) ──────────
    # Catches images that pass the colour gate but are not dermoscopy.
    validator = _load_skin_validator()
    if validator is not None:
        is_skin, dist = validator.is_skin(pil_img)
        if not is_skin:
            return False, (
                f"This image does not match the visual distribution of dermoscopy images "
                f"(distance score: {dist:.2f}). "
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

# ── Best-checkpoint auto-selection ─────────────────────────────
# For each model we gather all (checkpoint_abs_path, bacc, label) candidates
# and pick the one with the highest balanced accuracy.
# Ablation checkpoints live in results/ablations/{tag}_best.pth and their
# val-bacc is stored in the corresponding study JSON file.
# Base checkpoints live in checkpoints/{model}/... and their test bacc is
# read from results/{model}_metrics.json.

def _read_json_safe(p: Path) -> dict:
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {}

def _ablation_candidates(model_key: str) -> list:
    """Return list of (abs_path, bacc, label) from all ablation study JSONs for this model."""
    _model_json_glob = {
        "simple_cnn": "simplecnn_study*.json",
        "resnet50":   "resnet50_study*.json",
        "vit":        "vit_study*.json",
    }
    glob_pat = _model_json_glob.get(model_key)
    if not glob_pat:
        return []

    abl_dir     = Path(CFG.paths.results) / "ablations"
    study_jsons = sorted((abl_dir / "plots").glob(glob_pat))
    candidates  = []
    for sjson in study_jsons:
        data = _read_json_safe(sjson)
        for _condition, info in data.items():
            tag  = info.get("tag", "")
            bacc = info.get("best_val_bacc")
            if not tag or bacc is None:
                continue
            ckpt = abl_dir / "checkpoints" / f"{tag}_best.pth"
            if ckpt.exists():
                candidates.append((ckpt, float(bacc), f"ablation:{tag}"))
    return candidates

@st.cache_resource
def _pick_best_ckpt(model_key: str) -> tuple:
    """Return (abs_path, bacc, label) for the best checkpoint of this model."""
    base_ckpt_map = {
        "vit":        Path(CFG.paths.checkpoints) / "vit"       / "best.pth",
        "resnet50":   Path(CFG.paths.checkpoints) / "resnet50"  / "best.pth",
        "simple_cnn": Path(CFG.paths.checkpoints) / "simple_cnn" / "best.pth",
    }
    candidates = []

    # Base checkpoint
    base_p = base_ckpt_map.get(model_key)
    if base_p and base_p.exists():
        metrics_p = Path(CFG.paths.results) / f"{model_key}_metrics.json"
        base_bacc = _read_json_safe(metrics_p).get("balanced_accuracy", 0.0)
        candidates.append((base_p, float(base_bacc), "base"))

    # Ablation candidates
    candidates.extend(_ablation_candidates(model_key))

    if not candidates:
        # Fallback: return base path even if file is missing
        return (base_ckpt_map.get(model_key, Path(".")), 0.0, "base")

    best = max(candidates, key=lambda c: c[1])
    return best

# ── Model loaders ──────────────────────────────────────────────
@st.cache_resource
def _load(model_key: str, ckpt_abs: Path, label: str):
    dev = get_device(CFG)
    m   = build_model(model_key, dev)
    if ckpt_abs.exists():
        m.load_state_dict(_strip(torch.load(str(ckpt_abs), map_location=dev, weights_only=True)))
        m.eval()
    else:
        st.warning(f"Checkpoint not found: {ckpt_abs}")
    return m, dev

def _load_best(model_key: str):
    p, bacc, label = _pick_best_ckpt(model_key)
    return _load(model_key, p, label)

def load_vit():        return _load_best("vit")
def load_resnet():     return _load_best("resnet50")
def load_simple_cnn(): return _load_best("simple_cnn")

def pick_model(choice: str):
    if "ViT"       in choice: return load_vit()
    if "Simple CNN" in choice: return load_simple_cnn()
    return load_resnet()

def pick_gradcam_model(choice: str):
    """GradCAM only works on CNN-based models that expose get_gradcam_layer()."""
    if "ResNet" in choice: return load_resnet()
    return load_simple_cnn()

#  Sidebar 
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
        <div class="sb-name">MedSkin<em>AI</em></div>
      </div>
      <div class="sb-tagline">Skin Lesion Analysis Platform</div>
    </div>""", unsafe_allow_html=True)

    # Navigation — button per page (session_state tracks active page)
    st.markdown('<p class="sb-sect"></p>', unsafe_allow_html=True)
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
      <div class="sb-chip"><div class="sb-dot" style="background:#818cf8;"></div>
        <span class="sb-chip-name">ViT-B/16 + LoRA</span><span class="sb-badge">Transformer</span></div>
      <div class="sb-chip"><div class="sb-dot" style="background:#60a5fa;"></div>
        <span class="sb-chip-name">ResNet50</span><span class="sb-badge">Baseline</span></div>
      <div class="sb-chip"><div class="sb-dot" style="background:#9ca3af;"></div>
        <span class="sb-chip-name">Simple CNN</span><span class="sb-badge">Scratch</span></div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    
# ═════════════════════════
# PAGE 1 — Single Diagnosis
# ═════════════════════════
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


# ═══════════════════════════════
# PAGE 2 — GradCAM Explainability
# ═══════════════════════════════
def render_gradcam():
    page_banner("🌡️", "GradCAM",
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


# ═════════════════════════
# PAGE 3 — Model Comparison
# ═════════════════════════
def render_comparison():
    page_banner("📊", "Model Comparison",
                "All four architectures benchmarked on the held-out test set")

    res = Path(CFG.paths.results)
    csv_path = res / "model_comparison.csv"

    if not csv_path.exists():
        st.warning("No results found. Run `python train_all.py` first.")
        st.markdown("**Demo values (placeholder):**")
        st.dataframe(pd.DataFrame({
            "Model":           ["Simple CNN","ResNet50","ViT + LoRA"],
            "Balanced Acc":    ["0.612","0.761","0.843"],
            "Macro F1":        ["0.589","0.748","0.831"],
            "ROC-AUC (macro)": ["0.891","0.938","0.965"],
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
        model_names = ["simple_cnn","resnet50","vit"]
        disp_names  = ["Simple CNN","ResNet50","ViT+LoRA"]
        cols = st.columns(3, gap="small")
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
        cols = st.columns(3, gap="small")
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
        cols = st.columns(3, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / f"{mname}_roc_curves.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

    with tabs[5]:
        cols = st.columns(3, gap="small")
        for col, mname, dname in zip(cols, model_names, disp_names):
            p = res / f"{mname}_pr_curves.png"
            with col:
                if p.exists():
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(p), caption=dname, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════
# PAGE 4 — Embedding Explorer
# ═══════════════════════════
def render_embeddings():
    page_banner("🔵", "Embedding Explorer",
                "Each dot = one test image · colour = ground-truth class label")

    ca, cb, _ = st.columns([1,1,2])
    with ca:
        mc = st.selectbox("Embedding model", ["ViT+LoRA","ResNet50"])
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


# ═════════════════════════
# PAGE 5 — Batch Prediction
# ═════════════════════════
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


# ═════════════════════════
# PAGE 6 — Ablation Studies
# ═════════════════════════
_MODEL_STUDY_MAP = {

    "ViT+LoRA": [
        dict(json="vit_study1_augmentation.json",   png="vit_study1_augmentation.png",
             title="Study 1 — Data Augmentation",
             label_a="No augmentation",              label_b="With augmentation"),
        dict(json="vit_study2_class_weights.json",  png="vit_study2_class_weights.png",
             title="Study 2 — Class Weighting",
             label_a="Uniform loss",                label_b="Weighted CE loss"),
        dict(json="vit_study3_lora_rank.json",      png="vit_study3_lora_rank.png",
             title="Study 3 — LoRA Rank",
             label_a="LoRA rank=2",                 label_b="LoRA rank=8"),
    ],
}

_MODEL_CSV_PREFIX = {
    "ViT+LoRA":  "ViT+LoRA",
}


def _extract_study_metrics(cdata: dict) -> dict:
    """
    Pull best-val metrics from a study condition dict.
    Handles both the old JSON format (best_val_bacc only) and
    the new format (best_val_acc + best_val_f1 + best_val_precision + best_val_roc_auc).
    Falls back to history max when direct values are missing.
    """
    hist = cdata.get("history", {})
    acc     = cdata.get("best_val_acc")     or cdata.get("best_val_bacc")
    f1      = cdata.get("best_val_f1")
    prec    = cdata.get("best_val_precision")
    roc_auc = cdata.get("best_val_roc_auc")
    # Fallback: derive from training history
    if acc     is None and hist.get("val_acc"):
        acc     = max(hist["val_acc"])
    if f1      is None and hist.get("val_f1"):
        f1      = max(hist["val_f1"])
    if prec    is None and hist.get("val_precision"):
        prec    = max(hist["val_precision"])
    if roc_auc is None and hist.get("val_roc_auc"):
        valid = [v for v in hist["val_roc_auc"] if v == v]  # filter NaN
        roc_auc = max(valid) if valid else None
    return {
        "acc":     float(acc)     if acc     is not None else None,
        "f1":      float(f1)      if f1      is not None else None,
        "prec":    float(prec)    if prec    is not None else None,
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
    }


def _study_bar_chart(label_a: str, label_b: str,
                     m_a: dict, m_b: dict, title: str) -> go.Figure:
    """
    Grouped Plotly bar chart comparing two ablation conditions
    across all available metrics (Acc, F1, Precision, ROC-AUC).
    """
    metric_keys  = ["acc", "f1", "prec", "roc_auc"]
    metric_names = {
        "acc":     "Balanced Acc",
        "f1":      "Macro F1",
        "prec":    "Macro Precision",
        "roc_auc": "ROC-AUC (macro)",
    }

    # Only include metrics that have data for at least one condition
    available = [k for k in metric_keys
                 if m_a.get(k) is not None or m_b.get(k) is not None]

    x_labels = [metric_names[k] for k in available]
    vals_a   = [m_a.get(k, 0) or 0 for k in available]
    vals_b   = [m_b.get(k, 0) or 0 for k in available]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=label_a, x=x_labels, y=vals_a,
        marker_color="#ef4444",
        text=[f"{v:.3f}" for v in vals_a], textposition="outside",
        textfont=dict(size=11),
    ))
    fig.add_trace(go.Bar(
        name=label_b, x=x_labels, y=vals_b,
        marker_color="#2ecc71",
        text=[f"{v:.3f}" for v in vals_b], textposition="outside",
        textfont=dict(size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font_size=13),
        barmode="group",
        yaxis=dict(range=[0, 1.05], title="Score",
                   gridcolor="#f3f4f6", showgrid=True),
        xaxis=dict(title="Metric"),
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(255,255,255,.9)",
                    bordercolor="#e5e7eb", borderwidth=1),
        margin=dict(t=60, b=40, l=16, r=16),
        height=340,
    )
    fig.update_traces(marker_line_width=0)
    return fig


def _learning_curve_chart(label_a: str, label_b: str,
                           hist_a: dict, hist_b: dict, title: str) -> go.Figure | None:
    """
    Plotly line chart comparing val accuracy (and val_f1 / precision if present)
    over epochs for two ablation conditions. Returns None when no history available.
    Accepts both old key (val_bacc) and new key (val_acc).
    """
    va_a = hist_a.get("val_acc") or hist_a.get("val_bacc", [])
    va_b = hist_b.get("val_acc") or hist_b.get("val_bacc", [])
    if not va_a and not va_b:
        return None

    fig = go.Figure()
    if va_a:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(va_a) + 1)), y=va_a,
            name=f"{label_a} — Acc", mode="lines+markers",
            line=dict(color="#ef4444", width=2),
            marker=dict(size=5),
        ))
    if va_b:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(va_b) + 1)), y=va_b,
            name=f"{label_b} — Acc", mode="lines+markers",
            line=dict(color="#2ecc71", width=2),
            marker=dict(size=5),
        ))
    # Add F1 curves if present (dashed)
    vf_a = hist_a.get("val_f1", [])
    vf_b = hist_b.get("val_f1", [])
    if vf_a:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vf_a) + 1)), y=vf_a,
            name=f"{label_a} — F1", mode="lines",
            line=dict(color="#ef4444", width=1.5, dash="dot"),
        ))
    if vf_b:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vf_b) + 1)), y=vf_b,
            name=f"{label_b} — F1", mode="lines",
            line=dict(color="#2ecc71", width=1.5, dash="dot"),
        ))
    # Add Precision curves if present (dashed-long)
    vp_a = hist_a.get("val_precision", [])
    vp_b = hist_b.get("val_precision", [])
    if vp_a:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vp_a) + 1)), y=vp_a,
            name=f"{label_a} — Prec", mode="lines",
            line=dict(color="#ef4444", width=1.5, dash="longdash"),
        ))
    if vp_b:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(vp_b) + 1)), y=vp_b,
            name=f"{label_b} — Prec", mode="lines",
            line=dict(color="#2ecc71", width=1.5, dash="longdash"),
        ))
    # Add ROC-AUC curves if present (dashdot)
    va_auc_a = hist_a.get("val_roc_auc", [])
    va_auc_b = hist_b.get("val_roc_auc", [])
    if va_auc_a:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(va_auc_a) + 1)), y=va_auc_a,
            name=f"{label_a} — AUC", mode="lines",
            line=dict(color="#ef4444", width=2, dash="dashdot"),
        ))
    if va_auc_b:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(va_auc_b) + 1)), y=va_auc_b,
            name=f"{label_b} — AUC", mode="lines",
            line=dict(color="#2ecc71", width=2, dash="dashdot"),
        ))

    fig.update_layout(
        title=dict(text=f"Learning Curves — {title}", font_size=12),
        xaxis=dict(title="Epoch", dtick=1),
        yaxis=dict(title="Score", range=[0, 1.05],
                   gridcolor="#f3f4f6", showgrid=True),
        plot_bgcolor="white", paper_bgcolor="white", font_family="Inter",
        legend=dict(font_size=10, bgcolor="rgba(255,255,255,.9)",
                    bordercolor="#e5e7eb", borderwidth=1),
        margin=dict(t=50, b=40, l=16, r=16),
        height=320,
    )
    return fig


def _normalize_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    """Handle both old (Best Val Bacc) and new (Best Val Acc/F1/Prec/AUC) CSV formats."""
    if "Best Val Bacc" in df.columns and "Best Val Acc" not in df.columns:
        df = df.rename(columns={"Best Val Bacc": "Best Val Acc"})
    for col in ["Best Val Acc", "Best Val F1", "Best Val Prec", "Best Val AUC"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def render_ablations():
    page_banner("🔬", "Ablation Studies",
                "One variable changed at a time — quantifying each design choice")

    abl_dir  = Path(CFG.paths.results) / "ablations"
    csv_path = abl_dir / "ablation_summary.csv"

    # Check if any results exist at all 
    any_json = any(
        (abl_dir / study["json"]).exists()
        for studies in _MODEL_STUDY_MAP.values()
        for study   in studies
    )
    if not any_json and not csv_path.exists():
        empty_card("🔬", "No ablation results yet",
                   "Run `python train_all.py` to generate ablation results. "
                   "They will appear here automatically once training completes.")
        return

    # ── Load + normalise summary CSV 
    df_summary = None
    if csv_path.exists():
        df_summary = _normalize_summary_df(pd.read_csv(csv_path))

    # Global KPI tiles — best metric per model 
    acc_col = "Best Val Acc"
    tile_data = []
    for model_name, csv_prefix in _MODEL_CSV_PREFIX.items():
        peak_acc, peak_f1, peak_prec = None, None, None
        if df_summary is not None:
            mask = df_summary["Study"].str.startswith(csv_prefix)
            sub  = df_summary[mask]
            if not sub.empty and acc_col in sub.columns:
                peak_acc = sub[acc_col].max()
            if not sub.empty and "Best Val F1" in sub.columns:
                peak_f1  = sub["Best Val F1"].max()
            if not sub.empty and "Best Val Prec" in sub.columns:
                peak_prec = sub["Best Val Prec"].max()
        # Fallback: scan JSON files
        if peak_acc is None:
            for study in _MODEL_STUDY_MAP[model_name]:
                jp = abl_dir / study["json"]
                if not jp.exists():
                    continue
                try:
                    jdata = _read_json_safe(jp)
                    for cdata in jdata.values():
                        m = _extract_study_metrics(cdata)
                        if m["acc"] is not None:
                            peak_acc  = max(peak_acc  or 0, m["acc"])
                        if m["f1"]  is not None:
                            peak_f1   = max(peak_f1   or 0, m["f1"])
                        if m["prec"] is not None:
                            peak_prec = max(peak_prec or 0, m["prec"])
                except Exception:
                    pass
        if peak_acc is not None:
            display = f"{peak_acc:.3f}"
            if peak_f1   is not None: display += f"  ·  F1 {peak_f1:.3f}"
            if peak_prec is not None: display += f"  ·  Prec {peak_prec:.3f}"
            tile_data.append((f"{peak_acc:.3f}", f"{model_name} peak acc", "#0d9488"))

    if tile_data:
        stat_tiles(tile_data)
        st.markdown("<br>", unsafe_allow_html=True)

    # Summary table 
    if df_summary is not None and not df_summary.empty:
        st.markdown("### Summary Table — All Models × All Studies")
        num_cols = [c for c in ["Best Val Acc", "Best Val F1", "Best Val Prec", "Best Val AUC"]
                    if c in df_summary.columns]
        st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
        style = df_summary.style
        for c in num_cols:
            style = style.highlight_max(subset=[c], color="#ccfbf1")
        style = style.format({c: "{:.4f}" for c in num_cols})
        st.dataframe(style, use_container_width=True,
                     height=min(100 + len(df_summary) * 36, 700))
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-model interactive tabs 
    st.markdown("### Detailed Results — All Models")
    model_tabs = st.tabs(list(_MODEL_STUDY_MAP.keys()))

    for tab, (model_name, studies) in zip(model_tabs, _MODEL_STUDY_MAP.items()):
        with tab:
            # Model-level KPI row from CSV
            if df_summary is not None:
                prefix = _MODEL_CSV_PREFIX[model_name]
                sub    = df_summary[df_summary["Study"].str.startswith(prefix)]
                if not sub.empty:
                    kpi_cols = [c for c in ["Best Val Acc","Best Val F1","Best Val Prec","Best Val AUC"]
                                if c in sub.columns]
                    if kpi_cols:
                        kpi_items = []
                        for c in kpi_cols:
                            kpi_items.append((f"{sub[c].max():.3f}", f"Peak {c}", "#0d9488"))
                        kpi_items.append((str(len(sub)), "Conditions run", "#6b7280"))
                        stat_tiles(kpi_items)
                        st.markdown("<br>", unsafe_allow_html=True)

            # Per-study breakdown
            model_has_any = False
            for study in studies:
                jp  = abl_dir / study["json"]
                png = abl_dir / study["png"]

                #  Interactive charts from JSON 
                if jp.exists():
                    model_has_any = True
                    try:
                        jdata = _read_json_safe(jp)
                        keys  = list(jdata.keys())
                        if len(keys) >= 2:
                            cdata_a = jdata[keys[0]]
                            cdata_b = jdata[keys[1]]
                            m_a     = _extract_study_metrics(cdata_a)
                            m_b     = _extract_study_metrics(cdata_b)
                            hist_a  = cdata_a.get("history", {})
                            hist_b  = cdata_b.get("history", {})

                            st.markdown(
                                f'<div class="card card-sm" style="margin-bottom:.5rem;">'
                                f'<b style="font-size:.95rem;">{study["title"]}</b>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

                            col_bar, col_curve = st.columns([1, 1], gap="medium")
                            with col_bar:
                                fig = _study_bar_chart(
                                    study["label_a"], study["label_b"],
                                    m_a, m_b, study["title"],
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col_curve:
                                fig_lc = _learning_curve_chart(
                                    study["label_a"], study["label_b"],
                                    hist_a, hist_b, study["title"],
                                )
                                if fig_lc:
                                    st.plotly_chart(fig_lc, use_container_width=True)
                                else:
                                    st.info("No per-epoch history in this JSON.")

                            # Inline metric comparison table
                            def _fmt(v): return f"{v:.4f}" if v is not None else "—"
                            rows_tbl = [
                                {
                                    "Condition":       study["label_a"],
                                    "Best Val Acc":    _fmt(m_a["acc"]),
                                    "Best Val F1":     _fmt(m_a["f1"]),
                                    "Best Val Prec":   _fmt(m_a["prec"]),
                                    "Δ vs baseline":   "",
                                },
                                {
                                    "Condition":       study["label_b"],
                                    "Best Val Acc":    _fmt(m_b["acc"]),
                                    "Best Val F1":     _fmt(m_b["f1"]),
                                    "Best Val Prec":   _fmt(m_b["prec"]),
                                    "Δ vs baseline":   "",
                                },
                            ]
                            # Fill delta column
                            for metric_key, col_name in [("acc","Best Val Acc"),
                                                          ("f1","Best Val F1"),
                                                          ("prec","Best Val Prec")]:
                                va, vb = m_a.get(metric_key), m_b.get(metric_key)
                                if va is not None and vb is not None:
                                    delta = vb - va
                                    sign  = "+" if delta >= 0 else ""
                                    rows_tbl[1]["Δ vs baseline"] = (
                                        f"{sign}{delta:+.4f} acc"
                                        if metric_key == "acc" else
                                        rows_tbl[1]["Δ vs baseline"]
                                    )
                            tdf = pd.DataFrame(rows_tbl)
                            st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                            st.dataframe(tdf, use_container_width=True, hide_index=True)
                            st.markdown('</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"Could not parse {study['json']}: {e}")

                # Static PNG fallback 
                elif png.exists():
                    model_has_any = True
                    st.markdown('<div class="card card-sm">', unsafe_allow_html=True)
                    st.image(str(png), caption=study["title"], use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            if not model_has_any:
                st.info(f"No ablation results yet for {model_name}. "
                        "Run the training pipeline to generate them.")

    # Study design reference 
    with st.expander("ℹ️ Study design"):
        st.markdown("""
| Model | Study 1 | Study 2 | Study 3 |
|:---|:---|:---|:---|
| **SimpleCNN**       | Augmentation on/off | Class weighting on/off | 2 blocks ↔ 4 blocks |
| **ResNet50**        | Augmentation on/off | Class weighting on/off | Frozen ↔ full fine-tuning |
| **ViT+LoRA**        | Augmentation on/off | Class weighting on/off | LoRA rank=2 ↔ rank=8 |

All other hyperparameters are held constant within each study.
Each condition trains for up to **10 epochs** with patience-5 early stopping.
Solid lines = Balanced Accuracy · Dotted lines = Macro F1 · Long-dashed lines = Macro Precision
        """)


# ═════════==
# Dispatcher
# ══════════
PAGE_HANDLERS = {
    PAGE_OPTIONS[0]: render_single_diagnosis,
    PAGE_OPTIONS[1]: render_batch,
    PAGE_OPTIONS[2]: render_ablations,
    PAGE_OPTIONS[3]: render_gradcam,
    PAGE_OPTIONS[4]: render_comparison,
    PAGE_OPTIONS[5]: render_embeddings,

}

PAGE_HANDLERS[selected_page]()
