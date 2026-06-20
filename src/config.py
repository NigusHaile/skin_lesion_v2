"""
src/config.py
=============
Single configuration loader used by every module in the project.
Importing this module ensures ALL files share the same paths and hyperparameters.

Usage (in every other file):
import sys as _sys
from pathlib import Path as _Path
_SRC_ROOT = _Path(__file__).resolve().parent.parent
if str(_SRC_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SRC_ROOT))

    from src.config import CFG
    batch_size = CFG.dataloader.batch_size
"""

import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml


# Project root is the parent of src/; all relative paths in config.yaml resolve from here
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

# Keys under cfg.paths whose string values should be resolved to absolute paths
_PATH_KEYS = ("data_raw", "data_splits", "checkpoints", "results")


# Internal helpers 

def _to_namespace(d: dict) -> SimpleNamespace:
    """
    Recursively convert a nested dict to SimpleNamespace for attribute-style access.
    Leaf values are stored as-is; nested dicts become child namespaces.
    """
    ns = SimpleNamespace()
    for key, value in d.items():
        setattr(ns, key, _to_namespace(value) if isinstance(value, dict) else value)
    return ns


def _resolve_paths(cfg: SimpleNamespace) -> None:
    """
    Resolve all path strings under cfg.paths to absolute paths in-place.
    Values that are already absolute are left unchanged.
    """
    for key in _PATH_KEYS:
        raw = getattr(cfg.paths, key, None)
        if raw is not None:
            setattr(cfg.paths, key, str(PROJECT_ROOT / raw))


# Public API

def load_config(config_path: Path | str = _CONFIG_PATH) -> SimpleNamespace:
    """
    Parse config.yaml and return a dot-accessible SimpleNamespace.
    Path entries are resolved relative to PROJECT_ROOT.
    """
    with open(config_path) as f:
        cfg = _to_namespace(yaml.safe_load(f))
    _resolve_paths(cfg)
    return cfg


def seed_everything(seed: int) -> None:
    """
    Fix all random sources for full reproducibility across runs.
    Covers Python, NumPy, PyTorch (CPU + all GPUs), and hash randomisation.
    Note: deterministic mode disables cuDNN auto-tuning, which slightly reduces
    throughput — acceptable for reproducibility in a research setting.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    print(f"[config] Seed fixed → {seed}")


def get_device(cfg: SimpleNamespace) -> torch.device:
    """
    Resolve the target device from config, falling back to CPU if CUDA is unavailable.
    Prints VRAM when a GPU is found — useful for debugging OOM issues before training.
    """
    want_cuda = getattr(cfg.project, "device", "cpu") == "cuda"

    if want_cuda and torch.cuda.is_available():
        device  = torch.device("cuda")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[config] Device: {device} — {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    else:
        device = torch.device("cpu")
        if want_cuda:
            print("[config] CUDA requested but unavailable — falling back to CPU")
        else:
            print("[config] Device: CPU")

    return device


# Module-level singleton
# Loaded once at import time; all modules share the same object.
# Override with load_config(custom_path) only when running tests or sweeps.
CFG = load_config()