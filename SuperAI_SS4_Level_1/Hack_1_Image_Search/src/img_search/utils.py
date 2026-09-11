"""Small shared helpers: logging, device selection, seeding, L2 normalisation."""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    # การโหลดโมเดลจาก HF ยิง log ระดับ INFO ทีละสิบบรรทัด ซึ่งกลบ log ของเราหมด
    for noisy in ("PIL", "httpx", "urllib3", "filelock", "huggingface_hub", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("img_search")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    props = torch.cuda.get_device_properties(device.index or 0)
    return f"{props.name} ({props.total_memory / 1024**3:.1f} GB VRAM, CUDA {torch.version.cuda})"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def l2norm(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation, so a dot product is a cosine similarity."""
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), eps)
