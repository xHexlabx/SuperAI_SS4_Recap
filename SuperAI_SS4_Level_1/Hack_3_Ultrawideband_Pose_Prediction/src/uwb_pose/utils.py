"""Small shared helpers: seeding, device selection, logging, run naming."""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    return logging.getLogger("uwb_pose")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    idx = device.index or 0
    props = torch.cuda.get_device_properties(idx)
    return f"{props.name} ({props.total_memory / 1024**3:.1f} GB VRAM, CUDA {torch.version.cuda})"


def make_run_name(explicit: str | None) -> str:
    return explicit or datetime.now().strftime("run_%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
