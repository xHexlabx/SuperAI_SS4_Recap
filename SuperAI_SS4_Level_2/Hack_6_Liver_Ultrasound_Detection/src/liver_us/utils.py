"""Shared helpers: logging, seeding, device, run naming."""
from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    logging.getLogger("braindecode").setLevel(logging.ERROR)
    return logging.getLogger("liver_us")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_run_name(explicit: str | None, model_name: str) -> str:
    return explicit or datetime.now().strftime(f"{model_name}_%Y%m%d_%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
