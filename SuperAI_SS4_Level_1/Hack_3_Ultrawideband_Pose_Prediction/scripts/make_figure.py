#!/usr/bin/env python3
"""Render the three radar views for one sample of each class.

    uv run python scripts/make_figure.py [out.png]

The figure is the argument for this whole approach: if the classes look different as pictures,
an ImageNet backbone can tell them apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uwb_pose import radar  # noqa: E402
from uwb_pose.config import load_config  # noqa: E402

# Thai class names render as boxes without a Thai font, so label the figure in English.
EN = ["fall", "jump", "lie down", "run", "stand→sit", "sit→stand", "walk"]


def main() -> int:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "../../assets/uwb_encodings.png")
    cfg = load_config()
    root = cfg.resolve(cfg.data.root)
    ann = pd.read_csv(root / cfg.data.annotations_csv)
    views = list(cfg.encode.stack_methods)

    fig, axes = plt.subplots(len(views), 7, figsize=(16, 7.2), constrained_layout=True)
    for col, cls in enumerate(sorted(ann["class"].unique())):
        fid = ann[ann["class"] == cls].iloc[0]["id"]
        x = np.load(root / cfg.data.train_dir / f"{fid}.npy")
        for row, view in enumerate(views):
            ax = axes[row, col]
            ax.imshow(radar.encode_one(x, view, cfg.encode), aspect="auto", cmap="magma")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{cls} · {EN[cls]}", fontsize=10)
            if col == 0:
                ax.set_ylabel(view.replace("_", "\n"), fontsize=9)

    fig.suptitle(
        "UWB signal → image: three radar views per pose class "
        f"(clutter_removal={cfg.encode.clutter_removal}, doppler_crop={cfg.encode.doppler_crop})",
        fontsize=12,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
