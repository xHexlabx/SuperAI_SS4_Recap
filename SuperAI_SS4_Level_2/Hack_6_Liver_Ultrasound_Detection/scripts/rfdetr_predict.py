"""Predict val+test with an RF-DETR run and write preds_<split>.pkl in the same format as liver_us.predict.

usage: .venv-rfdetr/bin/python scripts/rfdetr_predict.py models/rfdetr_medium_train_os4_neg1 [--flip]
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--ckpt", default="checkpoint_best_ema.pth")
    p.add_argument("--thr", type=float, default=0.001)
    p.add_argument("--flip", action="store_true", help="also predict on the h-flipped image (fused later by WBF)")
    p.add_argument("--splits", default="val,test")
    a = p.parse_args()
    from rfdetr import RFDETR

    run = ROOT / a.run
    ckpt = run / a.ckpt
    if not ckpt.exists():
        cands = sorted(run.rglob("checkpoint_best_ema*"))
        ckpt = cands[-1] if cands else sorted(run.rglob("*.pth"))[-1]
    print("checkpoint", ckpt)
    model = RFDETR.from_checkpoint(str(ckpt))
    model.optimize_for_inference()
    names = getattr(model, "class_names", None)
    print("class_names", names)
    index = pd.read_csv(ROOT / "datasets/index.csv", dtype={"image_id": str})
    for split in a.splits.split(","):
        rows = index[index.split == split]
        for flip in ([False, True] if a.flip else [False]):
            preds = {}
            for i, r in enumerate(rows.itertuples()):
                im = Image.open(r.abs_path).convert("RGB")
                if flip:
                    im = im.transpose(Image.FLIP_LEFT_RIGHT)
                det = model.predict(im, threshold=a.thr)
                xyxy = np.asarray(det.xyxy, dtype=float).reshape(-1, 4)
                if flip and len(xyxy):
                    w = im.size[0]
                    xyxy = np.stack([w - xyxy[:, 2], xyxy[:, 1], w - xyxy[:, 0], xyxy[:, 3]], 1)
                preds[r.image_id] = [(int(c), *map(float, b), float(s)) for b, s, c in zip(xyxy, det.confidence, det.class_id)]
                if i % 500 == 0:
                    print(split, "flip" if flip else "", i, len(rows), flush=True)
            out = run / f"preds_{split}{'_flip' if flip else ''}.pkl"
            out.write_bytes(pickle.dumps(preds))
            cls = pd.Series([b[0] for v in preds.values() for b in v]).value_counts().sort_index()
            print("wrote", out, "class id counts", cls.to_dict())


if __name__ == "__main__":
    main()
