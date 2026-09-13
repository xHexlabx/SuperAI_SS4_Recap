"""Train RF-DETR on the same symlink layout as the YOLO runs (run inside .venv-rfdetr).

usage: .venv-rfdetr/bin/python scripts/rfdetr_train.py --train datasets/yolo/train_os4_neg1 --size medium --epochs 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ["FFC", "FFS", "HCC", "cyst", "hemangioma", "dysplastic", "CCA"]
SIZES = {"nano": "RFDETRNano", "small": "RFDETRSmall", "medium": "RFDETRMedium", "large": "RFDETRLarge"}


def make_dataset_dir(train_dir: Path, name: str) -> Path:
    d = ROOT / "datasets" / "rfdetr" / name
    d.mkdir(parents=True, exist_ok=True)
    for split, src in (("train", train_dir), ("valid", ROOT / "datasets/yolo/val"), ("test", ROOT / "datasets/yolo/test")):
        link = d / split
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(src.resolve())
    (d / "data.yaml").write_text(yaml.safe_dump(dict(train="train/images", val="valid/images", test="test/images",
                                                    nc=len(CLASSES), names=CLASSES), sort_keys=False))
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/yolo/train_os4_neg1")
    p.add_argument("--size", default="medium", choices=list(SIZES))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--resolution", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--patience", type=int, default=10)
    a = p.parse_args()
    import rfdetr

    train_dir = ROOT / a.train
    name = a.name or f"rfdetr_{a.size}{'_' + str(a.resolution) if a.resolution else ''}_{train_dir.name}"
    ds = make_dataset_dir(train_dir, name)
    out = ROOT / "models" / name
    out.mkdir(parents=True, exist_ok=True)
    kw = {"resolution": a.resolution} if a.resolution else {}
    model = getattr(rfdetr, SIZES[a.size])(**kw)
    model.train(dataset_dir=str(ds), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=a.accum, lr=a.lr,
                output_dir=str(out), early_stopping=True, early_stopping_patience=a.patience, lr_scheduler="cosine",
                num_workers=8, tensorboard=False, wandb=False, run_test=False)
    print("done", out)


if __name__ == "__main__":
    main()
