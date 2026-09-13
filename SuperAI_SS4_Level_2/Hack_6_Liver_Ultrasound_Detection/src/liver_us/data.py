"""Paths and the image index (one row per image across train / val / test)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASSES = ["FFC", "FFS", "HCC", "cyst", "hemangioma", "dysplastic", "CCA"]
ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
RAW = DATASETS / "raw"          # extracted competition tree (gitignored)
META = DATASETS / "meta"        # mapping.xlsx, sample_submission.csv
INDEX = DATASETS / "index.csv"


def find_split_dirs(raw: Path = RAW) -> dict[str, dict[str, Path]]:
    """Locate images/labels dirs for each split whatever the zip's nesting is."""
    out = {}
    for split in ("train", "val", "validation", "test"):
        imgs = sorted(raw.rglob(f"{split}/images"))
        if not imgs:
            continue
        d = imgs[0]
        lab = None
        for cand in ("labels", "annotations"):
            if (d.parent / cand).is_dir():
                lab = d.parent / cand
        out["val" if split == "validation" else split] = dict(images=d, labels=lab)
    return out


def build_index(raw: Path = RAW, meta: Path = META, out: Path = INDEX) -> pd.DataFrame:
    sizes = pd.read_csv(raw / "image_sizes.csv")
    sizes["image_id"] = sizes["path"].map(lambda p: Path(p).stem)
    sizes["split"] = sizes["path"].map(lambda p: "val" if "/val" in p else ("test" if "/test" in p else "train"))
    mapping = pd.read_excel(meta / "mapping.xlsx")
    mapping["image_id"] = mapping["Image File"].str.replace(r"\.jpg$", "", regex=True)
    mapping = mapping.rename(columns={"Source": "source", "Type": "type"})[["image_id", "source", "type"]]
    df = sizes.merge(mapping, on="image_id", how="left")
    dirs = find_split_dirs(raw)
    df["label_path"] = ""
    df["n_boxes"] = 0
    df["classes"] = ""
    for split, d in dirs.items():
        if d["labels"] is None:
            continue
        m = df["split"] == split
        for i in df.index[m]:
            p = d["labels"] / f"{df.at[i, 'image_id']}.txt"
            if p.exists():
                lines = [l.split() for l in p.read_text().strip().splitlines() if l.strip()]
                df.at[i, "label_path"] = str(p)
                df.at[i, "n_boxes"] = len(lines)
                df.at[i, "classes"] = " ".join(l[0] for l in lines)
    df["abs_path"] = df["path"].map(lambda p: str(raw / p))
    df = df.drop(columns=["path"])
    df.to_csv(out, index=False)
    return df


def load_index(path: Path = INDEX) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str, "classes": str})
    df["classes"] = df["classes"].fillna("")
    return df
