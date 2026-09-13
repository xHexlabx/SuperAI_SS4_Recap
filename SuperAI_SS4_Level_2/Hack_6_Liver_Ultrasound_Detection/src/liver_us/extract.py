"""Stream-extract the Kaggle zip without ever holding two copies of 15 GB on disk.

Small machine frames are copied byte-for-byte.  Large phone photos (max side
> ``max_side``) are re-encoded at ``max_side`` so the extracted tree stays a
few GB.  YOLO labels are normalised so training labels need no change; the
original (w, h) of every image is written to ``index.csv`` so test predictions
can be mapped back to original pixel coordinates.
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

MAX_SIDE = 2048
QUALITY = 94


def _process(args):
    zip_path, name, out_root, max_side = args
    out = Path(out_root) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        data = zf.read(name)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        out.write_bytes(data)
        return name, None, None, None, None
    im = Image.open(io.BytesIO(data))
    w, h = im.size
    if max(w, h) <= max_side:
        out.write_bytes(data)
        return name, w, h, w, h
    im.draft("RGB", (max_side, max_side))  # fast JPEG DCT downscale
    im = im.convert("RGB")
    s = max_side / max(w, h)
    nw, nh = round(w * s), round(h * s)
    im = im.resize((nw, nh), Image.LANCZOS)
    im.save(out, quality=QUALITY, optimize=True)
    return name, w, h, nw, nh


def extract(zip_path: str | os.PathLike, out_root: str | os.PathLike, max_side: int = MAX_SIDE, workers: int = 8):
    zip_path, out_root = Path(zip_path), Path(out_root)
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
    print(f"{len(names)} members in {zip_path.name}", file=sys.stderr)
    rows = []
    jobs = [(str(zip_path), n, str(out_root), max_side) for n in names]
    with ProcessPoolExecutor(workers) as ex:
        for i, r in enumerate(ex.map(_process, jobs, chunksize=32)):
            if r[1] is not None:
                rows.append(dict(path=r[0], orig_w=r[1], orig_h=r[2], w=r[3], h=r[4]))
            if i % 2000 == 0:
                print(f"  {i}/{len(names)}", file=sys.stderr)
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "image_sizes.csv", index=False)
    print(f"wrote {len(df)} image rows; resized {(df.orig_w != df.w).sum()}", file=sys.stderr)
    return df


if __name__ == "__main__":
    extract(sys.argv[1], sys.argv[2])
