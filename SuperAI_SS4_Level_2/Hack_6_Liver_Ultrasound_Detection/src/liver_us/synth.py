"""Simulate a phone photo of the ultrasound screen from a machine frame.

Observed on the real mobile images (train/val `source == mobile`, test = 61 % mobile):
the fan fills most of the photo (operator zooms in), strong defocus blur, brightness
lift with a mild blue tint, sensor grain, slight tilt.  We reproduce that chain on
machine frames and warp the YOLO boxes accordingly, writing a synthetic split that
the normal symlink layout can pick up.
"""
from __future__ import annotations

import io
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

OUT_MAX = 1536  # long side of the synthetic photo


def fan_bbox(gray: np.ndarray, thr: int = 12) -> tuple[int, int, int, int]:
    """Bounding box of the bright (ultrasound) region, ignoring the UI text columns."""
    m = gray > thr
    m[:, : int(gray.shape[1] * 0.06)] = False
    m[:, int(gray.shape[1] * 0.94):] = False
    ys, xs = np.where(m)
    if len(xs) < 100:
        h, w = gray.shape
        return 0, 0, w, h
    x1, x2 = np.percentile(xs, [1, 99]).astype(int)
    y1, y2 = np.percentile(ys, [1, 99]).astype(int)
    return int(x1), int(y1), int(x2), int(y2)


def synth_photo(img: np.ndarray, boxes: np.ndarray, rng: np.random.Generator):
    """img: HxWx3 uint8 (BGR), boxes: (N,4) xyxy pixels. Returns (photo, boxes_xyxy)."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fx1, fy1, fx2, fy2 = fan_bbox(gray)
    # 1. zoom: crop around the fan with random margins (photo is 4:3)
    mx = rng.uniform(-0.05, 0.15) * (fx2 - fx1)
    my = rng.uniform(-0.05, 0.15) * (fy2 - fy1)
    cx1, cy1 = max(0, fx1 - mx), max(0, fy1 - my)
    cx2, cy2 = min(w, fx2 + mx), min(h, fy2 + my)
    # adjust to 4:3 aspect
    cw, ch = cx2 - cx1, cy2 - cy1
    if cw / ch > 4 / 3:
        ch = cw * 3 / 4
    else:
        cw = ch * 4 / 3
    cx, cy = (cx1 + cx2) / 2 + rng.uniform(-0.03, 0.03) * w, (cy1 + cy2) / 2 + rng.uniform(-0.03, 0.03) * h
    src = np.float32([[cx - cw / 2, cy - ch / 2], [cx + cw / 2, cy - ch / 2], [cx + cw / 2, cy + ch / 2], [cx - cw / 2, cy + ch / 2]])
    # 2. perspective tilt: jitter the destination corners
    ow, oh = OUT_MAX, OUT_MAX * 3 // 4
    j = 0.06
    dst = (np.float32([[0, 0], [ow, 0], [ow, oh], [0, oh]]) + rng.uniform(-j, j, (4, 2)) * np.float32([ow, oh])).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(img, M, (ow, oh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    # boxes: warp the 4 corners, take the enclosing box
    out_boxes = []
    for x1, y1, x2, y2 in boxes:
        pts = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape(-1, 1, 2)
        p = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
        bx1, by1 = np.clip(p[:, 0].min(), 0, ow), np.clip(p[:, 1].min(), 0, oh)
        bx2, by2 = np.clip(p[:, 0].max(), 0, ow), np.clip(p[:, 1].max(), 0, oh)
        if (bx2 - bx1) * (by2 - by1) > 0.3 * (x2 - x1) * (y2 - y1) * (M[0, 0] * M[1, 1]):
            out_boxes.append([bx1, by1, bx2, by2])
    photo = photo.astype(np.float32)
    # 3. defocus blur (photos are soft) + slight motion blur
    k = int(rng.integers(3, 15)) | 1
    photo = cv2.GaussianBlur(photo, (k, k), 0)
    if rng.random() < 0.3:
        kk = int(rng.integers(5, 21)); kern = np.zeros((kk, kk), np.float32)
        ang = rng.uniform(0, np.pi); c, s = np.cos(ang), np.sin(ang)
        for t in np.linspace(-kk / 2, kk / 2, kk * 2):
            x, y = int(kk / 2 + t * c), int(kk / 2 + t * s)
            if 0 <= x < kk and 0 <= y < kk: kern[y, x] = 1
        kern /= max(kern.sum(), 1); photo = cv2.filter2D(photo, -1, kern)
    # 4. tone: brightness lift, gamma, contrast, blue-ish tint
    gamma = rng.uniform(0.6, 1.1)
    photo = 255.0 * (photo / 255.0) ** gamma
    photo = photo * rng.uniform(0.85, 1.25) + rng.uniform(0, 25)
    tint = np.array([rng.uniform(0, 12), rng.uniform(-4, 4), rng.uniform(-6, 4)], np.float32)  # BGR
    photo = photo + tint
    # 5. vignette / uneven illumination + optional glare blob
    yy, xx = np.mgrid[0:oh, 0:ow].astype(np.float32)
    vx, vy = rng.uniform(0.3, 0.7) * ow, rng.uniform(0.3, 0.7) * oh
    r2 = ((xx - vx) / ow) ** 2 + ((yy - vy) / oh) ** 2
    photo = photo * (1 - rng.uniform(0.1, 0.5) * r2 * 2)[..., None]
    if rng.random() < 0.3:
        gx, gy, gs = rng.uniform(0, ow), rng.uniform(0, oh), rng.uniform(0.05, 0.2) * ow
        glare = np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * gs ** 2))) * rng.uniform(20, 80)
        photo = photo + glare[..., None]
    # 6. moiré-like grating (weak) + sensor grain
    if rng.random() < 0.4:
        f = rng.uniform(0.02, 0.08); ang = rng.uniform(0, np.pi)
        grating = np.sin(2 * np.pi * f * (xx * np.cos(ang) + yy * np.sin(ang))) * rng.uniform(2, 8)
        photo = photo + grating[..., None]
    photo = photo + rng.normal(0, rng.uniform(2, 8), photo.shape).astype(np.float32)
    photo = np.clip(photo, 0, 255).astype(np.uint8)
    # 7. JPEG round trip at phone-like quality
    q = int(rng.integers(70, 95))
    _, enc = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, q])
    photo = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return photo, np.array(out_boxes, dtype=np.float32).reshape(-1, 4)


def _job(args):
    img_path, label_path, out_img, out_lbl, seed = args
    rng = np.random.default_rng(seed)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    cls, boxes = [], []
    if label_path and Path(label_path).exists():
        for line in Path(label_path).read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5: continue
            c, cx, cy, bw, bh = int(float(p[0])), *map(float, p[1:5])
            cls.append(c); boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    photo, nb = synth_photo(img, np.array(boxes, np.float32).reshape(-1, 4), rng)
    if len(boxes) and len(nb) != len(boxes):
        return None  # a lesion fell outside the crop: skip this sample to keep labels exact
    cv2.imwrite(out_img, photo, [cv2.IMWRITE_JPEG_QUALITY, 92])
    oh, ow = photo.shape[:2]
    if len(nb):
        lines = [f"{c} {(b[0]+b[2])/2/ow:.6f} {(b[1]+b[3])/2/oh:.6f} {(b[2]-b[0])/ow:.6f} {(b[3]-b[1])/oh:.6f}" for c, b in zip(cls, nb)]
        Path(out_lbl).write_text("\n".join(lines) + "\n")
    return out_img


def generate(index_csv: str, out_dir: str, n_per_image: int = 1, neg_fraction: float = 0.5, seed: int = 0, workers: int = 8):
    import pandas as pd

    df = pd.read_csv(index_csv, dtype={"image_id": str})
    df = df[(df.split == "train") & (df.source == "machine")]
    rng = np.random.default_rng(seed)
    neg = df[df.n_boxes == 0]; pos = df[df.n_boxes > 0]
    df = pd.concat([pos, neg[rng.random(len(neg)) < neg_fraction]])
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True); (out / "labels").mkdir(exist_ok=True)
    jobs = []
    for r in df.itertuples():
        for k in range(n_per_image):
            jobs.append((r.abs_path, r.label_path if isinstance(r.label_path, str) else "",
                         str(out / "images" / f"{r.image_id}__syn{k}.jpg"), str(out / "labels" / f"{r.image_id}__syn{k}.txt"),
                         int(rng.integers(0, 2**31))))
    n = 0
    with ProcessPoolExecutor(workers) as ex:
        for i, res in enumerate(ex.map(_job, jobs, chunksize=16)):
            n += res is not None
            if i % 2000 == 0: print(f"  {i}/{len(jobs)}", file=sys.stderr)
    print(f"wrote {n} synthetic photos to {out}", file=sys.stderr)


if __name__ == "__main__":
    generate(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1)
