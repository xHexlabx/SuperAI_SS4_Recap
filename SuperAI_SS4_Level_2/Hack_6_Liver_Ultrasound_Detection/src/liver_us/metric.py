"""Local mean-average-precision that mirrors what a no-confidence Kaggle submission gets.

The competition file has ``Annotation`` (Pascal VOC boxes) and ``Label`` columns but
*no score*.  Any host-side mAP implementation therefore ranks boxes either in file
order or with all scores tied, so we evaluate two ways:

* ``ranked``   – boxes sorted by our own confidence (upper bound; what a scored
  submission would get).
* ``unranked`` – every submitted box gets score 1.0 and the file order is used, i.e.
  what the leaderboard most plausibly computes.  This makes precision matter as much
  as recall, so the confidence threshold has to be tuned for it.

Boxes: ``(cls, x1, y1, x2, y2, score)``; GT: ``(cls, x1, y1, x2, y2)``.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

Box = Sequence[float]
IOU_COCO = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between (N,4) and (M,4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def _ap_from_tp(tp: np.ndarray, n_gt: int, interp: str = "all") -> float:
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / (ctp + cfp)
    if interp == "coco":  # 101-point
        mprec = np.maximum.accumulate(prec[::-1])[::-1]
        rs = np.linspace(0, 1, 101)
        idx = np.searchsorted(rec, rs, side="left")
        return float(np.mean([mprec[i] if i < len(mprec) else 0.0 for i in idx]))
    # VOC2010+ all-point interpolation
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def class_ap(
    preds: Mapping[str, Sequence[Box]],
    gts: Mapping[str, Sequence[Box]],
    cls: int,
    iou_thr: float,
    ranked: bool = True,
    interp: str = "all",
) -> tuple[float, int, int, int]:
    """AP for one class. Returns (ap, n_gt, n_tp, n_pred)."""
    gt_by_img = {}
    n_gt = 0
    for img, boxes in gts.items():
        b = np.array([bb[1:5] for bb in boxes if int(bb[0]) == cls], dtype=float).reshape(-1, 4)
        if len(b):
            gt_by_img[img] = b
            n_gt += len(b)
    records = []  # (score, order, img, box)
    order = 0
    for img, boxes in preds.items():
        for bb in boxes:
            if int(bb[0]) != cls:
                continue
            score = float(bb[5]) if (ranked and len(bb) > 5) else 1.0
            records.append((score, order, img, np.array(bb[1:5], dtype=float)))
            order += 1
    if ranked:
        records.sort(key=lambda r: (-r[0], r[1]))
    matched = {img: np.zeros(len(b), bool) for img, b in gt_by_img.items()}
    tp = np.zeros(len(records))
    for k, (_, _, img, box) in enumerate(records):
        g = gt_by_img.get(img)
        if g is None:
            continue
        ious = iou_matrix(box[None], g)[0]
        ious[matched[img]] = -1  # greedy: each GT matched once
        j = int(np.argmax(ious)) if len(ious) else -1
        if j >= 0 and ious[j] >= iou_thr:
            matched[img][j] = True
            tp[k] = 1
    return _ap_from_tp(tp, n_gt, interp), n_gt, int(tp.sum()), len(records)


def mean_ap(
    preds: Mapping[str, Sequence[Box]],
    gts: Mapping[str, Sequence[Box]],
    classes: Iterable[int] = range(7),
    iou_thrs: Sequence[float] = (0.5,),
    ranked: bool = True,
    interp: str = "all",
) -> dict:
    """Macro mAP over classes (classes without GT are skipped) and over IoU thresholds."""
    per_class = {}
    for c in classes:
        aps = []
        for t in iou_thrs:
            ap, n_gt, n_tp, n_pred = class_ap(preds, gts, c, t, ranked, interp)
            aps.append(ap)
        per_class[c] = dict(ap=float(np.nanmean(aps)), n_gt=n_gt, n_tp=n_tp, n_pred=n_pred)
    valid = [v["ap"] for v in per_class.values() if not np.isnan(v["ap"])]
    return dict(map=float(np.mean(valid)) if valid else 0.0, per_class=per_class)


def summarize(preds, gts, classes=range(7)) -> dict:
    """The handful of numbers we track: unranked/ranked at IoU .5 and .5:.95."""
    out = {}
    for name, ranked in (("unranked", False), ("ranked", True)):
        out[f"{name}@50"] = mean_ap(preds, gts, classes, (0.5,), ranked)["map"]
        out[f"{name}@50:95"] = mean_ap(preds, gts, classes, IOU_COCO, ranked)["map"]
    return out


# ---------- submission (de)serialisation ----------

def parse_submission(df: pd.DataFrame) -> dict[str, list[Box]]:
    out = {}
    for img, ann, lab in zip(df["Image File"], df["Annotation"], df["Label"]):
        boxes = ast.literal_eval(ann) if isinstance(ann, str) and ann.strip() else []
        labels = ast.literal_eval(lab) if isinstance(lab, str) and lab.strip() else []
        out[str(img)] = [(int(c), *map(float, b)) for b, c in zip(boxes, labels)]
    return out


def to_submission(preds: Mapping[str, Sequence[Box]], image_ids: Sequence[str]) -> pd.DataFrame:
    """Build the Kaggle csv: ints for coordinates, ``[]`` for empty images."""
    rows = []
    for img in image_ids:
        boxes = preds.get(str(img), [])
        ann = [[int(round(b[1])), int(round(b[2])), int(round(b[3])), int(round(b[4]))] for b in boxes]
        lab = [int(b[0]) for b in boxes]
        rows.append(dict(**{"Image File": img}, Annotation=str(ann), Label=str(lab)))
    return pd.DataFrame(rows)


def yolo_labels_to_gts(label_dir, sizes: Mapping[str, tuple[int, int]], image_ids: Sequence[str]) -> dict[str, list[Box]]:
    """Read YOLO txt files (cls cx cy w h, normalised) into pixel xyxy GT dict."""
    from pathlib import Path

    label_dir = Path(label_dir)
    gts = {}
    for img in image_ids:
        w, h = sizes[str(img)]
        p = label_dir / f"{img}.txt"
        boxes = []
        if p.exists():
            for line in p.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                c, cx, cy, bw, bh = int(float(parts[0])), *map(float, parts[1:5])
                boxes.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        gts[str(img)] = boxes
    return gts
