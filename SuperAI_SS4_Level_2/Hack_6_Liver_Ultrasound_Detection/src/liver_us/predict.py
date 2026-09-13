"""Inference, ensembling (WBF), threshold tuning on val with the local metric, submission."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .data import CLASSES, load_index
from .metric import mean_ap, summarize, to_submission, yolo_labels_to_gts

log = logging.getLogger("liver_us.predict")
Preds = dict[str, list[tuple]]   # image_id -> [(cls, x1, y1, x2, y2, score), ...] in *stored* pixel coords


def latest_run(out_dir: Path) -> Path:
    runs = [p for p in out_dir.iterdir() if (p / "weights" / "best.pt").exists() or list(p.glob("preds_val*.pkl"))]
    if not runs:
        raise FileNotFoundError(f"no finished runs under {out_dir}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def predict_split(run_dir: Path, split: str, cfg: Config, index: pd.DataFrame, cache: bool = True) -> Preds:
    """Run one model on one split; cache raw boxes (conf>=cfg.predict.conf) as a pickle in the run dir."""
    p = cfg.predict
    tag = f"{split}_{cfg.model.imgsz}_c{p.conf}_i{p.iou}_{'tta' if p.augment else 'notta'}{'_ag' if p.agnostic_nms else ''}"
    cache_path = run_dir / f"preds_{tag}.pkl"
    if cache and cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    external = sorted(run_dir.glob(f"preds_{split}*.pkl"))
    if external:  # produced by scripts/rfdetr_predict.py (or any other framework); fuse flip variants by WBF
        parts = [pickle.loads(f.read_bytes()) for f in external]
        if len(parts) == 1:
            return parts[0]
        sizes = {r.image_id: (r.w, r.h) for r in index[index.split == split].itertuples()}
        return wbf_merge(parts, sizes, cfg.predict.wbf_iou)
    from ultralytics import RTDETR, YOLO

    weights = run_dir / "weights" / "best.pt"
    family = "rtdetr" if "rtdetr" in run_dir.name else "yolo"
    model = (RTDETR if family == "rtdetr" else YOLO)(str(weights))
    rows = index[index.split == split]
    paths = rows.abs_path.tolist()
    ids = rows.image_id.tolist()
    preds: Preds = {}
    bs = 16
    for i in range(0, len(paths), bs):
        res = model.predict(paths[i:i + bs], imgsz=cfg.model.imgsz, conf=p.conf, iou=p.iou, max_det=p.max_det,
                            augment=p.augment, agnostic_nms=p.agnostic_nms, verbose=False)
        for img_id, r in zip(ids[i:i + bs], res):
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy(); conf = b.conf.cpu().numpy(); cls = b.cls.cpu().numpy().astype(int)
            preds[img_id] = [(int(c), *map(float, bb), float(s)) for bb, s, c in zip(xyxy, conf, cls)]
        if i % (bs * 50) == 0:
            log.info("  %s %d/%d", split, i, len(paths))
    if cache:
        cache_path.write_bytes(pickle.dumps(preds))
    return preds


def wbf_merge(pred_list: list[Preds], sizes: dict[str, tuple[int, int]], iou_thr: float, weights=None,
              skip_thr: float = 0.001) -> Preds:
    """Weighted boxes fusion across models (boxes normalised by stored image size)."""
    from ensemble_boxes import weighted_boxes_fusion

    out: Preds = {}
    ids = set().union(*[p.keys() for p in pred_list])
    for img in ids:
        w, h = sizes[img]
        bl, sl, ll = [], [], []
        for p in pred_list:
            boxes = p.get(img, [])
            bl.append([[b[1] / w, b[2] / h, b[3] / w, b[4] / h] for b in boxes] or np.zeros((0, 4)))
            sl.append([b[5] for b in boxes])
            ll.append([b[0] for b in boxes])
        if sum(len(s) for s in sl) == 0:
            out[img] = []
            continue
        bl = [np.clip(np.array(b, dtype=float).reshape(-1, 4), 0, 1) for b in bl]
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=iou_thr, skip_box_thr=skip_thr)
        out[img] = [(int(c), x1 * w, y1 * h, x2 * w, y2 * h, float(sc)) for (x1, y1, x2, y2), sc, c in zip(b, s, l)]
    return out


def apply_threshold(preds: Preds, thr, max_boxes: int, class_thr: dict[int, float] | None = None) -> Preds:
    """Keep boxes with score >= thr (per-class override), sorted by score, capped per image."""
    out = {}
    for img, boxes in preds.items():
        kept = [b for b in boxes if b[5] >= (class_thr or {}).get(b[0], thr)]
        kept.sort(key=lambda b: -b[5])
        out[img] = kept[:max_boxes]
    return out


def tune_threshold(preds: Preds, gts, max_boxes: int, grid=None, iou_thrs=(0.5,)) -> tuple[float, pd.DataFrame]:
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.95, 0.05), 2)
    rows = []
    for t in grid:
        sub = apply_threshold(preds, t, max_boxes)
        m = mean_ap(sub, gts, ranked=False, iou_thrs=iou_thrs)
        rows.append(dict(thr=t, map=m["map"], n_boxes=sum(len(v) for v in sub.values()),
                         **{f"ap_{CLASSES[c]}": v["ap"] for c, v in m["per_class"].items()}))
    df = pd.DataFrame(rows)
    best = float(df.loc[df["map"].idxmax(), "thr"])
    return best, df


def tune_class_thresholds(preds: Preds, gts, max_boxes: int, base: float, grid=None, iou_thrs=(0.5,), rounds: int = 2) -> dict[int, float]:
    """Coordinate ascent on per-class thresholds (each class's AP only depends on its own threshold)."""
    grid = grid if grid is not None else np.round(np.arange(0.05, 0.95, 0.05), 2)
    thr = {c: base for c in range(len(CLASSES))}
    for _ in range(rounds):
        for c in range(len(CLASSES)):
            best_t, best_ap = thr[c], -1
            for t in grid:
                trial = dict(thr); trial[c] = float(t)
                sub = apply_threshold(preds, base, max_boxes, trial)
                ap = mean_ap(sub, gts, classes=[c], ranked=False, iou_thrs=iou_thrs)["map"]
                if ap > best_ap + 1e-9:
                    best_t, best_ap = float(t), ap
            thr[c] = best_t
    return thr


def scale_to_original(preds: Preds, index: pd.DataFrame) -> Preds:
    """Map boxes from stored (possibly downscaled) pixels back to the original image pixels."""
    sc = {r.image_id: (r.orig_w / r.w, r.orig_h / r.h) for r in index.itertuples()}
    out = {}
    for img, boxes in preds.items():
        sx, sy = sc[img]
        out[img] = [(b[0], b[1] * sx, b[2] * sy, b[3] * sx, b[4] * sy, b[5]) for b in boxes]
    return out


def val_slices(index: pd.DataFrame, seed: int = 0) -> dict[str, set[str]]:
    """all / machine / mobile / testlike (all mobile + machine subsampled to the test's 61 % mobile share)."""
    v = index[index.split == "val"]
    mobile = v[v.source == "mobile"].image_id
    machine = v[v.source == "machine"].image_id
    n_machine = int(round(len(mobile) * 0.39 / 0.61))
    rng = np.random.default_rng(seed)
    sub = machine.sample(n_machine, random_state=int(rng.integers(1 << 30)))
    return dict(all=set(v.image_id), machine=set(machine), mobile=set(mobile), testlike=set(mobile) | set(sub))


def val_gts(index: pd.DataFrame):
    rows = index[index.split == "val"]
    sizes = {r.image_id: (r.w, r.h) for r in rows.itertuples()}
    label_dir = Path(rows.label_path[rows.label_path.astype(str).str.len() > 0].iloc[0]).parent
    return yolo_labels_to_gts(label_dir, sizes, rows.image_id.tolist())


def run(cfg: Config) -> Path:
    index = load_index()
    out_dir = cfg.resolve(cfg.train.out_dir)
    runs = [cfg.resolve(r) if "/" in r else out_dir / r for r in cfg.predict.runs] or [latest_run(out_dir)]
    log.info("runs: %s", [r.name for r in runs])
    sizes = {r.image_id: (r.w, r.h) for r in index.itertuples()}
    gts = val_gts(index)
    val_preds, test_preds = [], []
    for r in runs:
        vp = predict_split(r, "val", cfg, index)
        log.info("%s val (ranked, all boxes): %s", r.name, summarize(vp, gts))
        val_preds.append(vp)
        test_preds.append(predict_split(r, "test", cfg, index))
    if len(runs) > 1:
        vp = wbf_merge(val_preds, sizes, cfg.predict.wbf_iou)
        tp = wbf_merge(test_preds, sizes, cfg.predict.wbf_iou)
        log.info("WBF val (ranked, all boxes): %s", summarize(vp, gts))
    else:
        vp, tp = val_preds[0], test_preds[0]
    p = cfg.predict
    slices = val_slices(index)
    tune_ids = slices[p.tune_slice]
    vp_t = {k: v for k, v in vp.items() if k in tune_ids}
    gts_t = {k: v for k, v in gts.items() if k in tune_ids}
    if p.threshold is None:
        thr, table = tune_threshold(vp_t, gts_t, p.max_boxes)
        log.info("threshold sweep on '%s' slice (unranked mAP@50):\n%s", p.tune_slice,
                 table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        class_thr = tune_class_thresholds(vp_t, gts_t, p.max_boxes, thr) if p.class_thresholds else None
    else:
        thr, class_thr = p.threshold, None
    sub_val = apply_threshold(vp, thr, p.max_boxes, class_thr)
    final = {}
    for name, ids in slices.items():
        final[name] = summarize({k: v for k, v in sub_val.items() if k in ids}, {k: v for k, v in gts.items() if k in ids})
        log.info("thr=%.2f class_thr=%s -> %-8s %s", thr, class_thr, name, {k: round(v, 4) for k, v in final[name].items()})
    sub_test = scale_to_original(apply_threshold(tp, thr, p.max_boxes, class_thr), index)
    test_ids = index[index.split == "test"].image_id.tolist()
    df = to_submission(sub_test, test_ids)
    df["Image File"] = df["Image File"].astype(int)
    out = cfg.resolve(p.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    meta = dict(runs=[str(r) for r in runs], threshold=thr, class_thr=class_thr, val=final,
                n_test_boxes=int(sum(len(v) for v in sub_test.values())))
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    log.info("wrote %s (%d test boxes)", out, meta["n_test_boxes"])
    return out
