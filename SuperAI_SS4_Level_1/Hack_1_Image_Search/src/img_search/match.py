"""Turn embeddings into a class per test image, including the "none of the 22" class.

สองการตัดสินใจที่แยกกันชัดเจน:
  1. คลาสไหนใกล้ที่สุด   -> `class_scores` (proto / max / topk)
  2. ใกล้พอจะนับว่าตรงไหม -> `decide` (threshold / ratio test / negative gallery)

ข้อ 2 คือจุดที่ baseline เดิมพัง: มันฮาร์ดโค้ด cosine > 0.75 ไว้บน pooler_output ของ CLIP
ซึ่งไม่ใช่สเปซที่ CLIP ใช้วัดความคล้ายด้วยซ้ำ
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .utils import l2norm

log = logging.getLogger("img_search.match")


@dataclass
class Scores:
    known: np.ndarray            # (N, n_classes) คะแนนของคลาส 0..21
    negative: np.ndarray | None  # (N,) คะแนนของ negative gallery ถ้ามี


def class_scores(emb: np.ndarray, gallery_emb: np.ndarray, gallery_cls: np.ndarray, *,
                 n_classes: int, unknown_class: int, aggregate: str = "topk",
                 topk: int = 3, neg_quantile: float = 1.0) -> Scores:
    """Cosine similarity from each row of `emb` to each class in the gallery.

    `neg_quantile` คุมว่าคลาส 22 จะถูกสรุปด้วยสถิติตัวไหน และ **สำคัญกว่าที่คิด**:
    คลาสปกติมีโลโก้อ้างอิงรูปเดียว แต่คลาส 22 มีตั้ง 2,376 รูป การใช้ `max` (quantile = 1.0)
    จึงเป็นการเทียบ "หนึ่งตัวอย่าง" กับ "ค่าสูงสุดของสองพันตัวอย่าง" ซึ่งไม่ยุติธรรม
    และยิ่ง gallery ใหญ่ขึ้นก็ยิ่งเอนเอียง — ค่า quantile ที่ต่ำกว่า 1 ไม่โตตามขนาด gallery
    """
    sim = emb @ gallery_emb.T                       # (N, G) — ทุกอย่าง normalise มาแล้ว

    def reduce(block: np.ndarray) -> np.ndarray:
        if block.shape[1] == 0:
            return np.full(block.shape[0], -np.inf, dtype=np.float32)
        if aggregate == "max":
            return block.max(axis=1)
        if aggregate == "topk":
            k = min(topk, block.shape[1])
            return np.sort(block, axis=1)[:, -k:].mean(axis=1)
        raise ValueError(f"unknown gallery.aggregate {aggregate!r}")

    if aggregate == "proto":
        protos = []
        for c in range(n_classes):
            m = gallery_cls == c
            protos.append(gallery_emb[m].mean(axis=0) if m.any() else np.zeros(gallery_emb.shape[1]))
        known = emb @ l2norm(np.stack(protos)).T
        neg_mask = gallery_cls == unknown_class
        negative = None
        if neg_mask.any():
            # คลาส "ไม่ตรงสักอัน" ไม่ได้เป็นก้อนเดียวในสเปซ — ค่าเฉลี่ยจึงไม่มีความหมาย
            negative = _negative_score(sim[:, neg_mask], neg_quantile)
        return Scores(known.astype(np.float32), negative)

    known = np.stack([reduce(sim[:, gallery_cls == c]) for c in range(n_classes)], axis=1)
    neg_mask = gallery_cls == unknown_class
    negative = _negative_score(sim[:, neg_mask], neg_quantile) if neg_mask.any() else None
    return Scores(known.astype(np.float32), None if negative is None else negative.astype(np.float32))


def _negative_score(block: np.ndarray, quantile: float) -> np.ndarray:
    """Summarise 'how close is this image to *some* other brand'."""
    if quantile >= 1.0:
        return block.max(axis=1).astype(np.float32)
    return np.quantile(block, quantile, axis=1).astype(np.float32)


def expand_gallery(pool_emb: np.ndarray, gallery_emb: np.ndarray, gallery_cls: np.ndarray, *,
                   k: int, n_classes: int, unknown_class: int, aggregate: str, topk: int,
                   reject: str, threshold: float, ratio: float, neg_margin: float,
                   neg_quantile: float = 1.0, max_sim: float = 1.0,
                   expand_negatives: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Query expansion: fold the most confident unlabelled images back into the gallery.

    โลโก้ใน `queries/` เป็นภาพเกลี้ยง ๆ ภาพเดียวต่อคลาส แต่ภาพใน test ของแบรนด์เดียวกันมีหลายเวอร์ชัน
    (สีต่าง ครอปต่าง มีสโลแกนพ่วง) พอเราเอาภาพที่ทายได้มั่นใจที่สุดคลาสละ k รูปกลับเข้า gallery
    คลาสนั้นก็ครอบคลุมความหลากหลายมากขึ้น แล้วรอบสองจะจับตัวที่ก้ำกึ่งได้เพิ่ม

    เป็นเทคนิค transductive มาตรฐานของงาน retrieval — ใช้แค่ *ภาพ* ของ test ไม่ได้ใช้ label
    เลือกเฉพาะตัวที่ผ่านกฎ reject แล้วเท่านั้น เพื่อไม่ให้ความผิดพลาดสะสม
    """
    if k <= 0:
        return gallery_emb, gallery_cls

    scores = class_scores(pool_emb, gallery_emb, gallery_cls, n_classes=n_classes,
                          unknown_class=unknown_class, aggregate=aggregate, topk=topk,
                          neg_quantile=neg_quantile)
    # `reject="none"` = เลือกจาก argmax ล้วน ๆ โดยไม่ให้กฎ reject มาตัดใครออกก่อน
    # สำคัญเมื่อโลโก้อ้างอิงของคลาสไหน "คุณภาพแย่" (เช่น query 5 เป็น Starbucks ขาวดำ):
    # ภาพจริงของแบรนด์นั้นจะถูก negative gallery วีโต้ตั้งแต่รอบแรก แล้วไม่มีวันได้เข้า gallery
    # ทั้งที่มันคือรูปที่จะยกคะแนนของคลาสนั้นขึ้นมาได้
    out = decide(scores, reject=reject, threshold=threshold, ratio=ratio,
                 neg_margin=neg_margin, unknown_class=unknown_class)

    extra_idx, extra_cls = [], []
    for c in range(n_classes):
        picked = np.flatnonzero(out["pred"] == c)
        if len(picked) == 0:
            continue
        ranked = picked[np.argsort(out["top1_score"][picked])[::-1]]
        chosen = _diverse_topk(pool_emb, gallery_emb[gallery_cls == c], ranked, k, max_sim)
        extra_idx.extend(chosen)
        extra_cls.extend([c] * len(chosen))

    if expand_negatives:
        # ฝั่ง "ไม่ตรงสักอัน" กินพื้นที่กว้างกว่ามาก (แบรนด์อะไรก็ได้ในโลก) จึงรับได้มากกว่า
        # เรียงตามความมั่นใจว่าไม่ใช่ = ชนะคะแนนของคลาสที่ดีที่สุดไปเท่าไร
        picked = np.flatnonzero(out["pred"] == unknown_class)
        if len(picked):
            conf = out["neg_score"][picked] - out["top1_score"][picked]
            best = picked[np.argsort(conf)[::-1][: k * n_classes]]
            extra_idx.extend(best.tolist())
            extra_cls.extend([unknown_class] * len(best))

    if not extra_idx:
        log.info("query expansion: nothing confident enough to add")
        return gallery_emb, gallery_cls

    log.info("query expansion: +%d image(s) across %d class(es)",
             len(extra_idx), len(set(extra_cls)))
    return (np.concatenate([gallery_emb, pool_emb[extra_idx]], axis=0),
            np.concatenate([gallery_cls, np.array(extra_cls, dtype=gallery_cls.dtype)]))


def _diverse_topk(pool_emb: np.ndarray, class_emb: np.ndarray, ranked: np.ndarray,
                  k: int, max_sim: float) -> list[int]:
    """Take the k best candidates, skipping ones that duplicate what the class already has.

    ถ้าไม่กันตรงนี้ คลาสที่มีภาพซ้ำกับโลโก้อ้างอิงเยอะ ๆ จะใช้โควตา k ไปกับสำเนาของตัวเองจนหมด
    (คลาส 5 มี Starbucks ขาวดำแบบเดียวกับ query อยู่หลายสิบรูป คะแนน 1.00) แล้วเวอร์ชันจริง
    ที่หน้าตาต่างออกไป — Starbucks สีเขียว — ก็ไม่มีวันได้เข้า gallery ทั้งที่เป็นภาพที่มีค่าที่สุด
    """
    if max_sim >= 1.0:
        return ranked[:k].tolist()
    have = [class_emb] if len(class_emb) else []
    out: list[int] = []
    for i in ranked:
        if len(out) >= k:
            break
        v = pool_emb[i]
        if have and max(float((h @ v).max()) for h in have) > max_sim:
            continue
        out.append(int(i))
        have.append(v[None, :])
    return out


def normalise(scores: Scores, mode: str) -> Scores:
    """Optional per-class score normalisation.

    บางโลโก้ (เช่นตัวอักษรล้วน) เป็น "hub" ที่ได้คะแนนสูงกับทุกภาพ การหักค่าเฉลี่ยของแต่ละคลาส
    บนชุดที่กำลังทำนายอยู่ ช่วยให้คลาสแข่งกันอย่างเป็นธรรมขึ้น
    """
    if mode == "none":
        return scores
    if mode != "zscore":
        raise ValueError(f"unknown match.score_norm {mode!r}")
    k = scores.known
    z = (k - k.mean(axis=0, keepdims=True)) / np.maximum(k.std(axis=0, keepdims=True), 1e-6)
    neg = scores.negative
    if neg is not None:
        neg = (neg - neg.mean()) / max(float(neg.std()), 1e-6)
    return Scores(z.astype(np.float32), None if neg is None else neg.astype(np.float32))


def per_class_thresholds(known: np.ndarray, base: float, *, min_gap: float,
                         min_members: int = 8, hi: float = 0.97,
                         pool_emb: np.ndarray | None = None,
                         min_coherence: float = 0.0) -> np.ndarray:
    """Raise the threshold for individual classes whose low tail is one foreign brand.

    โลโก้อ้างอิงของแต่ละคลาสคุณภาพไม่เท่ากัน threshold เดียวจึงพอดีกับทุกคลาสไม่ได้ —
    คลาสที่ query คมชัด ภาพที่ตรงจริงเกาะอยู่ 0.94-1.00 ส่วนแบรนด์อื่นที่บังเอิญหน้าตาคล้ายกัน
    จะเกาะกลุ่มต่ำกว่านั้นอย่างชัดเจน แต่คลาสที่ query เบลอหรือขาวดำ ภาพที่ตรงจริงจะไล่ระดับ
    ต่อเนื่องลงมาถึง 0.84 การยก threshold จึงต้องทำเป็นรายคลาส และต้องมั่นใจก่อนว่าตัดถูกตัว

    ใช้เงื่อนไขสองชั้น ซึ่งอ่านได้จากข้อมูลที่ไม่มี label ทั้งคู่:

      1. **มีช่องว่าง** — คะแนนของคลาสนั้นแยกเป็นสองก้อนโดยห่างกันเกิน `min_gap`
      2. **ก้อนล่างเกาะกันแน่น** — รูปในก้อนล่างเหมือน *กันเอง* เกิน `min_coherence`

    ข้อ 2 คือตัวตัดสินจริง: แบรนด์แปลกปลอมที่หลุดเข้ามาจะมาเป็นฝูงของภาพโลโก้เดียวกัน
    (เหมือนกันเอง ~0.98) ส่วนหางที่ถูกต้องคือโลโก้เวอร์ชันต่าง ๆ ของแบรนด์เดียวกัน
    (แคมเปญคนละตัว สีคนละแบบ) ซึ่งกระจายกว่ามาก (~0.85) — ถ้าดูแค่ขนาดช่องว่างจะแยกสองกรณีนี้ไม่ออก
    """
    n_classes = known.shape[1]
    best, arg = known.max(axis=1), known.argmax(axis=1)
    out = np.full(n_classes, base, dtype=np.float64)
    if min_gap <= 0:
        return out

    for c in range(n_classes):
        idx = np.flatnonzero((arg == c) & (best >= base))
        if len(idx) < min_members:
            continue
        order = idx[np.argsort(best[idx])[::-1]]
        s = best[order]
        gaps = s[:-1] - s[1:]
        # จุดตัดต้องอยู่ใต้ `hi` ไม่งั้นจะไปตัดกลุ่มภาพที่แทบเป็นไฟล์เดียวกับ query
        usable = s[:-1] <= hi
        if not usable.any():
            continue
        i = int(np.flatnonzero(usable)[np.argmax(gaps[usable])])
        if gaps[i] < min_gap:
            continue
        if min_coherence > 0:
            below = order[i + 1:]
            if len(below) < 2 or pool_emb is None:
                continue
            block = pool_emb[below] @ pool_emb[below].T
            np.fill_diagonal(block, np.nan)
            if float(np.nanmean(block)) < min_coherence:
                continue
        out[c] = float((s[i] + s[i + 1]) / 2)
    return out


def decide(scores: Scores, *, reject: str, threshold: float | np.ndarray, ratio: float,
           neg_margin: float, unknown_class: int) -> dict[str, np.ndarray]:
    """Apply the reject rule(s) and return predictions plus the numbers behind them.

    `threshold` เป็นตัวเลขเดียว หรือเป็น array ยาวเท่าจำนวนคลาสก็ได้ (per-class threshold)
    """
    known = scores.known
    order = np.argsort(known, axis=1)[:, ::-1]
    best = order[:, 0]
    best_s = np.take_along_axis(known, order[:, :1], axis=1)[:, 0]
    second_s = np.take_along_axis(known, order[:, 1:2], axis=1)[:, 0]

    rules = {r for r in reject.split("+") if r and r != "none"}
    unknown_rules = rules - {"threshold", "ratio", "gallery"}
    if unknown_rules:
        raise ValueError(f"unknown match.reject rule(s) {sorted(unknown_rules)}")

    rejected = np.zeros(len(known), dtype=bool)
    if "threshold" in rules:
        thr = np.asarray(threshold)[best] if np.ndim(threshold) else threshold
        rejected |= best_s < thr
    if "ratio" in rules:
        rejected |= best_s < ratio * second_s
    if "gallery" in rules:
        if scores.negative is None:
            raise ValueError(
                "match.reject asks for 'gallery' but the gallery has no class-22 items — "
                "check gallery.sources / query_map.yaml"
            )
        rejected |= scores.negative >= best_s - neg_margin

    pred = np.where(rejected, unknown_class, best)
    return {
        "pred": pred.astype(int),
        "top1": best.astype(int),
        "top1_score": best_s,
        "top2": order[:, 1].astype(int),
        "top2_score": second_s,
        "neg_score": scores.negative if scores.negative is not None else np.full(len(known), np.nan),
        "rejected": rejected,
    }


def evaluate(pred: np.ndarray, truth: np.ndarray, *, n_classes: int,
             unknown_class: int) -> dict[str, float]:
    """Accuracy overall, on the 22 real classes, and on the reject class."""
    known_mask = truth != unknown_class
    per_class_f1 = []
    for c in list(range(n_classes)) + [unknown_class]:
        tp = int(((pred == c) & (truth == c)).sum())
        fp = int(((pred == c) & (truth != c)).sum())
        fn = int(((pred != c) & (truth == c)).sum())
        if tp + fn == 0:
            continue
        per_class_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
    return {
        "accuracy": float((pred == truth).mean()),
        "macro_f1": float(np.mean(per_class_f1)) if per_class_f1 else 0.0,
        "known_acc": float((pred[known_mask] == truth[known_mask]).mean()) if known_mask.any() else float("nan"),
        "unknown_recall": float((pred[~known_mask] == unknown_class).mean()) if (~known_mask).any() else float("nan"),
        "n": int(len(truth)),
    }


def sweep_threshold(scores: Scores, truth: np.ndarray, *, reject: str, ratio: float,
                    neg_margin: float, unknown_class: int,
                    n_grid: int = 200) -> tuple[float, dict[str, float]]:
    """Pick the absolute threshold that maximises accuracy on a labelled split."""
    if "threshold" not in reject:
        metrics = evaluate(decide(scores, reject=reject, threshold=0.0, ratio=ratio,
                                  neg_margin=neg_margin, unknown_class=unknown_class)["pred"],
                           truth, n_classes=scores.known.shape[1], unknown_class=unknown_class)
        return float("nan"), metrics

    best_s = scores.known.max(axis=1)
    lo, hi = float(np.percentile(best_s, 1)), float(np.percentile(best_s, 99))
    grid = np.linspace(lo, hi, n_grid)

    best = (-1.0, float("nan"), {})
    for t in grid:
        out = decide(scores, reject=reject, threshold=float(t), ratio=ratio,
                     neg_margin=neg_margin, unknown_class=unknown_class)
        m = evaluate(out["pred"], truth, n_classes=scores.known.shape[1], unknown_class=unknown_class)
        if m["accuracy"] > best[0]:
            best = (m["accuracy"], float(t), m)
    return best[1], best[2]
