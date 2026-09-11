"""Figures for eyeballing the data and the pipeline's decisions.

    uv run python scripts/visualize.py queries      # โลโก้ 22 รูปที่โจทย์ถาม
    uv run python scripts/visualize.py map          # query ← โฟลเดอร์ที่จับคู่ได้ (ตรวจ query_map.yaml)
    uv run python scripts/visualize.py test         # ตัวอย่างภาพ test พร้อมคลาสที่ทำนาย
    uv run python scripts/visualize.py errors       # ภาพที่ทำนายผิดบน validation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from img_search import benchmark as benchmark_mod  # noqa: E402
from img_search import data as data_mod  # noqa: E402
from img_search import match as match_mod  # noqa: E402
from img_search import proxy as proxy_mod  # noqa: E402
from img_search.config import load_config  # noqa: E402
from img_search.mapping import load_map  # noqa: E402
from img_search.utils import ensure_dir, setup_logging  # noqa: E402

FIG_DIR = "models/figures"


def _show(ax, path: str, title: str = "", color: str = "black") -> None:
    with Image.open(path) as im:
        ax.imshow(im.convert("RGB").resize((160, 160), Image.BICUBIC))
    ax.set_title(title, fontsize=7, color=color)
    ax.axis("off")


def _save(fig, cfg, name: str) -> Path:
    out = ensure_dir(cfg.resolve(FIG_DIR)) / name
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out


def fig_queries(cfg) -> None:
    q = data_mod.queries_index(cfg)
    fig, axes = plt.subplots(2, 11, figsize=(22, 4.4))
    for ax, row in zip(axes.ravel(), q.itertuples()):
        _show(ax, row.path, f"class {row.cls}")
    fig.suptitle("queries/ — one reference logo per class", fontsize=12)
    _save(fig, cfg, "queries.png")


def fig_map(cfg, n_examples: int = 5) -> None:
    mapping = load_map(cfg)
    q = data_mod.queries_index(cfg).set_index("cls")
    train = data_mod.train_index(cfg)

    rows = sorted(mapping)
    fig, axes = plt.subplots(len(rows), n_examples + 1,
                             figsize=(1.5 * (n_examples + 1), 1.6 * len(rows)))
    for r, cls in enumerate(rows):
        folder = mapping[cls]
        _show(axes[r, 0], q.loc[cls, "path"], f"query {cls}", "tab:blue")
        pool = train[train["folder"] == folder]["path"].tolist() if folder else []
        for c in range(n_examples):
            ax = axes[r, c + 1]
            if c < len(pool):
                _show(ax, pool[c], folder if c == 0 else "")
            else:
                ax.axis("off")
                if c == 0:
                    ax.set_title("(unmatched)", fontsize=7, color="tab:red")
    fig.suptitle("query_map.yaml — reference logo (left) vs. the train/ folder it matched", fontsize=12)
    _save(fig, cfg, "query_map.png")


def fig_test(cfg, n: int = 40, seed: int = 0) -> None:
    debug = cfg.resolve(cfg.predict.debug_csv or "models/predict_debug.csv")
    if not debug.exists():
        raise SystemExit(f"{debug} not found — run `make predict` first")
    import pandas as pd

    df = pd.read_csv(debug)
    test = data_mod.test_index(cfg).set_index(cfg.data.id_col)
    df = df.sample(min(n, len(df)), random_state=seed)

    cols = 10
    rowsn = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(1.6 * cols, 1.8 * rowsn))
    # itertuples() ตั้งชื่อฟิลด์ใหม่เมื่อคอลัมน์ชนคำสงวนของ Python — และคอลัมน์นี้ชื่อ "class"
    for ax, (_, row) in zip(axes.ravel(), df.iterrows()):
        cls = row[cfg.data.label_col]
        color = "tab:red" if cls == cfg.data.unknown_class else "tab:green"
        _show(ax, test.loc[row[cfg.data.id_col], "path"],
              f"{cls}  ({row['top1_score']:.2f})", color)
    for ax in axes.ravel()[len(df):]:
        ax.axis("off")
    fig.suptitle("test/ — predicted class (red = 22, no match)", fontsize=12)
    _save(fig, cfg, "test_predictions.png")


def fig_errors(cfg, n: int = 40) -> None:
    """Mistakes on proxy fold 0 — the fastest way to see *how* an encoder is failing."""
    fold = proxy_mod.build_folds(cfg, n_folds=1, seed=cfg.benchmark.seed,
                                 min_images=cfg.benchmark.min_images,
                                 seen_ratio=cfg.benchmark.seen_ratio,
                                 neg_gallery_frac=cfg.benchmark.neg_gallery_frac)[0]
    scores, truth, _ = benchmark_mod._fold_scores(cfg, fold, cfg.embed.models)
    thr = benchmark_mod.evaluate_model(cfg, cfg.embed.models, [fold])["threshold_production"]
    out = match_mod.decide(scores, reject=cfg.match.reject, threshold=thr, ratio=cfg.match.ratio,
                           neg_margin=cfg.match.neg_margin, unknown_class=cfg.data.unknown_class)

    wrong = np.flatnonzero(out["pred"] != truth)[:n]
    print(f"fold 0 accuracy {float((out['pred'] == truth).mean()):.4f} — showing {len(wrong)} of "
          f"{int((out['pred'] != truth).sum())} error(s)")
    if len(wrong) == 0:
        return

    cols = 10
    rowsn = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rowsn, cols, figsize=(1.6 * cols, 2.1 * rowsn))
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        want = fold.brands.get(int(truth[i]), "22 / none")
        got = fold.brands.get(int(out["pred"][i]), "22 / none")
        _show(ax, fold.val.loc[i, "path"], f"true {want}\npred {got}\n{out['top1_score'][i]:.2f}",
              "tab:red")
    for ax in np.atleast_1d(axes).ravel()[len(wrong):]:
        ax.axis("off")
    fig.suptitle(f"proxy-fold errors ({'+'.join(cfg.embed.models)})", fontsize=12)
    _save(fig, cfg, "errors.png")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["queries", "map", "test", "errors"])
    p.add_argument("--config", default=None)
    p.add_argument("--set", dest="overrides", action="append", default=[])
    p.add_argument("-n", type=int, default=None, help="how many images to show")
    args = p.parse_args()
    setup_logging()

    cfg = load_config(args.config, args.overrides)
    if args.what == "queries":
        fig_queries(cfg)
    elif args.what == "map":
        fig_map(cfg, args.n or 5)
    elif args.what == "test":
        fig_test(cfg, args.n or 40)
    else:
        fig_errors(cfg, args.n or 40)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
