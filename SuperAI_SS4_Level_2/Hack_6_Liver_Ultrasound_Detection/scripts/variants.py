"""Diagnostic submission variants from a run's cached test predictions (to learn what the scorer does).

usage: uv run python scripts/variants.py models/<run> <preds_test_*.pkl name> <out_prefix>
"""
import pickle, random, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from liver_us.data import load_index
from liver_us.metric import to_submission
from liver_us.predict import apply_threshold, scale_to_original

run, pkl, prefix = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
index = load_index()
test_ids = index[index.split == "test"].image_id.tolist()
tp = pickle.loads((run / pkl).read_bytes())
main_thr, main_cls = 0.30, {0: 0.2, 1: 0.3, 2: 0.5, 3: 0.25, 4: 0.4, 5: 0.25, 6: 0.4}

def write(name, preds, ids=test_ids):
    df = to_submission(scale_to_original(preds, index), ids); df["Image File"] = df["Image File"].astype(int)
    out = Path("submissions") / f"{prefix}_{name}.csv"; df.to_csv(out, index=False)
    print(name, sum(len(v) for v in preds.values()), "boxes ->", out)

write("allboxes_c005", apply_threshold(tp, 0.05, 100))          # many low-conf boxes: hurts if scorer ignores confidence
write("allboxes_c001", apply_threshold(tp, 0.001, 100))
random.seed(0); ids = list(test_ids); random.shuffle(ids)
write("main_shuffled_rows", apply_threshold(tp, main_thr, 5, main_cls), ids)   # row order test
write("global_thr030", apply_threshold(tp, 0.30, 5))            # per-class thresholds vs one global
write("thr030_max1", apply_threshold(tp, 0.30, 1))              # 1 box per image
