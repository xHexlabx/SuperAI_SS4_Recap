"""End-to-end check that needs no GPU, no LLM server and no network.

Runs the matching engine against hand-written cases first (so a broken evaluator is caught
before anything expensive starts), then compiles every real clause with the regex baseline
and scores it on train.

    uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legal_act.baseline import build_rules  # noqa: E402
from legal_act.config import load_config  # noqa: E402
from legal_act.data import load_committees, load_split, parse_signers  # noqa: E402
from legal_act.engine import score  # noqa: E402
from legal_act.rules import Branch, ClauseRule, Slot, decide  # noqa: E402
from legal_act.utils import normalise_name, thai_number  # noqa: E402

A, B, C, D = "อรุชา เขมทโรนนท์", "ภาณุพงษ์ สภานนท์", "ชิติพัทธ์ สร้อยสังวาลย์", "อธิบดี คุ่ยประเสริฐ"
DIRECTORS = {normalise_name(n) for n in (A, B, C, D)}

CASES = [
    # "กรรมการคนใดคนหนึ่งลงลายมือชื่อ"
    ("any one director", Branch(slots=[Slot()], total=1), [([A], True), ([A, B], False), ([], False)]),
    # "กรรมการสองในสี่คนลงลายมือชื่อร่วมกัน" — exactly two, and two *different* people
    ("two of any four", Branch(slots=[Slot(), Slot()], total=2),
     [([A, B], True), ([A], False), ([A, B, C], False), ([A, A], False)]),
    # "(ก หรือ ข) ลงลายมือชื่อร่วมกับ (ค หรือ ง) รวมเป็นสองคน"
    ("one from each group", Branch(slots=[Slot(names=(A, B)), Slot(names=(C, D))], total=2),
     [([A, C], True), ([C, A], True), ([A, B], False), ([C, D], False), ([A], False)]),
    # "ก และ ข ลงลายมือชื่อร่วมกับกรรมการอื่นอีกหนึ่งคน รวมเป็นสามคน"
    ("two named plus any other", Branch(slots=[Slot(names=(A,)), Slot(names=(B,)), Slot()], total=3),
     [([A, B, C], True), ([A, B, D], True), ([A, C, D], False), ([A, B], False)]),
    # a slot list shorter than `total` is padded with "any director" slots
    ("padded total", Branch(slots=[Slot(names=(A,))], total=2),
     [([A, B], True), ([A], False), ([B, C], False)]),
]


def check_engine() -> None:
    failures = 0
    for name, branch, expectations in CASES:
        rule = ClauseRule(clause_id="t", rg="0", branches=[branch])
        for signers, want in expectations:
            got = decide(rule, signers, "การทำนิติกรรม", DIRECTORS)
            if got != want:
                failures += 1
                print(f"  ✗ {name}: {signers} -> {got}, expected {want}")
    # scope handling
    only = Branch(slots=[Slot(names=(A,))], total=1, mode="only", scope="เช็ค", acts=["การลงนามในเช็ค"])
    rule = ClauseRule(clause_id="t", rg="0", branches=[only])
    for act, want in [("การลงนามในเช็ค", True), ("การขอวีซ่า", False)]:
        if decide(rule, [A], act, DIRECTORS) != want:
            failures += 1
            print(f"  ✗ scope only: {act} -> expected {want}")

    assert thai_number("สอง") == 2 and thai_number("สิบเอ็ด") == 11 and thai_number("3") == 3
    assert normalise_name("นายอธิบดี คุ่ยประเสริฐ") == normalise_name("อธิบดี  คุ่ยประเสริฐ ")
    assert parse_signers("['ก ข', 'ค ง']") == ["ก ข", "ค ง"]

    if failures:
        raise SystemExit(f"engine check failed: {failures} case(s)")
    print(f"  ✓ engine: {sum(len(c[2]) for c in CASES) + 2} cases pass")


def main() -> int:
    print("1) rule engine")
    check_engine()

    print("2) regex baseline on the real data")
    cfg = load_config()
    if not (cfg.resolve(cfg.data.root) / cfg.data.train_csv).exists():
        print("  ! datasets/train.csv missing — run `make data` to score the baseline")
        return 0
    frames = [load_split(cfg, s) for s in ("train", "test")]
    committees = load_committees(cfg)
    rules = build_rules(cfg, frames, committees)
    stats = score(frames[0], rules, committees)
    print(f"  ✓ {len(rules)} clauses compiled, train acc {stats['accuracy']:.4f}, "
          f"{stats['solved_clauses']}/{stats['total_clauses']} clauses solved")
    print("\nall good — `make compile` next (needs a Qwen endpoint).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
