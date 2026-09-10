"""A no-LLM compiler, for a reference number and for smoke-testing the engine.

It reads only the numbers out of a clause — "สองในห้าคน", "รวมเป็นสามคน", "ทุกคน" — plus which
directors are mentioned by name, and ignores every conditional branch. That is roughly what a
careful regex baseline can do, and the gap between it and `make compile` is what the LLM buys.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .data import (Committee, clause_table, load_committees, load_patterns, load_split)
from .engine import save_rules, score
from .rules import Branch, ClauseRule, Slot
from .utils import normalise_name, thai_number

log = logging.getLogger("legal_act.baseline")

NUM = r"(\d+|ศูนย์|หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ|สิบเอ็ด|ยี่สิบ)"


def _num(text: str | None) -> int | None:
    return thai_number(text) if text else None


def named_directors(context: str, committee: Committee | None) -> list[str]:
    """Which registered directors are actually written into the clause text."""
    if committee is None:
        return []
    flat = normalise_name(context)
    return [n for n in committee.names if normalise_name(n) in flat]


def compile_one(clause_id: str, rg: str, context: str, committee: Committee | None) -> ClauseRule:
    n_directors = len(committee) if committee else 0
    text = re.sub(r"\s+", "", str(context))

    total: int | None = None
    at_least = False

    if re.search(r"กรรมการทุกคน|กรรมการทั้งหมด", text) and n_directors:
        total = n_directors
    if total is None:
        m = re.search(NUM + r"ใน" + NUM + r"คน", text)
        if m:
            total = _num(m.group(1))
    if total is None:
        m = re.search(r"รวมเป็น" + NUM + r"คน", text)
        if m:
            total = _num(m.group(1))
    if total is None:
        m = re.search(r"อย่างน้อย" + NUM + r"คน", text)
        if m:
            total, at_least = _num(m.group(1)), True
    if total is None:
        m = re.search(NUM + r"คนขึ้นไป", text)
        if m:
            total, at_least = _num(m.group(1)), True
    if total is None:
        m = re.search(r"กรรมการ" + NUM + r"คนลงลายมือชื่อ", text)
        if m:
            total = _num(m.group(1))
    if total is None and re.search(r"คนใดคนหนึ่ง|หนึ่งคนลงลายมือชื่อ|ผู้จัดการลงลายมือชื่อ", text):
        total = 1
    if total is None:
        total = 2  # the most common size when the clause does not say

    names = named_directors(context, committee)
    # A clause that lists names but joins them with "และ" wants all of them, not a choice.
    joined_and = bool(names) and "หรือ" not in text

    def slots_for(k: int) -> list[Slot]:
        if not names:
            return [Slot() for _ in range(k)]
        if joined_and and len(names) == k:
            return [Slot(names=(n,)) for n in names]
        return [Slot(names=tuple(names)) for _ in range(k)]

    sizes = range(total, n_directors + 1) if (at_least and n_directors >= total) else [total]
    branches = [Branch(slots=slots_for(k), total=k) for k in sizes if k > 0]
    return ClauseRule(clause_id=clause_id, rg=rg, branches=branches, note="regex baseline")


def build_rules(cfg, frames: list[pd.DataFrame], committees: dict[str, Committee]) -> dict[str, ClauseRule]:
    clauses = clause_table(frames)
    return {
        row["clause_id"]: compile_one(row["clause_id"], row["rg"], row["context"],
                                      committees.get(row["rg"]))
        for row in clauses.to_dict("records")
    }


def run(cfg) -> dict[str, ClauseRule]:
    frames = [load_split(cfg, s) for s in cfg.compile.splits]
    committees = load_committees(cfg)
    load_patterns(cfg)  # fail early if the file is missing
    rules = build_rules(cfg, frames, committees)

    train = next((f for f in frames if f["split"].iloc[0] == "train"), None)
    if train is not None:
        stats = score(train, rules, committees)
        log.info("regex baseline: train acc %.4f, %d/%d clauses solved",
                 stats["accuracy"], stats["solved_clauses"], stats["total_clauses"])

    save_rules(rules, cfg.resolve(cfg.compile.out))
    return rules
