"""Stage A/B/C: turn 308 clause texts into rules, then let train labels fix what is wrong.

Why per clause and not per row: `context` repeats. 10,264 rows across train+test contain only
308 distinct (company, clause) pairs, so asking the model once per clause is 33x cheaper than
asking once per row — and it makes the pipeline self-consistent, because two rows quoting the
same clause can no longer get contradictory answers.
"""

from __future__ import annotations

import logging

import pandas as pd

from .data import (Committee, clause_table, legal_acts_per_clause, load_committees,
                   load_patterns, load_split)
from .engine import save_rules, score, snap_rule_names
from .llm import ChatClient
from .prompts import (RULE_SCHEMA, SCOPE_SCHEMA, compile_messages, repair_messages,
                      scope_messages)
from .rules import ALWAYS, Branch, ClauseRule

log = logging.getLogger("legal_act.compile")


def _to_rule(clause_id: str, rg: str, raw: dict) -> ClauseRule:
    branches = [Branch.from_dict(b) for b in raw.get("branches", [])]
    branches = [b for b in branches if b.slots or b.total]
    if not branches:
        return ClauseRule(clause_id=clause_id, rg=rg, error="model returned no usable branch")
    return ClauseRule(clause_id=clause_id, rg=rg, branches=branches, note=str(raw.get("note", "")))


def compile_clauses(cfg, client: ChatClient, clauses: pd.DataFrame,
                    committees: dict[str, Committee], patterns: dict[int, str]) -> dict[str, ClauseRule]:
    """Stage A — one call per clause, then snap the names onto real directors."""
    rows = clauses.to_dict("records")

    def build(row):
        c = committees.get(row["rg"])
        return compile_messages(
            context=row["context"],
            directors=c.names if c else [],
            pattern=int(row["pattern"]) if pd.notna(row["pattern"]) else None,
            template=patterns.get(int(row["pattern"])) if pd.notna(row["pattern"]) else None,
            conditions=row["conditions"],
        )

    replies = client.map_json(rows, build, RULE_SCHEMA, desc="compile")

    rules: dict[str, ClauseRule] = {}
    snapped = 0
    for row, reply in zip(rows, replies):
        cid, rg = row["clause_id"], row["rg"]
        if reply is None:
            rules[cid] = ClauseRule(clause_id=cid, rg=rg, error="no usable reply from the model")
            continue
        try:
            rule = _to_rule(cid, rg, reply)
        except Exception as exc:  # noqa: BLE001 — a bad reply is data, not a crash
            rules[cid] = ClauseRule(clause_id=cid, rg=rg, error=f"unusable reply: {exc}")
            continue
        if cfg.compile.snap_names and rule.ok:
            snapped += snap_rule_names(rule, committees.get(rg), cfg.compile.snap_cutoff)
        rules[cid] = rule

    bad = [r for r in rules.values() if not r.ok]
    log.info("stage A: %d/%d clauses compiled (%d names snapped, %d failed)",
             len(rules) - len(bad), len(rules), snapped, len(bad))
    return rules


def resolve_scopes(cfg, client: ChatClient, rules: dict[str, ClauseRule],
                   clauses: pd.DataFrame, acts_by_clause: dict[str, list[str]]) -> None:
    """Stage B — decide which legal acts each conditional branch covers.

    Only clauses that actually have a conditional branch go through this, which is roughly a
    third of them; the rest are answered by their single always-on rule.
    """
    ctx_by_id = dict(zip(clauses["clause_id"], clauses["context"]))
    jobs = []
    for cid, rule in rules.items():
        if not rule.ok or not rule.needs_scope:
            continue
        acts = acts_by_clause.get(cid, [])
        if not acts:
            continue
        idx = [i for i, b in enumerate(rule.branches) if b.mode != ALWAYS]
        jobs.append({"clause_id": cid, "acts": acts, "branch_idx": idx})

    if not jobs:
        log.info("stage B: no conditional clause to resolve")
        return

    def build(job):
        rule = rules[job["clause_id"]]
        brief = [{"index": i, "mode": rule.branches[i].mode, "scope": rule.branches[i].scope}
                 for i in job["branch_idx"]]
        return scope_messages(ctx_by_id[job["clause_id"]], brief, job["acts"])

    replies = client.map_json(jobs, build, SCOPE_SCHEMA, desc="scope  ")

    resolved = 0
    for job, raw in zip(jobs, replies):
        rule = rules[job["clause_id"]]
        acts = job["acts"]
        if raw is None:
            continue
        for item in raw.get("branches", []):
            i = int(item.get("index", -1))
            if not 0 <= i < len(rule.branches):
                continue
            picked = []
            for a in item.get("acts", []):
                if isinstance(a, (int, float)) and 1 <= int(a) <= len(acts):
                    picked.append(acts[int(a) - 1])
                elif isinstance(a, str) and a in acts:
                    picked.append(a)
            rule.branches[i].acts = sorted(set(picked))
        resolved += 1
    log.info("stage B: resolved scopes for %d/%d conditional clauses", resolved, len(jobs))


def repair(cfg, client: ChatClient, rules: dict[str, ClauseRule], train: pd.DataFrame,
           clauses: pd.DataFrame, committees: dict[str, Committee],
           patterns: dict[int, str], rounds: int) -> None:
    """Stage C — use the train labels as a test suite for the compiler.

    The labels are a deterministic function of (clause, signers, legal act) — 4,233 distinct
    question groups in train, zero of them contradictory — so a clause that scores below 1.00
    means the rule is wrong, not that the data is noisy. That makes a repair loop worth running.
    """
    clause_rows = {cid: g for cid, g in train.groupby("clause_id")}
    meta = clauses.set_index("clause_id").to_dict("index")

    for rnd in range(rounds):
        stats = score(train, rules, committees)
        broken = [cid for cid, row in stats["per_clause"].iterrows()
                  if row["acc"] < 1.0 and cid in clause_rows and rules.get(cid, None) is not None]
        log.info("repair round %d: train acc %.4f, %d/%d clauses solved, %d to repair",
                 rnd + 1, stats["accuracy"], stats["solved_clauses"], stats["total_clauses"], len(broken))
        if not broken:
            return

        jobs = []
        for cid in broken:
            g = clause_rows[cid]
            from .engine import predict_rows
            pred = predict_rows(g, rules, committees)
            bad = g.loc[pred != g["answer"].astype(int)]
            failures = [
                {"signers": list(r["signers"]), "legal_act": r["legal_act"],
                 "truth": int(r["answer"]), "pred": int(pred.loc[i])}
                for i, r in bad.head(12).iterrows()
            ]
            jobs.append({"clause_id": cid, "failures": failures})

        def build(job):
            cid = job["clause_id"]
            m = meta[cid]
            c = committees.get(m["rg"])
            return repair_messages(
                context=m["context"],
                directors=c.names if c else [],
                pattern=int(m["pattern"]) if pd.notna(m["pattern"]) else None,
                template=patterns.get(int(m["pattern"])) if pd.notna(m["pattern"]) else None,
                conditions=m["conditions"],
                rule=rules[cid].to_dict(),
                failures=job["failures"],
            )

        replies = client.map_json(jobs, build, RULE_SCHEMA, desc=f"repair{rnd + 1}")

        for job, reply in zip(jobs, replies):
            cid = job["clause_id"]
            if reply is None:
                continue
            try:
                candidate = _to_rule(cid, rules[cid].rg, reply)
            except Exception:  # noqa: BLE001
                continue
            if not candidate.ok:
                continue
            if cfg.compile.snap_names:
                snap_rule_names(candidate, committees.get(candidate.rg), cfg.compile.snap_cutoff)
            # carry over the resolved scopes where the branch layout is unchanged
            old = rules[cid]
            if len(old.branches) == len(candidate.branches):
                for ob, nb in zip(old.branches, candidate.branches):
                    if not nb.acts and nb.mode == ob.mode:
                        nb.acts = ob.acts
            # keep the repair only if it is an improvement on this clause
            g = clause_rows[cid]
            before = score(g, {cid: old}, committees)["accuracy"]
            after = score(g, {cid: candidate}, committees)["accuracy"]
            if after > before:
                rules[cid] = candidate

    stats = score(train, rules, committees)
    log.info("after repair: train acc %.4f, %d/%d clauses solved",
             stats["accuracy"], stats["solved_clauses"], stats["total_clauses"])


def run(cfg) -> dict[str, ClauseRule]:
    frames = {s: load_split(cfg, s) for s in cfg.compile.splits}
    committees = load_committees(cfg)
    patterns = load_patterns(cfg)
    clauses = clause_table(list(frames.values()))
    log.info("%d rows -> %d unique clauses (%.0fx fewer llm calls)",
             sum(len(f) for f in frames.values()), len(clauses),
             sum(len(f) for f in frames.values()) / max(1, len(clauses)))

    client = ChatClient(cfg)
    log.info("llm: %s", client.ping())

    rules = compile_clauses(cfg, client, clauses, committees, patterns)

    if cfg.compile.scope_stage:
        resolve_scopes(cfg, client, rules, clauses, legal_acts_per_clause(list(frames.values())))

    if "train" in frames:
        stats = score(frames["train"], rules, committees)
        log.info("before repair: train acc %.4f, %d/%d clauses solved",
                 stats["accuracy"], stats["solved_clauses"], stats["total_clauses"])
        if cfg.compile.self_repair_rounds > 0:
            repair(cfg, client, rules, frames["train"], clauses, committees, patterns,
                   cfg.compile.self_repair_rounds)

    save_rules(rules, cfg.resolve(cfg.compile.out))
    return rules
