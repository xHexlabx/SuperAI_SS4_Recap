"""Stage A/B/C: turn 308 clause texts into rules, then let train labels fix what is wrong.

Why per clause and not per row: `context` repeats. 10,264 rows across train+test contain only
308 distinct (company, clause) pairs, so asking the model once per clause is 33x cheaper than
asking once per row — and it makes the pipeline self-consistent, because two rows quoting the
same clause can no longer get contradictory answers.
"""

from __future__ import annotations

import logging

import pandas as pd

from .data import (Committee, asked_sizes_per_clause, clause_table, legal_acts_per_clause,
                   load_committees, load_patterns, load_split)
from .engine import save_rules, score, snap_rule_names
from .llm import ChatClient
from .prompts import (RULE_SCHEMA, SCOPE_SCHEMA, compile_messages, consistency_messages,
                      repair_messages, scope_messages)
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
    """Stage B — decide which legal acts each carved-out situation covers.

    Resolved once per DISTINCT scope text, not once per branch: an "except" branch and its
    "only" twin share a scope, and asking about them separately let the model answer them
    differently (one clause came back with 1 act on the "only" side and 22 on the "except"
    side — the general rule was switched off for 21 acts nobody had excluded). After the model
    answers, `_harmonise` makes every "except" branch exclude exactly what the "only" branches
    claim, which is what an exception means.
    """
    ctx_by_id = dict(zip(clauses["clause_id"], clauses["context"]))
    jobs = []
    for cid, rule in rules.items():
        if not rule.ok or not rule.needs_scope:
            continue
        acts = acts_by_clause.get(cid, [])
        scopes = sorted({b.scope.strip() for b in rule.branches if b.mode != ALWAYS and b.scope.strip()})
        if not acts or not scopes:
            continue
        jobs.append({"clause_id": cid, "acts": acts, "scopes": scopes})

    if not jobs:
        log.info("stage B: no conditional clause to resolve")
        return

    def build(job):
        brief = [{"index": i, "scope": sc} for i, sc in enumerate(job["scopes"])]
        return scope_messages(ctx_by_id[job["clause_id"]], brief, job["acts"])

    replies = client.map_json(jobs, build, SCOPE_SCHEMA, desc="scope  ")

    resolved = 0
    for job, raw in zip(jobs, replies):
        rule = rules[job["clause_id"]]
        acts, scopes = job["acts"], job["scopes"]
        if raw is None:
            continue
        acts_of_scope: dict[str, list[str]] = {}
        for item in raw.get("branches", []):
            i = int(item.get("index", -1))
            if not 0 <= i < len(scopes):
                continue
            picked = []
            for a in item.get("acts", []):
                if isinstance(a, (int, float)) and 1 <= int(a) <= len(acts):
                    picked.append(acts[int(a) - 1])
                elif isinstance(a, str) and a in acts:
                    picked.append(a)
            acts_of_scope[scopes[i]] = sorted(set(picked))
        for b in rule.branches:
            if b.mode != ALWAYS and b.scope.strip() in acts_of_scope:
                b.acts = list(acts_of_scope[b.scope.strip()])
        resolved += 1

    for rule in rules.values():
        if rule.ok:
            _harmonise(rule)
    log.info("stage B: resolved scopes for %d/%d conditional clauses", resolved, len(jobs))


def _harmonise(rule: ClauseRule) -> None:
    """An "except" branch withdraws the general rule exactly where an "only" branch takes over."""
    from .rules import EXCEPT, ONLY

    only_acts: set[str] = set()
    for b in rule.branches:
        if b.mode == ONLY:
            only_acts.update(b.acts)
    if not only_acts:
        return
    for b in rule.branches:
        if b.mode == EXCEPT:
            b.acts = sorted(only_acts)


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
        stats = score(train, rules, committees, semantics=cfg.compile.semantics)
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
            pred = predict_rows(g, rules, committees, semantics=cfg.compile.semantics)
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
            before = score(g, {cid: old}, committees, semantics=cfg.compile.semantics)["accuracy"]
            after = score(g, {cid: candidate}, committees, semantics=cfg.compile.semantics)["accuracy"]
            if after > before:
                rules[cid] = candidate

    stats = score(train, rules, committees, semantics=cfg.compile.semantics)
    log.info("after repair: train acc %.4f, %d/%d clauses solved",
             stats["accuracy"], stats["solved_clauses"], stats["total_clauses"])


def consistency_repair(cfg, client: ChatClient, rules: dict[str, ClauseRule],
                       clauses: pd.DataFrame, asked: dict[str, list[int]],
                       committees: dict[str, Committee], patterns: dict[int, str],
                       train: pd.DataFrame | None, rounds: int) -> None:
    """Repair rules that contradict their own questions — no labels involved.

    The question generator never asks for more signatures than a clause can accept (true for
    88 of the 95 train clauses whose rule is provably correct), so a rule whose largest total
    is below the largest set asked has missed a way of signing. That check reads only the test
    *inputs*, so it fixes test clauses the labelled repair loop can never reach.
    """
    meta = clauses.set_index("clause_id").to_dict("index")
    train_rows = {cid: g for cid, g in train.groupby("clause_id")} if train is not None else {}

    def violating() -> list[str]:
        out = []
        for cid, rule in rules.items():
            if not rule.ok or cid not in asked:
                continue
            if max(asked[cid]) > max(b.required for b in rule.branches):
                out.append(cid)
        return out

    for rnd in range(rounds):
        bad = violating()
        log.info("consistency round %d: %d/%d clauses cannot accept the biggest set asked",
                 rnd + 1, len(bad), len(rules))
        if not bad:
            return

        def build(cid):
            m = meta[cid]
            c = committees.get(m["rg"])
            return consistency_messages(
                context=m["context"],
                directors=c.names if c else [],
                pattern=int(m["pattern"]) if pd.notna(m["pattern"]) else None,
                template=patterns.get(int(m["pattern"])) if pd.notna(m["pattern"]) else None,
                conditions=m["conditions"],
                rule=rules[cid].to_dict(),
                asked=asked[cid],
                totals=[b.required for b in rules[cid].branches],
            )

        replies = client.map_json(bad, build, RULE_SCHEMA, desc=f"consist{rnd + 1}")

        kept = 0
        for cid, reply in zip(bad, replies):
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
            old = rules[cid]
            if len(old.branches) == len(candidate.branches):
                for ob, nb in zip(old.branches, candidate.branches):
                    if not nb.acts and nb.mode == ob.mode:
                        nb.acts = ob.acts
            # the new rule has to actually satisfy the invariant ...
            if max(asked[cid]) > max(b.required for b in candidate.branches):
                continue
            # ... and, where labels exist, it must not make that clause worse
            g = train_rows.get(cid)
            if g is not None and score(g, {cid: candidate}, committees, semantics=cfg.compile.semantics)["accuracy"] < \
                    score(g, {cid: old}, committees, semantics=cfg.compile.semantics)["accuracy"]:
                continue
            rules[cid] = candidate
            kept += 1
        log.info("consistency round %d: kept %d/%d repairs", rnd + 1, kept, len(bad))


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

    asked = asked_sizes_per_clause(list(frames.values()))
    if cfg.compile.consistency_rounds > 0:
        consistency_repair(cfg, client, rules, clauses, asked, committees, patterns,
                           frames.get("train"), cfg.compile.consistency_rounds)

    if "train" in frames:
        stats = score(frames["train"], rules, committees, semantics=cfg.compile.semantics)
        log.info("before repair: train acc %.4f, %d/%d clauses solved",
                 stats["accuracy"], stats["solved_clauses"], stats["total_clauses"])
        if cfg.compile.self_repair_rounds > 0:
            repair(cfg, client, rules, frames["train"], clauses, committees, patterns,
                   cfg.compile.self_repair_rounds)

    save_rules(rules, cfg.resolve(cfg.compile.out))
    return rules
