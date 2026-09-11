"""The rule DSL and its evaluator.

Every one of the 62 templates in patterns.csv collapses into the same shape:

    a branch = a list of signature SLOTS + the exact number of signatures required

A slot is either a named shortlist ("นาย ก. หรือ นาย ข.") or "any director of this company".
A clause is a list of branches joined by OR, each carrying the scope it applies to — that is
how "เว้นแต่ / เฉพาะกรณี" exceptions are represented.

Deciding a row is then a bipartite matching problem: the people who signed must be assignable
one-to-one onto the slots of at least one applicable branch, with nobody filling two slots.
Making the count *exact* matters — "รวมเป็นสองคน" is not satisfied by three signatures.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Iterable

from .utils import normalise_name

ALWAYS, ONLY, EXCEPT = "always", "only", "except"


@dataclass(frozen=True)
class Slot:
    """One signature. Empty `names` means any director of the company qualifies."""

    names: tuple[str, ...] = ()
    label: str = ""

    @property
    def any_director(self) -> bool:
        return not self.names

    def keys(self) -> set[str]:
        return {normalise_name(n) for n in self.names}

    def to_dict(self) -> dict[str, Any]:
        if self.any_director:
            return {"any_director": True, "label": self.label}
        return {"names": list(self.names), "label": self.label}

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Slot":
        names = raw.get("names") or []
        if raw.get("any_director") or not names:
            return Slot(names=(), label=str(raw.get("label", "")))
        return Slot(names=tuple(str(n) for n in names), label=str(raw.get("label", "")))


@dataclass
class Branch:
    """One way of validly signing, plus when that way is allowed."""

    slots: list[Slot]
    total: int | None = None
    scope: str = ""
    mode: str = ALWAYS
    acts: list[str] = field(default_factory=list)  # filled in by the scope stage

    @property
    def required(self) -> int:
        return self.total if self.total is not None else len(self.slots)

    def normalised_slots(self) -> list[Slot]:
        """Pad with 'any director' slots when the clause says a bigger total than it spells out."""
        need = self.required
        slots = list(self.slots[:need])
        while len(slots) < need:
            slots.append(Slot())
        return slots

    def applies_to(self, legal_act: str) -> bool:
        """Is this branch usable for this kind of legal act?

        An "only" branch whose scope was never resolved carries no information, so it is left
        unrestricted rather than switched off. Switching it off is a guess that the branch can
        never be used — the most destructive guess available, and on this data it silently
        removed 31 clauses' worth of valid ways to sign.
        """
        if self.mode == ALWAYS or not self.acts:
            return True
        listed = legal_act in self.acts
        return listed if self.mode == ONLY else not listed

    def to_dict(self) -> dict[str, Any]:
        return {
            "slots": [s.to_dict() for s in self.slots],
            "total": self.total,
            "scope": self.scope,
            "mode": self.mode,
            "acts": list(self.acts),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Branch":
        mode = str(raw.get("mode", ALWAYS))
        if mode not in (ALWAYS, ONLY, EXCEPT):
            mode = ALWAYS
        total = raw.get("total")
        return Branch(
            slots=[Slot.from_dict(s) for s in raw.get("slots", [])],
            total=int(total) if isinstance(total, (int, float)) and total else None,
            scope=str(raw.get("scope", "")),
            mode=mode,
            acts=[str(a) for a in raw.get("acts", [])],
        )


@dataclass
class ClauseRule:
    """The compiled form of one company's signing-authority clause."""

    clause_id: str
    rg: str
    branches: list[Branch] = field(default_factory=list)
    note: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.branches) and not self.error

    @property
    def needs_scope(self) -> bool:
        """Only clauses whose branches disagree about scope need the second LLM stage."""
        return any(b.mode != ALWAYS for b in self.branches)

    def to_dict(self) -> dict[str, Any]:
        return {
            "clause_id": self.clause_id,
            "rg": self.rg,
            "branches": [b.to_dict() for b in self.branches],
            "note": self.note,
            "error": self.error,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "ClauseRule":
        return ClauseRule(
            clause_id=str(raw["clause_id"]),
            rg=str(raw["rg"]),
            branches=[Branch.from_dict(b) for b in raw.get("branches", [])],
            note=str(raw.get("note", "")),
            error=str(raw.get("error", "")),
        )


def _match_all(candidates: list[set[int]], n_slots: int) -> bool:
    """Kuhn's algorithm: can every signer be given a distinct slot they qualify for?"""
    slot_of: list[int | None] = [None] * n_slots

    def augment(signer: int, seen: set[int]) -> bool:
        for slot in candidates[signer]:
            if slot in seen:
                continue
            seen.add(slot)
            if slot_of[slot] is None or augment(slot_of[slot], seen):
                slot_of[slot] = signer
                return True
        return False

    return all(augment(i, set()) for i in range(len(candidates)))


def branch_allows(branch: Branch, signers: Iterable[str], directors: set[str]) -> bool:
    """Does this exact set of signatures satisfy this branch?"""
    keys = [normalise_name(s) for s in signers if normalise_name(s)]
    slots = branch.normalised_slots()
    if len(keys) != len(slots):
        return False
    # the same person signing twice is one signature, not two
    if len(set(keys)) != len(keys):
        return False

    slot_keys = [s.keys() for s in slots]
    candidates: list[set[int]] = []
    for k in keys:
        allowed = {
            j for j, s in enumerate(slots)
            if (k in directors if s.any_director else k in slot_keys[j])
        }
        if not allowed:
            return False
        candidates.append(allowed)
    return _match_all(candidates, len(slots))


def decide(rule: ClauseRule, signers: Iterable[str], legal_act: str, directors: set[str],
           semantics: str = "exact") -> bool:
    """A row is allowed when at least one branch in scope accepts the signature set.

    `semantics` decides what an EXTRA signature means:
      exact    — "รวมเป็นสองคน" means two, so a third signer invalidates the set.
      at_least — extra signatures are harmless; some subset of the signers must satisfy a branch.

    Train cannot tell these apart (they disagree on 35 of 4,429 rows, 19-16 in favour of
    at_least) because its questions never ask for more signers than a clause needs. Test asks
    such questions constantly, so the choice matters there and only there.
    """
    signers = list(signers)
    for branch in rule.branches:
        if not branch.applies_to(legal_act):
            continue
        if branch_allows(branch, signers, directors):
            return True
        if semantics == "at_least" and len(signers) > branch.required:
            for subset in itertools.combinations(signers, branch.required):
                if branch_allows(branch, subset, directors):
                    return True
    return False
