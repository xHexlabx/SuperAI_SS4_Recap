"""Prompt construction. Kept apart from the driver so a prompt tweak is one readable diff."""

from __future__ import annotations

import json
from typing import Sequence

# Grammars used to *retry* a reply that came back malformed. They are not sent on the first
# attempt: constrained decoding costs quality on a task like this, and Qwen3-14B gets the shape
# right 95% of the time on its own — so the grammar is the safety net, not the default.
RULE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["branches"],
    "properties": {
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["mode", "total", "slots"],
                "properties": {
                    "scope": {"type": "string"},
                    "mode": {"type": "string", "enum": ["always", "only", "except"]},
                    "total": {"type": "integer"},
                    "slots": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "names": {"type": "array", "items": {"type": "string"}},
                                "any_director": {"type": "boolean"},
                            },
                        },
                    },
                },
            },
        }
    },
}

SCOPE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["branches"],
    "properties": {
        "branches": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["index", "acts"],
                "properties": {
                    "index": {"type": "integer"},
                    "acts": {"type": "array", "items": {"type": "integer"}},
                },
            },
        }
    },
}

COMPILE_SYSTEM = """\
You are a Thai corporate-law analyst. You convert a company's signing-authority clause
(ข้อความอำนาจกรรมการ) into a formal machine-checkable rule. You never answer questions about
specific people — you only translate the clause. You reply with JSON and nothing else.

THE RULE FORMAT

{"branches": [ {"scope": "...", "mode": "always|only|except", "total": <int>,
                "slots": [ {"names": ["..."]} | {"any_director": true} ]} ]}

* A BRANCH is one complete, valid way of signing. Branches are joined by OR: a signature set
  is valid if it satisfies at least one branch that is in scope.
* A SLOT is exactly ONE signature. {"names": [...]} means any one person from that shortlist
  may fill it. {"any_director": true} means any director of the company may fill it.
* "total" is the EXACT number of signatures the branch requires. There must be exactly
  `total` slots. Signing with more or fewer people than `total` is invalid.
* The same person can never fill two slots — the checker enforces that — so
  "ลงลายมือชื่อร่วมกับกรรมการอื่นอีกหนึ่งคน" is simply one more {"any_director": true} slot.
* "mode" says when the branch is usable:
    - "always": the branch is the general rule, usable for any legal act.
    - "only"  : the branch is usable ONLY for the acts described in "scope"
                (เฉพาะกรณี.../ ในกรณี... / สำหรับ...).
    - "except": the branch is usable for every act EXCEPT those described in "scope"
                (ยกเว้น... / เว้นแต่...).
  Put the Thai wording of the situation in "scope" (short but complete, including any amount
  thresholds such as "มูลค่าเกินหนึ่งล้านบาท"). Leave "scope" empty when mode is "always".

EXCEPTIONS COME IN PAIRS — THIS IS THE PART MOST OFTEN GOT WRONG

"เว้นแต่ / ยกเว้น / เฉพาะกรณี" always creates TWO branches, and the general one stops being
"always". A clause reading

    กรรมการคนใดคนหนึ่งลงลายมือชื่อและประทับตราสำคัญของบริษัท เว้นแต่กรณีการลงลายมือชื่อใน
    เอกสารเปลี่ยนแปลงผู้ถือหุ้น ให้ นาย ก. และ นาย ข. ลงลายมือชื่อร่วมกัน

becomes

    {"branches": [
      {"mode": "except", "scope": "การลงลายมือชื่อในเอกสารเปลี่ยนแปลงผู้ถือหุ้น",
       "total": 1, "slots": [{"any_director": true}]},
      {"mode": "only",   "scope": "การลงลายมือชื่อในเอกสารเปลี่ยนแปลงผู้ถือหุ้น",
       "total": 2, "slots": [{"names": ["ก"]}, {"names": ["ข"]}]}]}

If you leave the general branch on "always", one director alone would wrongly be able to sign
the excepted documents. When a clause labels its cases "(ก) กรณีทั่วไป ... (ข) เฉพาะกรณี ...",
give the general case mode "except" with the scope of case (ข), not mode "always".

HOW TO READ THE COMMON WORDINGS

* "กรรมการสองในห้าคนลงลายมือชื่อร่วมกัน" (no names given)
      -> total 2, slots: [{"any_director": true}, {"any_director": true}]
* "นาย ก. นาย ข. นาย ค. นาย ง. นาย จ. กรรมการสองในห้าคนนี้ลงลายมือชื่อร่วมกัน" (names given)
      -> total 2, slots: [{"names": [ก,ข,ค,ง,จ]}, {"names": [ก,ข,ค,ง,จ]}]
* "นาย ก. หรือ นาย ข. ลงลายมือชื่อร่วมกับ นาย ค. หรือ นาย ง. รวมเป็นสองคน"
      -> total 2, slots: [{"names": [ก,ข]}, {"names": [ค,ง]}]
* "นาย ก. และ นาย ข. ลงลายมือชื่อร่วมกับกรรมการอื่นอีกหนึ่งคน รวมเป็นสามคน"
      -> total 3, slots: [{"names": [ก]}, {"names": [ข]}, {"any_director": true}]
* "กรรมการคนใดคนหนึ่งลงลายมือชื่อ"  -> total 1, slots: [{"any_director": true}]
* "กรรมการทุกคนลงลายมือชื่อร่วมกัน" -> total = number of directors, one {"names": [X]} slot
  per director.
* Groups: "กรรมการกลุ่ม ก หนึ่งคน ลงลายมือชื่อร่วมกับกรรมการกลุ่ม ข หนึ่งคน"
      -> total 2, slots: [{"names": [...group ก...]}, {"names": [...group ข...]}]
* When the clause allows a CHOICE of groups ("สองในสามกลุ่มนี้ กลุ่มละหนึ่งคน"), write out one
  branch per combination — ก+ข, ก+ค, ข+ค.
* Numbered alternatives — "1) ... หรือ 2) ... หรือ 3) ..." or "(ก) ... (ข) ..." — are separate
  branches. Never merge them.
* ตราประทับของบริษัท (the company seal) is not a person: it never becomes a slot and never
  changes "total".

OTHER RULES

* Write every person's name EXACTLY as it appears in the DIRECTORS list given to you, without
  the title (นาย/นาง/นางสาว). If the clause names somebody who is not in that list, copy the
  name from the clause as-is.
* Enumerate every branch the clause allows. Missing a branch makes valid signatures look
  invalid; inventing one does the opposite.
* Output the JSON object only. No explanation, no markdown fence.
"""


def compile_user(context: str, directors: Sequence[str], pattern: int | None,
                 template: str | None, conditions: Sequence[str]) -> str:
    lines = ["DIRECTORS ของบริษัทนี้ (ตามลำดับการจดทะเบียน):"]
    lines += [f"  {i + 1}. {n}" for i, n in enumerate(directors)] or ["  (ไม่มีข้อมูล)"]
    if template:
        lines.append("")
        lines.append(f"PATTERN HINT (แม่แบบ {pattern} ที่ผู้จัดทำโจทย์ติดป้ายไว้): {template}")
        lines.append("ใช้เป็นเบาะแสได้ แต่ถ้าข้อความจริงไม่ตรงกับแม่แบบ ให้ยึดข้อความจริง")
    if conditions:
        lines.append("")
        lines.append("ข้อความเงื่อนไขที่โจทย์ตัดมาให้ (บอกว่า clause นี้มีข้อยกเว้นอยู่จริง):")
        lines += [f"  - {c}" for c in conditions]
    lines.append("")
    lines.append("CLAUSE (ข้อความอำนาจกรรมการ):")
    lines.append(context)
    lines.append("")
    lines.append("Return the JSON rule now.")
    return "\n".join(lines)


def compile_messages(context: str, directors: Sequence[str], pattern: int | None,
                     template: str | None, conditions: Sequence[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": COMPILE_SYSTEM},
        {"role": "user", "content": compile_user(context, directors, pattern, template, conditions)},
    ]


SCOPE_SYSTEM = """\
You are a Thai corporate-law analyst. A signing-authority clause has branches that only apply
in certain situations. Given each branch's situation description and a numbered list of legal
acts (ประเภทนิติกรรม) that people ask about, decide which acts fall inside each description.

Reply with JSON only:

{"branches": [{"index": <branch index>, "acts": [<act numbers that fall inside its scope>]}]}

* Judge by meaning, not by wording. "ธุรกรรมทางการเงิน" and "การลงนามเบิกจ่ายเงินในบัญชีธนาคาร"
  both fall inside "นิติกรรมเกี่ยวกับการเงิน".
* Respect amounts. If a scope says "มูลค่าเกินหนึ่งล้านบาท" then an act written as
  "การซื้อขายทรัพย์สิน 5,000,000 บาท" is inside it, but "การทำสัญญาเช่า มูลค่า 20,000 บาท"
  is outside it. An act with no amount stated is outside an amount-gated scope.
* A scope that reads "กรณีทั่วไป" covers every act that no more specific branch covers.
* List an act number under a branch only when it really belongs there. Empty lists are fine.
* Include one entry for every branch index given. Output the JSON object only.
"""


def scope_messages(context: str, branches: Sequence[dict], acts: Sequence[str]) -> list[dict[str, str]]:
    lines = ["CLAUSE:", context, "", "BRANCHES ที่มีเงื่อนไข:"]
    for b in branches:
        kind = "ใช้ได้เฉพาะกรณี" if b["mode"] == "only" else "ใช้ได้ทุกกรณี ยกเว้น"
        lines.append(f"  [{b['index']}] {kind}: {b['scope']}")
    lines.append("")
    lines.append("LEGAL ACTS ที่ต้องจัดกลุ่ม:")
    lines += [f"  {i + 1}. {a}" for i, a in enumerate(acts)]
    lines.append("")
    lines.append("Return the JSON mapping now.")
    return [
        {"role": "system", "content": SCOPE_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


REPAIR_SYSTEM = COMPILE_SYSTEM + """\

You are now REPAIRING a rule you produced earlier. You will be shown the rule and the labelled
examples it gets wrong. Every example is ground truth. Work out what the rule mis-reads —
a missing branch, a wrong "total", a name in the wrong slot, a scope that is too wide or too
narrow — and return a corrected rule in the same JSON format. Keep everything that already
works. Output the JSON object only.
"""


def repair_messages(context: str, directors: Sequence[str], pattern: int | None,
                    template: str | None, conditions: Sequence[str],
                    rule: dict, failures: Sequence[dict]) -> list[dict[str, str]]:
    lines = [compile_user(context, directors, pattern, template, conditions)]
    lines.append("")
    lines.append("RULE ที่คุณให้มารอบก่อน:")
    lines.append(json.dumps(rule, ensure_ascii=False, indent=1))
    lines.append("")
    lines.append("ตัวอย่างที่กฎนี้ตัดสินผิด (truth คือคำตอบที่ถูก):")
    for f in failures:
        lines.append(
            f"  - ผู้ลงนาม {f['signers']} | นิติกรรม: {f['legal_act']} "
            f"| truth={f['truth']} แต่กฎตอบ {f['pred']}"
        )
    lines.append("")
    lines.append("Return the corrected JSON rule now.")
    return [
        {"role": "system", "content": REPAIR_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


CONSISTENCY_SYSTEM = COMPILE_SYSTEM + """\

You are now FIXING a rule that contradicts the questions being asked about its clause.

The people who wrote the questions never ask for more signatures than the clause can accept:
if a clause is asked about a set of three signers, the clause has some way of being signed by
exactly three people. So when your rule's largest "total" is smaller than the largest set
asked about, your rule has MISSED a way of signing — usually a branch further down the clause,
a group combination you did not expand, or a "รวมเป็น x คน" you read as a smaller number.

Re-read the clause and return a corrected rule that includes a branch for every signature
count listed. Keep the branches that are already right. Output the JSON object only.
"""


def consistency_messages(context: str, directors: Sequence[str], pattern: int | None,
                         template: str | None, conditions: Sequence[str],
                         rule: dict, asked: Sequence[int], totals: Sequence[int]) -> list[dict[str, str]]:
    lines = [compile_user(context, directors, pattern, template, conditions)]
    lines.append("")
    lines.append("RULE ที่คุณให้มารอบก่อน:")
    lines.append(json.dumps(rule, ensure_ascii=False, indent=1))
    lines.append("")
    lines.append(f"คำถามที่ถูกถามกับ clause นี้ มีชุดผู้ลงนามขนาด {list(asked)} คน")
    lines.append(f"แต่กฎของคุณรับได้แค่ {sorted(set(totals))} คน")
    lines.append(f"=> clause นี้ต้องมีทางเซ็นด้วยผู้ลงนาม {max(asked)} คนพอดี อย่างน้อยหนึ่งทาง")
    lines.append("")
    lines.append("Return the corrected JSON rule now.")
    return [
        {"role": "system", "content": CONSISTENCY_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]
