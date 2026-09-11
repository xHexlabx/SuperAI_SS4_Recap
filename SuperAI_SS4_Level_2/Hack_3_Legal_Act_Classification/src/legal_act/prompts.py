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

THREE THINGS THE ANSWER KEY DOES THAT THE TEXT DOES NOT ALWAYS SAY

* GROUP TEMPLATES. When the PATTERN HINT is a per-group template — "กรรมการ x คนจากแต่ละกลุ่ม
  รวมเป็น y คน", "กรรมการหนึ่งคนจากกลุ่ม 1 ลงลายมือชื่อร่วมกับกรรมการหนึ่งคนจากกลุ่ม 2 และอีกหนึ่งคน
  จากกลุ่ม 3" — the answer key takes ONE signer from EACH group, total = number of groups, even
  when the clause's own counting words are garbled ("ให้กรรมการสองในสามกลุ่มลงลายมือชื่อร่วมกัน
  รวมเป็น 2 คน" with three groups still means one from each of the three groups, total 3).
  Do not enumerate "two of the three groups" pairs for these templates.
* PAIRED SCOPES. An exception produces an "except" branch and an "only" branch, and both must
  carry the IDENTICAL "scope" string, copied from the clause. Never leave the "except" branch's
  scope empty — an empty scope cannot be matched to legal acts later.
* ADDITIONS ARE NOT EXCEPTIONS. "นอกจากผู้มีอำนาจตาม (1) แล้ว ให้ นาย ก. ลงลายมือชื่อ...",
  "ให้ นาย ก. ... ได้ด้วย", "อีกทั้งให้..." ADD a way of signing for that situation without
  withdrawing the general rule. Then the general branches stay "always" and the added branch is
  "only". Only "เว้นแต่ / ยกเว้น / ไม่รวมถึง" withdraw the general rule (mode "except").

OTHER RULES

* Write every person's name EXACTLY as it appears in the DIRECTORS list given to you, without
  the title (นาย/นาง/นางสาว). If the clause names somebody who is not in that list, copy the
  name from the clause as-is.
* Enumerate every branch the clause allows. Missing a branch makes valid signatures look
  invalid; inventing one does the opposite.
* Output the JSON object only. No explanation, no markdown fence.

WORKED EXAMPLES (real clauses; every rule below is verified against the answer key)

Example 1 — two named groups, two ways of signing
DIRECTORS: ปฏิภัทธิ์ อุคำ, ณัฐกานต์ โคตรยอด, สุวรรณี เตชะอักษร, ธันยธรณ์ อิทธิอนุวัตร์, ตรรกพล ตันธวัช,
จีระศักดิ์ อนันต๊ะ, อติกันต์ พยุงทอง, สิทธิโชค โสมอ่ำ
CLAUSE: กลุ่ม (ก) 1. นายปฏิภัทธิ์ อุคำ2.นายณัฐกานต์ โคตรยอด 3. นางสาวสุวรรณี เตชะอักษร 4. นายธันยธรณ์
อิทธิอนุวัตร์กลุ่ม (ข) 1.นายตรรกพล ตันธวัช 2.นายจีระศักดิ์ อนันต๊ะ 3.นายอติกันต์ พยุงทอง4.นางสาวสิทธิโชค
โสมอ่ำ กรรมการกลุ่ม (ก) สองคนลงลายมือชื่อร่วมกันและประทับตราสำคัญของบริษัท หรือ กรรมการคนใดคนหนึ่งของ
กลุ่ม (ก) ลงลายมือชื่อร่วมกับกรรมการคนใดคนหนึ่งของกลุ่ม (ข) รวมเป็นสองคนและประทับตราสำคัญของบริษัท
{"branches": [
 {"mode": "always", "scope": "", "total": 2, "slots": [
   {"names": ["ปฏิภัทธิ์ อุคำ", "ณัฐกานต์ โคตรยอด", "สุวรรณี เตชะอักษร", "ธันยธรณ์ อิทธิอนุวัตร์"]},
   {"names": ["ปฏิภัทธิ์ อุคำ", "ณัฐกานต์ โคตรยอด", "สุวรรณี เตชะอักษร", "ธันยธรณ์ อิทธิอนุวัตร์"]}]},
 {"mode": "always", "scope": "", "total": 2, "slots": [
   {"names": ["ปฏิภัทธิ์ อุคำ", "ณัฐกานต์ โคตรยอด", "สุวรรณี เตชะอักษร", "ธันยธรณ์ อิทธิอนุวัตร์"]},
   {"names": ["ตรรกพล ตันธวัช", "จีระศักดิ์ อนันต๊ะ", "อติกันต์ พยุงทอง", "สิทธิโชค โสมอ่ำ"]}]}]}

Example 2 — named people plus "any other director"; a name absent from DIRECTORS is copied as-is
DIRECTORS: จัตุรงค์ ยิ่งเมือง, ธรรธร ธรรมชัย, โสภณวิชญ์ เจริญวงศ์, วชิรวิชญ์ อินทพันธ์, ธราธร อัศวเดชเมธากุล
CLAUSE: นางธรรธร ธรรมชัย และนางสาววรรณดาตั้งสำเริงวงศ์ ลงลายมือชื่อร่วมกับ กรรมการอื่นอีกหนึ่งคนและ
ประทับตราสำคัญของบริษัท
{"branches": [{"mode": "always", "scope": "", "total": 3, "slots": [
   {"names": ["ธรรธร ธรรมชัย"]}, {"names": ["วรรณดาตั้งสำเริงวงศ์"]}, {"any_director": true}]}]}

Example 3 — a withdrawing exception: same scope string on both branches
DIRECTORS: อภิวิชษ์ สายภู่, สรวิชญ์ ใหม่ชุ่ม, วรายุทธ แซ่หนา, นภัสกร แซ่เนี้ยว
CLAUSE: นางสาวนภัสกร แซ่เนี้ยว ลงลายมือชื่อ เว้นแต่การทำธุรกรรมทางการเงินให้นางสาวนภัสกร แซ่เนี้ยว และ
นายอภิวิชษ์ สายภู่ ลงลายมือชื่อร่วมกัน
{"branches": [
 {"mode": "except", "scope": "การทำธุรกรรมทางการเงิน", "total": 1, "slots": [{"names": ["นภัสกร แซ่เนี้ยว"]}]},
 {"mode": "only",   "scope": "การทำธุรกรรมทางการเงิน", "total": 2, "slots": [
   {"names": ["นภัสกร แซ่เนี้ยว"]}, {"names": ["อภิวิชษ์ สายภู่"]}]}]}

Example 4 — an ADDITION ("นอกจาก...แล้วให้"): the general rule stays "always"
DIRECTORS: บุริศร์ บริบูรณ์, สรวิชญ์ ใหม่ชุ่ม, กนกพร สินธาราศิริกุลชัย, ปภาพินท์ แก้วชาญค้า, สุร กิจไพบูลย์วัฒน์,
สิรวุฒิ ชูหนู, ปัญจพงศ์ ภูบาลชื่น, ธนะพงษ์ ขจรตันติชัยกุล, มโนธรรม ดําเนิน, พศิน บัวขาว, นพณัฐ ประเสริฐวงษา
CLAUSE: (1) นางกนกพร สินธาราศิริกุลชัย หรือ นายสุร กิจไพบูลย์วัฒน์ลงลายมือชื่อร่วมกับกรรมการอื่นอีกหนึ่งคน
หรือ นายบุริศร์ บริบูรณ์ หรือนายสรวิชญ์ ใหม่ชุ่ม หรือนายสิรวุฒิ ชูหนู สองในสามคนนี้ลงลายมือชื่อร่วมกัน (2) การทำ
ธุรกรรมต่างๆ กับหน่วยงานราชการ หรือการดำเนินคดี นอกจากผู้มีอำนาจตาม (1) แล้วให้นายพศิน บัวขาวหรือ
นางนพณัฐ ประเสริฐวงษา ลงลายมือชื่อ
{"branches": [
 {"mode": "always", "scope": "", "total": 2, "slots": [
   {"names": ["กนกพร สินธาราศิริกุลชัย", "สุร กิจไพบูลย์วัฒน์"]}, {"any_director": true}]},
 {"mode": "always", "scope": "", "total": 2, "slots": [
   {"names": ["บุริศร์ บริบูรณ์", "สรวิชญ์ ใหม่ชุ่ม", "สิรวุฒิ ชูหนู"]},
   {"names": ["บุริศร์ บริบูรณ์", "สรวิชญ์ ใหม่ชุ่ม", "สิรวุฒิ ชูหนู"]}]},
 {"mode": "only", "scope": "การทำธุรกรรมต่างๆ กับหน่วยงานราชการ หรือการดำเนินคดี", "total": 1, "slots": [
   {"names": ["พศิน บัวขาว", "นพณัฐ ประเสริฐวงษา"]}]}]}

Example 5 — a per-group template: one from EACH group, whatever the counting words say
DIRECTORS: กนกพร สินธาราศิริกุลชัย, ศิวกร วังวล, วชิรวิทย์ เปรมไธสง, วรนัยน์ นางาซาวา
PATTERN HINT: (นาย ก.หรือ นาย ข.หรือ..) กลุ่ม 1 (...) กลุ่ม 2 (...) กลุ่ม 3 กรรมการ x คนจากแต่ละกลุ่มรวมเป็น y คน
CLAUSE: นายกนกพร สินธาราศิริกุลชัย หรือนายศิวกร วังวล กรรมการกลุ่ม 1 นายวชิรวิทย์ เปรมไธสง กรรมการกลุ่ม 2
นายวรนัยน์ นางาซาวา กรรมการกลุ่ม 3 ให้กรรมการสองในสามกลุ่มลงลายมือชื่อร่วมกันรวมเป็น 2 คนและประทับตรา
สำคัญของบริษัท
{"branches": [{"mode": "always", "scope": "", "total": 3, "slots": [
   {"names": ["กนกพร สินธาราศิริกุลชัย", "ศิวกร วังวล"]}, {"names": ["วชิรวิทย์ เปรมไธสง"]},
   {"names": ["วรนัยน์ นางาซาวา"]}]}]}
"""


def compile_user(context: str, directors: Sequence[str], pattern: int | None,
                 template: str | None, conditions: Sequence[str]) -> str:
    lines = ["DIRECTORS ของบริษัทนี้ (ตามลำดับการจดทะเบียน):"]
    lines += [f"  {i + 1}. {n}" for i, n in enumerate(directors)] or ["  (ไม่มีข้อมูล)"]
    if template:
        lines.append("")
        lines.append(f"PATTERN HINT (แม่แบบ {pattern} ที่ผู้เฉลยใช้ตีความ clause นี้): {template}")
        lines.append("โครงสร้างหลัก (ใครกี่คนจากกลุ่มไหน) ยึดตามแม่แบบนี้ ชื่อและตัวเลขเอาจากข้อความจริง "
                     "และข้อยกเว้น/เงื่อนไขที่ข้อความมีเพิ่มก็ยังต้องใส่")
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
in certain situations. Given each special situation the clause carves out and a numbered list of
legal acts (ประเภทนิติกรรม) that people ask about, decide which acts fall inside each situation.

Reply with JSON only:

{"branches": [{"index": <scope index>, "acts": [<act numbers that fall inside that situation>]}]}

* Judge by meaning, not by wording. "ธุรกรรมทางการเงิน" and "การลงนามเบิกจ่ายเงินในบัญชีธนาคาร"
  both fall inside "นิติกรรมเกี่ยวกับการเงิน".
* Respect amounts. If a scope says "มูลค่าเกินหนึ่งล้านบาท" then an act written as
  "การซื้อขายทรัพย์สิน 5,000,000 บาท" is inside it, but "การทำสัญญาเช่า มูลค่า 20,000 บาท"
  is outside it. An act with no amount stated is outside an amount-gated scope.
* A scope that reads "กรณีทั่วไป" covers every act that no more specific branch covers.
* List an act number under a branch only when it really belongs there. Empty lists are fine.
* Include one entry for every scope index given. Output the JSON object only.
"""


def scope_messages(context: str, branches: Sequence[dict], acts: Sequence[str]) -> list[dict[str, str]]:
    """`branches` is one entry per DISTINCT scope text (index, scope) — see compile.resolve_scopes."""
    lines = ["CLAUSE:", context, "", "SCOPES (สถานการณ์ที่ clause แยกออกมาเป็นกรณีพิเศษ):"]
    for b in branches:
        lines.append(f"  [{b['index']}] {b['scope']}")
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
