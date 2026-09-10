# ⚖️ Legal Act Classification

<p>
 <b>SuperAI Season 4 | Level 2 | Hackathon 3</b> &nbsp;·&nbsp; <i>Python-first rebuild (2026)</i>
</p>

<p>
 โจทย์คือ <b>อ่านข้อความ "อำนาจกรรมการ" ของบริษัท</b> แล้วตอบว่า ถ้าคนกลุ่มนี้ลงลายมือชื่อ
 เพื่อทำนิติกรรมประเภทนี้ <b>ทำได้หรือไม่ (1/0)</b><br>
 รอบเดิมเป็นงานที่ต้องพึ่ง Domain Expert หนักมาก รอบนี้เราตั้งโจทย์ใหม่ว่า
 <b>จะใช้ LLM ฟรี (Qwen) แทน Domain Expert ได้แค่ไหน</b>
</p>

> **Task** : Rule Reasoning (text → boolean) &nbsp;|&nbsp; **Tools** : Qwen3-14B (vLLM) · bipartite matching · pandas

<hr>

## 🔁 รอบนี้ต่างจากเดิมยังไง

| | รอบเดิม | รอบนี้ |
|---|---|---|
| รูปแบบโค้ด | Notebook (ยังเป็น placeholder) | **Python package** (`src/legal_act/`) + CLI |
| โมเดล | — | **Qwen3-14B-AWQ** รันบนเครื่องตัวเอง (RTX 3080 Ti 16 GB) ผ่าน vLLM |
| บทบาทของ LLM | ตอบ 0/1 ตรง ๆ | **คอมไพล์ข้อความกฎหมายเป็นกฎ JSON** แล้วให้โค้ดตัดสิน |
| ข้อมูล | แจกมาให้ | โหลดเองจาก **Kaggle** (`legal-act-classification`) |
| Reproduce | ต้องมีข้อมูลอยู่แล้ว | `make setup && make data && make serve && make compile && make predict` |
| Git | ข้อมูล/โมเดลหลุดเข้า repo ได้ | `datasets/` `models/` `submissions/` `.venv/` ถูก **gitignore** ทั้งหมด |

<hr>

## 🔍 อ่านข้อมูลก่อนเขียนโค้ด — 3 อย่างที่เปลี่ยนวิธีทำทั้งหมด

### 1. label เป็นฟังก์ชัน deterministic ไม่ใช่ปัญหา ML

<p>
 จับกลุ่ม train ด้วย <code>(context, เซ็ตรายชื่อผู้ลงนาม, legal_act)</code> ได้ <b>4,233 กลุ่ม</b>
 และ <b>ไม่มีกลุ่มไหนที่ label ขัดกันเองเลยแม้แต่กลุ่มเดียว</b><br>
 แปลว่าคำตอบถูกกำหนดโดยกฎล้วน ๆ — ถ้าเราอ่านกฎออก เพดานคือ 1.00 ไม่ใช่ค่าที่ต้องไล่จูน
 และในทางกลับกัน <b>ทุกข้อที่ผิด แปลว่าเราอ่านข้อความผิด</b> ไม่ใช่ "โมเดลยังไม่ converge"
</p>

### 2. `context` ซ้ำกันมหาศาล → LLM ควรถูกเรียกต่อ "ข้อความ" ไม่ใช่ต่อ "แถว"

| | train | test | รวม |
|---|--:|--:|--:|
| จำนวนแถว | 4,429 | 5,835 | 10,264 |
| ข้อความอำนาจกรรมการที่ไม่ซ้ำ | 144 | 247 | **373** |

<p>
 ถ้ายิง prompt ทีละแถวคือ 10,264 ครั้ง แต่ถ้ายิงทีละ clause คือ <b>373 ครั้ง — น้อยกว่า 27 เท่า</b><br>
 และที่สำคัญกว่าเรื่องค่าใช้จ่ายคือ <b>ความคงเส้นคงวา</b> : สองแถวที่อ้างข้อความเดียวกัน
 จะตอบขัดกันเองไม่ได้อีกต่อไป เพราะมันใช้กฎก้อนเดียวกัน
</p>

### 3. ชื่อคนใน `question` เป็นกรรมการจริงเกือบ 100%

<p>
 23,181 จาก 23,183 ชื่อ ตรงกับ <code>committee.csv</code> ของบริษัทนั้นแบบเป๊ะ ๆ หลังตัดช่องว่างทิ้ง
 (อีก 2 ชื่อคือพิมพ์ตก — <code>อธิบดี ค่ยประเสริฐ</code> vs <code>อธิบดี คุ่ยประเสริฐ</code>)<br>
 ข้อนี้ให้ของแถมสำคัญ : ชื่อที่ LLM ถอดออกมาจากข้อความ <b>"snap" กลับไปหาชื่อกรรมการจริงได้เสมอ</b>
 ความผิดพลาดจากการที่โมเดลพิมพ์ชื่อเพี้ยนหรือลืมตัดคำนำหน้าจึงถูกซ่อมทิ้งอัตโนมัติ
</p>

<hr>

## 🧠 แนวคิด : ให้ LLM เป็น "คอมไพเลอร์" ไม่ใช่ "ผู้ตัดสิน"

<p>
 <code>patterns.csv</code> มีแม่แบบข้อความอยู่ 62 แบบ ตั้งแต่ <code>กรรมการ x ใน y คนลงลายมือชื่อ</code>
 ไปจนถึงแบบที่มีกลุ่ม ก/ข/ค ซ้อนกัน แต่ <b>ทุกแบบยุบลงเป็นโครงเดียวกันได้</b> :
</p>

```
branch = รายการ "ช่องลงนาม" (slot) + จำนวนลายเซ็นที่ต้องมีเป๊ะ ๆ
slot   = ลายเซ็น 1 อัน — เป็นรายชื่อที่เลือกได้ 1 คน หรือ "กรรมการคนใดก็ได้"
clause = branch หลายอัน ต่อกันด้วย OR โดยแต่ละอันพ่วง "ขอบเขต" ที่มันใช้ได้
```

<p>
 การตัดสิน 1 แถวจึงกลายเป็น <b>ปัญหา bipartite matching</b> — คนที่เซ็นทั้งหมดต้องจับคู่
 <b>หนึ่งต่อหนึ่ง</b> กับ slot ของ branch ใด branch หนึ่งที่อยู่ในขอบเขต โดยห้ามใครถือสอง slot
</p>

| ข้อความ | แปลเป็น |
|---|---|
| `กรรมการคนใดคนหนึ่งลงลายมือชื่อ` | total 1 · slots `[any]` |
| `กรรมการสองในห้าคนลงลายมือชื่อร่วมกัน` | total 2 · slots `[any, any]` |
| `(ก หรือ ข) ร่วมกับ (ค หรือ ง) รวมเป็นสองคน` | total 2 · slots `[{ก,ข}, {ค,ง}]` |
| `ก และ ข ร่วมกับกรรมการอื่นอีกหนึ่งคน รวมเป็นสามคน` | total 3 · slots `[{ก}, {ข}, any]` |
| `กรรมการทุกคนลงลายมือชื่อ` | total = จำนวนกรรมการ · slots `[any × n]` |

<p>
 <b>ทำไม "จำนวนต้องเป๊ะ" ถึงสำคัญ</b> — <code>รวมเป็นสองคน</code> ไม่ได้แปลว่า "อย่างน้อยสองคน"
 การเซ็นสามคนคือผิด และข้อสอบก็ถามแบบนี้จริง ๆ (ดูตัวอย่าง 20004 ในข้อมูล)<br>
 <b>ทำไมต้องเป็น matching ไม่ใช่แค่นับ</b> — <code>กรรมการอื่นอีกหนึ่งคน</code> แปลว่า "คนที่ยังไม่ได้ใช้ slot"
 การบังคับ matching หนึ่งต่อหนึ่งทำให้เงื่อนไข "ต้องเป็นคนละคน" เกิดขึ้นเองโดยไม่ต้องเขียนกฎเพิ่ม
</p>

### ข้อยกเว้นมาเป็นคู่เสมอ

<p>
 <code>เว้นแต่ / ยกเว้น / เฉพาะกรณี</code> คือจุดที่ยากที่สุดของโจทย์นี้ เพราะมันสร้าง branch สองอันพร้อมกัน
 และทำให้ <b>กฎทั่วไปเลิกเป็นกฎทั่วไป</b> :
</p>

```
กรรมการคนใดคนหนึ่งลงลายมือชื่อ เว้นแต่การเปลี่ยนแปลงผู้ถือหุ้น ให้ ก. และ ข. ลงร่วมกัน

  branch 1  mode=except  scope="การเปลี่ยนแปลงผู้ถือหุ้น"  total 1  [any]
  branch 2  mode=only    scope="การเปลี่ยนแปลงผู้ถือหุ้น"  total 2  [{ก}, {ข}]
```

<p>
 ถ้าปล่อย branch 1 เป็น <code>always</code> กรรมการคนเดียวจะเซ็นเอกสารเปลี่ยนผู้ถือหุ้นได้ ซึ่งผิด<br>
 การตัดสินว่า <code>legal_act</code> ของแถวนี้ตกอยู่ในขอบเขตไหน เป็นงานความหมายล้วน ๆ
 (เช่น <code>การซื้อขายทรัพย์สิน 5,000,000 บาท</code> เข้าเงื่อนไข <code>มูลค่าเกินหนึ่งล้านบาท</code>
 แต่ <code>การทำสัญญาเช่า มูลค่า 20,000 บาท</code> ไม่เข้า) — จึงเป็นงานของ LLM อีกรอบหนึ่ง
</p>

<hr>

## 🔧 Pipeline

```
Stage A  คอมไพล์กฎ      373 clause  →  373 prompt  →  JSON rule ต่อ clause
Stage B  ตีความขอบเขต   เฉพาะ clause ที่มี branch แบบมีเงื่อนไข → จับ legal_act เข้ากลุ่ม
Stage C  ซ่อมด้วย train  clause ไหนที่กฎทำแถว train พัง ส่งกลับให้ LLM พร้อมตัวอย่างที่ผิด
Stage D  ตัดสิน          bipartite matching — ไม่มีโมเดลเกี่ยวข้อง ผลจึง deterministic
```

<p>
 <b>Stage C คือกำไรที่ได้จากข้อสังเกตข้อ 1</b> — ในเมื่อ label เป็นฟังก์ชัน deterministic
 ชุด train ก็คือ <b>test suite ของคอมไพเลอร์</b> ไม่ใช่แค่ข้อมูลเทรน<br>
 clause ที่ทำคะแนนไม่ถึง 1.000 แปลว่ากฎผิดแน่นอน เราจึงส่งข้อความเดิม + กฎเดิม + แถวที่ตัดสินผิด
 กลับเข้าไปให้โมเดลแก้ แล้ว <b>รับกฎใหม่เฉพาะเมื่อมันดีขึ้นจริงบน clause นั้น</b> (ไม่ดีขึ้นก็ทิ้ง)
</p>

<hr>

## 📊 ผลการทดลอง

<p>
 ทุกตัวเลขข้างล่างวัดบน <b>train ทั้ง 4,429 แถว</b> ไม่ใช่ validation split — เพราะ train ไม่ได้ถูกใช้
 ฝึกโมเดล มันถูกใช้เป็น <b>test suite ของคอมไพเลอร์</b> เท่านั้น (LLM เห็นแค่ข้อความ clause กับรายชื่อกรรมการ)
</p>

| ขั้น | train acc | clause ที่แก้ได้ครบ |
|---|---|---|
| ตอบ 0 ทุกข้อ (majority class) | 0.6724 | — |
| คอมไพเลอร์ regex ล้วน ไม่ใช้ LLM | 0.7559 | 53 / 144 |
| Qwen3-14B stage A+B (ยังไม่ซ่อม) | 0.8681 | 76 / 144 |
| ↳ + repair รอบ 1 | 0.8993 | 91 / 144 |
| ↳ + repair รอบ 2 | **0.9043** | **93 / 144** |

<p>
 อ้างอิงจาก public leaderboard ของ competition : baseline ที่ผู้จัดวางไว้ = <b>0.66683</b> ,
 อันดับ 1 = <b>0.90740</b> , ทีม <code>Optimizer-1</code> ของรอบเดิม = <b>0.86754</b><br>
 (ยังไม่ได้ submit ของรอบนี้เข้าไปวัด — ตัวเลข 0.9043 ข้างบนคือ train ไม่ใช่ leaderboard)
</p>

### สิ่งที่เรียนรู้

<p>
 <b>1. การเปลี่ยนหน้าที่ของ LLM ให้ผลมากกว่าการเปลี่ยนโมเดล</b> — ระหว่าง regex (0.7559) กับ
 Qwen ที่ยังไม่ซ่อม (0.8681) ต่างกัน 11 จุด และสิ่งที่เพิ่มเข้ามาไม่ใช่ "ความฉลาด" แต่คือ
 <b>ความสามารถอ่านกลุ่ม ก/ข และข้อยกเว้น</b> ซึ่ง regex ทำไม่ได้เลย ส่วนการนับเลขไทยกับการ
 จับคู่ slot นั้น regex ก็ทำได้อยู่แล้ว — งานส่วนนี้จึงไม่ควรฝากไว้กับโมเดล
</p>

<p>
 <b>2. repair loop คุ้มที่สุดในบรรดาทุกอย่างที่ทำ</b> — +3.6 จุดจาก prompt เดิม โมเดลเดิม
 ไม่มีการจูนอะไรเลย<br>
 มันได้ผลเพราะ label เป็น deterministic : เราชี้ได้ว่า <i>clause ไหน</i> ผิด และส่ง
 <i>ตัวอย่างที่ผิด</i> กลับไปให้โมเดลดู ต่างจากการบอกลอย ๆ ว่า "ลองใหม่อีกที"<br>
 และเพราะเรารับกฎใหม่<b>เฉพาะเมื่อมันดีขึ้นจริงบน clause นั้น</b> การซ่อมจึงเดินหน้าอย่างเดียว
 ไม่มีทางถอยหลัง
</p>

<p>
 <b>3. ความผิดพลาดกระจุกตัว ไม่ได้เกลี่ยทั่ว</b> — clause ที่ยังพลาดหนักที่สุดคือ clause เดียว
 ที่กิน 259 แถว (5.8% ของ train) ทั้งหมด<br>
 การไล่ดูทีละ clause ด้วย <code>scripts/inspect_clause.py</code> จึงคุ้มกว่าการไล่ดูทีละแถวมาก
</p>

<p>
 <b>4. Qwen3-14B ตอบ JSON เพี้ยน 18/373 clause (4.8%)</b> และเพี้ยนแบบเดียวกันหมด คือ
 <b>ปิด array <code>branches</code> เร็วไปแล้วเปิดอันใหม่</b> (<code>}], [{</code>)<br>
 ตอนนี้โค้ดจัดการโดย <b>ถาม LLM ซ้ำโดยบังคับ JSON schema</b> เฉพาะคำตอบที่ parse ไม่ผ่าน —
 ไม่บังคับตั้งแต่แรกเพราะ constrained decoding ทำให้คุณภาพการอ่านกฎแย่ลง ในเมื่อพัง 5%
 ก็ควรจ่ายเฉพาะ 5% นั้น
</p>

<p>
 <b>5. ที่เหลืออีก 13 clause ยังคอมไพล์ไม่ได้</b> — กระทบ 2.1% ของ train และ 3.8% ของ test
 (แถวเหล่านี้ตกไปใช้ค่า fallback = 0)<br>
 <b>หมายเหตุ:</b> ตัวเลขในตารางข้างบนวัดจากรอบที่ <i>ยังไม่มี</i> การ retry ด้วย JSON schema
 การรัน <code>make compile</code> ใหม่วันนี้จึงควรได้เท่าเดิมหรือดีกว่า
</p>

### ต้นทุน

| | |
|---|---|
| LLM call ทั้งหมด | ~700 ครั้ง (373 compile + 69 scope + ~250 repair) |
| ถ้ายิงทีละแถวแทน | 10,264 ครั้ง |
| เวลารันทั้ง pipeline | ~17 นาที บน RTX 3080 Ti 16 GB |
| VRAM | 9.4 GB (weights) + 2.8 GB (KV cache) |
| ค่าใช้จ่าย | 0 บาท |

<hr>

## 🚀 วิธีใช้งาน

**ต้องมีก่อน** : [`uv`](https://docs.astral.sh/uv/) และ Kaggle account

```bash
# 0) ติดตั้ง Kaggle CLI + login (ทำครั้งเดียวต่อเครื่อง)
uv tool install kaggle
kaggle auth login

# 1) สร้าง environment (.venv ถูก gitignore ไว้)
make setup

# 2) เช็คว่า engine ถูกต้อง โดยไม่ต้องมี LLM หรือ GPU (ไม่ถึงนาที)
uv run python scripts/smoke_test.py

# 3) โหลดข้อมูลจาก Kaggle ลง datasets/
#    ⚠️ ต้องกด "I Understand and Accept" ที่หน้า rules ของ competition ก่อน
make data

# 4) เปิด Qwen เป็น OpenAI-compatible server (คนละ terminal, ใช้ VRAM ~10 GB)
make serve

# 5) คอมไพล์กฎ → วัดผลบน train → สร้าง submission
make compile
make eval
make predict
```

### อยากใช้ API ฟรีแทนการรันเอง

<p>
 ทุก backend คุยผ่าน <code>/v1/chat/completions</code> เหมือนกันหมด เปลี่ยนแค่ config :
</p>

```bash
export LLM_API_KEY=sk-or-...
make compile SET="--set llm.base_url=https://openrouter.ai/api/v1 \
                  --set llm.model=qwen/qwen3-235b-a22b:free \
                  --set llm.thinking=default \
                  --set llm.concurrency=2"
```

<p>
 <code>llm.thinking</code> : <code>off</code> = สั่ง Qwen3 ไม่ต้องคิดก่อนตอบ (เร็วกว่าและ parse JSON ง่ายกว่า) ·
 <code>on</code> = ให้คิด · <code>default</code> = ไม่ส่ง flag นี้ สำหรับ API ที่ไม่รู้จัก <code>chat_template_kwargs</code>
</p>

### เครื่องมือสำหรับไล่ดูว่ากฎข้อไหนยังเพี้ยน

```bash
uv run python scripts/inspect_clause.py            # 10 clause ที่แย่ที่สุด + แถวที่ผิด
uv run python scripts/inspect_clause.py 105529030059   # เจาะดูบริษัทเดียว
uv run python scripts/error_analysis.py            # error แยกตาม pattern / จำนวนคนเซ็น
uv run legal-compile --baseline                    # คอมไพเลอร์ regex ล้วน ไม่ใช้ LLM
```

<hr>

## 📁 โครงสร้างโฟลเดอร์

```
Hack_3_Legal_Act_Classification/
├── configs/default.yaml         # ค่าทั้งหมดอยู่ที่นี่ที่เดียว
├── src/legal_act/
│   ├── config.py                # YAML -> dataclass + override จาก CLI
│   ├── data.py                  # โหลดจาก Kaggle, ยุบแถวเป็น clause, index คณะกรรมการ
│   ├── rules.py                 # ⭐ DSL (slot/branch) + bipartite matching
│   ├── prompts.py               # prompt ของ stage A / B / C อยู่รวมกันที่เดียว
│   ├── llm.py                   # OpenAI-compatible client + disk cache + retry
│   ├── compile.py               # stage A/B/C — ตัวขับหลัก
│   ├── baseline.py              # คอมไพเลอร์ regex ไว้เทียบว่า LLM ช่วยได้เท่าไร
│   ├── engine.py                # เอากฎไปตัดสินทุกแถว + ให้คะแนนบน train
│   ├── evaluate.py / predict.py # entry point ของแต่ละขั้น
│   └── cli.py                   # legal-download / compile / eval / predict
├── scripts/smoke_test.py        # ตรวจ engine ด้วยเคสที่เขียนมือ ไม่ต้องมี LLM
├── notebooks/                   # EDA เท่านั้น — logic ไม่อยู่ในนี้
├── datasets/ models/ submissions/   # gitignored ทั้งหมด
└── Makefile
```

<hr>

## ⚡ ทำไมต้อง cache คำตอบของ LLM

<p>
 คำตอบทุกครั้งถูกเก็บลงดิสก์โดย hash จาก <code>(model, temperature, thinking, messages)</code><br>
 แก้ prompt ของ stage B แล้วรัน <code>make compile</code> ใหม่ จึงจ่ายเฉพาะ stage B ส่วน stage A
 อ่านจาก cache ทันที — วนแก้ prompt ได้ถี่ ๆ โดยไม่เสียโควตาและไม่ต้องรอ
</p>

<p>
 ล้าง cache ด้วย <code>make clean-cache</code>
</p>

<hr>

<p align="center">
 <a href="../../README.md">⬅️ กลับไปหน้าหลัก</a>
</p>
