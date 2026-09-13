# 📋 สถานะงาน & สิ่งที่ต้องทำต่อ

> เขียนไว้ตอน **2026-09-13 09:05** เพื่อย้ายไปทำต่อบนเครื่อง remote (เครื่องเดิมดับระหว่างเทรน)

<hr>

## ✅ ทำเสร็จแล้ว

### 1. เข้าใจโจทย์ครบ (ยืนยันด้วยข้อมูลจริง ไม่ใช่เดา)

| สิ่งที่รู้ | ยืนยันยังไง |
|---|---|
| ข้อมูลคือ **Empatica E4** (ชุด DREAMT ของ PhysioNet) **ไม่ใช่ EEG** | 8 คอลัมน์ BVP/ACC_XYZ/TEMP/EDA/HR/IBI ตรงเป๊ะ + อัตราสุ่มจริงตรงกับสเปกเครื่อง |
| train 83 คืน · test 10 คืน · 1 epoch = 1920 แถว = 30 วิ @ 64 Hz | นับจากไฟล์จริง |
| 3 คลาส **N 65.5 % · W 23.8 % · R 10.7 %** | นับจาก label ทั้งหมด |
| metric คือ **macro F1** | submission เก่าที่ทาย N ล้วนได้ 0.2666 = macro-F1 ของ constant predictor พอดี |
| **ไฟล์ test เรียงตามเวลาและต่อกลับเป็นคืนเดียวได้** ← จุดชี้ขาด | TEMP/EDA/HR ต่อกันสนิทระหว่างไฟล์ (ช่องว่าง 0.01 vs 0.27 เมื่อสุ่มลำดับ) |
| **ยัง submit แบบ late ได้** | submit จริงสำเร็จเมื่อ 2026-09-13 |

### 2. Package ใช้งานได้ครบ (`src/sleep_stage/`)

`data` → `features` (96 ตัว/epoch) → `context` (z-score ต่อคืน + rolling/lag) → `models` (lgbm / bilstm / gru / tcn) → `calibrate` → `smooth` (Viterbi) → `predict`
พร้อม `configs/default.yaml`, `Makefile`, `scripts/ablate.py`, `scripts/ensemble.py`

### 3. EDA เสร็จ — `notebooks/01_eda.ipynb` รันผ่าน สร้างรูป 5 รูปลง `assets/`

### 4. ผลที่ได้จริง

| ขั้น | OOF macro-F1 (GroupKFold 5 fold ตามคืน) |
|---|---|
| LightGBM feature ชุดแรก, argmax | 0.5800 |
| &nbsp;&nbsp;+ Viterbi | 0.6167 |
| LightGBM feature v2 (เติม RIIV/RSA การหายใจ), argmax | 0.5994 |
| &nbsp;&nbsp;+ Viterbi | 0.6276 |
| &nbsp;&nbsp;+ calibration ต่อคลาส (bias N −0.53 · R +0.97 · W −0.43) | **0.6448** |

**Kaggle (submit แล้ว, ref 56196621)**

| | public | private |
|---|---|---|
| รอบเดิม FFT + AutoGluon (อันดับ 28/156) | 0.58881 | 0.59865 |
| ที่ 1 ของตารางรอบแข่งจริง | 0.59331 | — |
| **รอบนี้ LGBM v2 + Viterbi + calibration** | **0.66428** | **0.67136** |

ไฟล์ที่ส่งเก็บไว้ที่ `submissions/submission_lgbm_v2.csv`
probability ของ run นี้ archive ไว้ที่ `results/saved_runs/lgbm_v2/` (แปลงเป็น float32 แล้ว, ~1 MB) — **ไม่ต้องเทรน LGBM ใหม่บนเครื่อง remote** แค่ `make restore`

<hr>

## ⏳ ค้างอยู่ — sweep ของ sequence model

`scripts/ablate.py seq` รันไปได้แค่ variant แรก 2 จาก 3 fold แล้วเครื่องดับ **ยังไม่มี `results/ablation_seq.csv`**

สิ่งที่เห็นจาก 2 fold นั้น (bilstm e60, balanced loss, 3 fold, 198 feature)

| fold | raw macro-F1 | หลัง Viterbi | N | R | W |
|---|---|---|---|---|---|
| 0 | **0.6311** | 0.6054 | 0.723 | **0.498** | 0.595 |
| 1 | 0.5973 | 0.5765 | 0.735 | 0.421 | 0.574 |

อ่านได้สองอย่าง
1. **raw ของ BiLSTM ดีกว่า raw ของ LGBM ชัดเจน** (0.6311 vs 0.5877 ที่ fold เดียวกันของ lgbm_v2) และ **R ดีกว่า** → มีของ
2. **Viterbi ทำให้แย่ลง** เพราะ loss ถ่วงน้ำหนักคลาส (`loss_class_weight=balanced`) ทำให้ posterior ไม่ใช่ p(y|x) จริง แล้ว `smooth.prior_power=1` ไปหารด้วย prior อีกรอบ = แก้ซ้ำสองที
   → variant `plain loss` กับ `smooth.prior_power=0` มีไว้วัดเรื่องนี้ และขั้น `calibrate` น่าจะกลบปัญหาได้เอง

<hr>

## 🖥️ ทำไมเครื่องเดิมดับ

เครื่องเดิม: **ไม่มี NVIDIA** (AMD Phoenix3 iGPU เท่านั้น) — 16 core / 11 GB RAM
ตอนดับกำลังรัน LightGBM `data.n_jobs=14` + torch 16 thread ต่อเนื่องกันเกือบ 40 นาที · load average ~18
บนเครื่อง remote ควร
- ตั้ง `data.n_jobs` ไว้ที่ ~จำนวน core ลบ 2 อย่าแตะเพดาน
- ใช้ **GPU** สำหรับ sequence net (เพิ่ง wire ไว้ให้แล้ว ดูข้อ 1 ด้านล่าง)

<hr>

## 🎯 TODO ตามลำดับ

### 0. เตรียมเครื่อง remote

```bash
git pull
cd SuperAI_SS4_Individual_Hackathon/Hack_5_Sleep_Stages_Classification
make setup                 # uv sync (torch 2.14 + cu130 อยู่ใน uv.lock แล้ว)
make data                  # โหลดจาก Kaggle ~700 MB แตกแล้ว ~7 GB (ต้องมี ~/.kaggle/)
make cache                 # สกัด feature ต่อคืน ~30 วินาที บน 14 core
make restore               # คืน results/saved_runs/ กลับเข้า models/ จะได้ไม่ต้องเทรน LGBM ใหม่
```

ตรวจว่า GPU ใช้ได้จริง
```bash
uv run python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 1. ⚠️ smoke-test โค้ด GPU ก่อนอย่างอื่น — **ยังไม่เคยรันสำเร็จเลยสักครั้ง**

ผมเพิ่ง wire `model.device` (auto / cpu / cuda) เข้า `models.py` ไปตอน 08:5x แล้วเครื่องดับตอนกำลังจะเทสต์พอดี
ตรวจแล้วแค่ว่า `resolve_device()` ทำงานถูกและไฟล์ parse ผ่าน — **แต่ path เทรนจริงบน CUDA ยังไม่เคยถูกรัน**

```bash
uv run sleep-train --set model.name=bilstm --set model.epochs=3 --set model.n_seeds=1 \
  --set model.avg_last=1 --set cv.n_folds=2 --set cv.full_fit=false \
  --set smooth.calibrate=false --set train.run_name=smoke_gpu
rm -rf models/smoke_gpu
```
ควรเห็น log `sequence model bilstm on cuda:0 (...)` และจบใน ~1 นาที
จุดที่น่าจะพังถ้าพัง: tensor `weight` ของ loss, mask indexing `logits[mask]`, และ `_snapshot()` ที่ย้ายกลับ CPU

### 2. รัน sweep ต่อ

```bash
uv run python scripts/ablate.py seq --set model.n_seeds=1
```
6 variant (bilstm e60 balanced / e60 plain loss / e60 no-Viterbi / e150 / tcn e150 / gru e150) ที่ 3 fold
บน CPU เครื่องเดิมประมาณ 12.5 นาที/variant · บน 3080 Ti น่าจะเหลือ ~2–4 นาที/variant
ผลเขียนต่อท้าย `results/ablation_seq.csv` และ **ข้าม variant ที่มีในไฟล์แล้ว** → ถ้าดับอีกก็รันคำสั่งเดิมซ้ำได้เลย

ถ้า GPU เร็วจริง รัน suite ที่เหลือด้วย
```bash
uv run python scripts/ablate.py capacity     # hidden/layers/dropout/crop/สถาปัตยกรรม
uv run python scripts/ablate.py features     # ตัด per-subject norm / clock / rolling / lag ทีละอย่าง
```

### 3. เทรนตัวชนะแบบเต็ม แล้ว ensemble

```bash
# แทน <best> ด้วย override ของ variant ที่ชนะใน ablation_seq.csv
uv run sleep-train --set model.name=bilstm --set model.n_seeds=3 \
  --set cv.n_folds=5 --set cv.full_fit=true --set train.run_name=seq_final

uv run python scripts/ensemble.py --write     # เลือกส่วนผสมด้วย OOF macro-F1 แล้วเขียน submission.csv
```
`ensemble.py` จะ greedy เลือกจาก run ทั้งหมดที่มี `oof_probs.npy` + `test_probs_full.npy`
(ตอนนี้มี `lgbm_v2` อยู่แล้วจาก `make restore`) และจูน class bias ใหม่ให้ blend นั้นโดยเฉพาะ

### 4. submit

```bash
make submit MSG="ensemble bilstm + lgbm, Viterbi + macro-F1 calibration, OOF <score>"
```
**โควตา 4 ครั้ง/วัน** (วันนี้ใช้ไป 1) · ผลขึ้นภายใน ~30 วินาที ดูด้วย
```bash
uv run kaggle competitions submissions -c signal-processing-sleep-staging-classification | head -4
```

### 5. ปิดงาน

- เติมผล ensemble ลงตารางใน `README.md` ของโฟลเดอร์ (มีหัวข้อ "📊 ผลลัพธ์" รออยู่แล้ว)
- เติมบรรทัดคะแนนรอบใหม่ใน `README.md` หลักของ repo (ส่วน Sleep Stages Classification)
- commit + push

<hr>

## 💡 ไอเดียที่ยังไม่ได้ลอง (ถ้าเวลาเหลือ)

| ไอเดีย | เหตุผล |
|---|---|
| **hierarchical**: แยก W-vs-sleep ก่อน แล้วค่อย N-vs-R | W แยกด้วย actigraphy ได้ง่ายมาก ส่วน N/R นิ่งเหมือนกันทั้งคู่ ต้องใช้ feature คนละชุด |
| **confidence-based abstention** | งานวิจัยปี 2026 รายงานว่าตัด 20 % ของ epoch ที่มั่นใจน้อยที่สุดทิ้ง ทำให้ κ ขึ้นจาก 0.452 → 0.512 (ใช้ไม่ได้ตรง ๆ กับ Kaggle ที่ต้องตอบทุกแถว แต่ใช้เป็น feature ให้ชั้นบนได้) |
| **pseudo-labeling บน test** | test มี 10 คืนเต็ม ๆ ที่ไม่ได้ใช้เลย |
| feature การหายใจเพิ่ม: ความแปรปรวนของ **แอมพลิจูด** การหายใจ ไม่ใช่แค่ความถี่ | REM มี apnea/hypopnea ถี่กว่า |
| `minutes since last wake bout` | REM density สูงขึ้นตอนใกล้เช้าและหลังผ่าน NREM cycle |
| เพดานที่ควรคาดหวัง | งานวิจัย wearable 3 คลาส อยู่ที่ κ ~0.42–0.52 · ตอนนี้เราได้ **0.467** แล้ว จึงเหลือ headroom ไม่มาก — กำไรน่าจะมาจาก **กฎการตัดสิน** มากกว่าตัวโมเดล |

<hr>

## ⚠️ กับดักที่เจอมาแล้ว อย่าเหยียบซ้ำ

1. **อย่าเชื่อ correlation ระหว่าง HR ที่เราหาเองกับคอลัมน์ HR ของเครื่อง** — ค่ามันต่ำ (0.2–0.6) แต่ MAE ดี (0.6–1 bpm) เพราะ HR ในคืนเดียวแกว่งน้อย correlation เลยถูก noise กลบ ใช้ MAE ตัดสิน
2. **BVP ที่ข้อมือ noise สูง** — หา peak ตรง ๆ บนแบนด์ 0.5–8 Hz จะนับ dicrotic notch เป็นบีตซ้ำ ต้องหาบนแบนด์แคบ 0.5–3.5 Hz ก่อนแล้วค่อย refine (ทำไปแล้วใน `detect_beats`)
3. **แก้ไฟล์ `.py` ระหว่างมี job รันอยู่ ไม่มีผลกับ job นั้น** (module ถูก import ไปแล้ว) — `lgbm_v2` เลยไม่ได้ calibration ในตัว ต้องมาจูนทีหลังจาก OOF ที่เซฟไว้
4. **`CACHE_VERSION` ใน `data.py` = `v2`** ถ้าแก้ `features.py` ต้องขยับเป็น v3 ไม่งั้นมันใช้ cache เก่า
5. **zsh** กิน `[...]` เป็น glob — เวลา override ต้อง quote: `--set 'predict.runs=["lgbm_v2"]'`
