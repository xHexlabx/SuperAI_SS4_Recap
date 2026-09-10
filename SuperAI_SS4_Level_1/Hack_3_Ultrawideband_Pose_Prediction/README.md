![uwb_pose_prediction](../../assets/uwb_pose_prediction.webp)

# 🔊 UWB Pose Prediction

<p>
 <b>SuperAI Season 4 | Level 1 | Hackathon 3</b> &nbsp;·&nbsp; <i>Python-first rebuild (2026)</i>
</p>

<p>
 ใน Hackathon นี้ จะเป็นการทำ <b>Signal Classification</b> จากสัญญาณ <b>Ultra-Wide-Band</b> โดยจะให้ทำการ Classified ท่า Pose ต่าง ๆ จากตัวสัญญาณ<br>
 <b>ข้อสังเกตของรอบนี้</b> คือ โจทย์ที่ให้มาเป็น <b>array ของสัญญาณ</b> ก็จริง แต่เราสามารถ <u>แปลงสัญญาณให้เป็นรูปภาพ</u> แล้วแก้เป็นโจทย์<br>
 <b>Image Classification</b> แทนได้ ซึ่งทำให้เรายืม backbone ที่ pretrain มาจาก ImageNet มาใช้ได้เต็ม ๆ
</p>

> **Task** : Signal Classification (signal → image) &nbsp;|&nbsp; **Tools** : timm · PyTorch · SciPy · PyWavelets

<hr>

## 🔁 รอบนี้ต่างจากเดิมยังไง

| | รอบเดิม | รอบนี้ |
|---|---|---|
| รูปแบบโค้ด | Notebook อย่างเดียว | **Python package** (`src/uwb_pose/`) + CLI, notebook ไว้ทำ EDA เท่านั้น |
| Compute | เครื่องของค่าย | **เครื่องตัวเอง** — RTX 3080 Ti Mobile 16 GB VRAM |
| ข้อมูล | แจกมาให้ | โหลดเองจาก **Kaggle** (`uwb-pose-prediction`) |
| Reproduce | ต้องมีข้อมูลอยู่แล้ว | `make setup && make data && make train` จบ |
| Git | ข้อมูล/โมเดลหลุดเข้า repo ได้ | `datasets/` `models/` `submissions/` `.venv/` ถูก **gitignore** ทั้งหมด |

<hr>

## 🧠 แนวคิด : Signal → Image

<p>
 หนึ่งตัวอย่างคือ IQ เชิงซ้อนขนาด <b>2560 × 56</b> จากเรดาร์ Novelda X4M03<br>
 — 2560 คือเฟรมตามเวลาจริง (slow time) , 56 คือช่องระยะ (range bin)<br>
 มันเป็น <b>เมทริกซ์ 2 มิติอยู่แล้ว</b> จึงเรนเดอร์เป็นภาพแล้ว fine-tune backbone จาก ImageNet ได้เลย
</p>

| encoder | ทำอะไร | เก็บแกนเวลา | เก็บแกนระยะ |
|---|---|:-:|:-:|
| `spectrogram2d` | STFT **ทีละ range bin** แล้วรวม magnitude ทีหลัง | ✓ | ✗ (รวม) |
| `micro_doppler` | รวม complex ข้าม range **ก่อน** แล้ว STFT | ✓ | ✗ (รวม) |
| `rti` | Range-Time Intensity — คนอยู่ตรงไหนเมื่อไหร่ | ✓ | ✓ |
| `range_doppler` | FFT ทั้งคลิป — ที่ระยะไหนเคลื่อนที่เร็วเท่าไร | ✗ | ✓ |
| `rd_t0/t1/t2` | range_doppler ราย 3 ช่วงเวลา ใส่คนละ channel | บางส่วน | ✓ |
| `fft2` | FFT 2 มิติทั้งก้อน | ✗ | ✗ |
| `spectrogram / cwt / gaf / recurrence` | encoder ทั่วไปสำหรับสัญญาณ 1 มิติ | — | — |

<hr>

## 📊 ผลการทดลอง (5-fold CV)

<p>
 ทดลองกว่า 45 แบบด้วย <code>scripts/ablate.py</code> ทุกตัวใช้ fold split เดียวกัน
</p>

### วิธีแปลงสัญญาณเป็นภาพ (schedule สั้น 25 epochs)

| encoder | acc | macro-F1 |
|---|---|---|
| **`spectrogram2d`** | **0.8671 ± 0.017** | 0.8508 |
| `micro_doppler` | 0.8099 ± 0.014 | 0.7891 |
| `rti` | 0.7835 ± 0.030 | 0.7579 |
| `stack` (3 มุมมองรวมกัน) | 0.7820 ± 0.012 | 0.7663 |
| `rd_time` | 0.7403 ± 0.027 | 0.7043 |
| `spectrogram_1d` (ฐานเปรียบเทียบ 1 มิติ) | 0.7372 ± 0.009 | 0.7085 |
| `range_doppler` | 0.7002 ± 0.023 | 0.6781 |
| `fft2` | 0.6460 ± 0.013 | 0.6223 |

### หลังจูน encoding + schedule + backbone

| config | acc | macro-F1 |
|---|---|---|
| **maxvit + mixup 0.8, 100 epochs** | **0.9274 ± 0.008** | 0.9174 |
| maxvit + mixup 0.6, 100 epochs | 0.9258 ± 0.013 | 0.9209 |
| convnext + mixup 0.4/0.6, 60 epochs | 0.9243 ± 0.013 | 0.9125 |
| range-band ×3 + convnext + mixup 0.6 | 0.9243 ± 0.011 | 0.9190 |
| convnext_base | 0.9212 ± 0.006 | 0.9107 |
| maxvit + mixup 0.4, 60 epochs | 0.9196 ± 0.008 | 0.9102 |
| convnext_small | 0.9150 ± 0.000 | 0.9016 |
| **ensemble 3 runs × 5 folds** | **0.9382** | — |

### สิ่งที่เรียนรู้

<p>
 <b>1. ยิ่งเก็บโครงสร้างไว้มาก ยิ่งดี</b> — ลำดับของ encoder คือลำดับของ "ยุบแกนไหนทิ้งบ้าง"<br>
 <code>fft2</code> ยุบทั้งเวลาและระยะ · <code>range_doppler</code> ยุบเวลา · <code>spectrogram2d</code> เก็บเวลาไว้<br>
 ช่องว่างบนสุดถึงล่างสุดคือ 22 จุด ซึ่งใหญ่กว่าผลของการเลือกโมเดลมาก
</p>

<p>
 <b>2. ลำดับการรวมสำคัญที่สุด</b> — <code>micro_doppler</code> รวม complex ข้าม range ก่อน STFT
 ทำให้เกิดการหักล้างเชิงเฟส <b>เสียแอมพลิจูดไป 44%</b> เพราะร่างกายคนกินหลาย range bin<br>
 <code>spectrogram2d</code> ถอด magnitude ก่อนแล้วค่อยรวม จึงบวกกันอย่างเดียว → +5.7 จุด
</p>

<p>
 <b>3. การ "รวม" แกน range เป็นข้อดี ไม่ใช่ข้อเสีย</b> — เราลองเก็บแกน range ไว้โดยแบ่งเป็น
 3 ช่วงแล้วใส่คนละ RGB channel (<code>sgband0/1/2</code>) ผลออกมา <b>เสมอ</b> ไม่ดีขึ้น<br>
 เหตุผลน่าจะเป็นว่า <b>ตำแหน่งที่คนยืนเป็นตัวแปรรบกวน ไม่ใช่สัญญาณ</b> — คนวิ่งที่ระยะ 2 เมตร
 กับ 4 เมตรก็คือวิ่งเหมือนกัน การรวมข้าม range จึงทำให้ encoding ไม่ขึ้นกับตำแหน่ง
</p>

<p>
 <b>4. รวมหลาย encoder เป็น RGB ทำให้แย่ลงทุกครั้ง</b> — ด้วยข้อมูล 647 ตัว
 การเพิ่มมุมมองที่อ่อนกว่าคือการเพิ่มที่ให้ overfit<br>
 (แต่การ <b>เฉลี่ยคำทำนาย</b> ของโมเดลที่เทรนแยกกันเป็นคนละเรื่อง — ดู <code>scripts/ensemble.py</code>)
</p>

<p>
 <b>5. <code>dynamic_range: 40</code> ที่ยืมมาจาก baseline คือค่าที่ทำร้ายผลลัพธ์</b> —
 สัญญาณจริงกินช่วง 41-60 dB ใต้ peak การ clip ที่ 40 dB จึงตัดข้อมูลจริงทิ้งใน 99% ของตัวอย่าง<br>
 ค่า 60 dB ขึ้นไปไม่ตัดอะไรแล้ว (80 / 100 / 140 dB ให้ผลตรงกันทุกตำแหน่งทศนิยม)
</p>

<p>
 <b>6. ความแรงของ mixup กับความยาวการเทรนต้องจับคู่กัน</b> — นี่คือผลที่สวนสัญชาตญาณที่สุด
</p>

| mixup α | เทรนสั้น (25ep, patience 5) | เทรนยาว (60ep, patience 20) |
|---:|---|---|
| ไม่ใช้ | 0.8933 | 0.9103 |
| 0.2 | 0.9010 | 0.9088 |
| 0.4 | **0.8779** ← แย่สุด | **0.9196** ← ดีสุด |

<p>
 <b>α = 0.4 เป็นค่าที่แย่ที่สุดตอนเทรนสั้น และดีที่สุดตอนเทรนยาว</b> — การจูนทีละพารามิเตอร์
 จึงหลอกได้ง่าย ถ้าดูแค่คอลัมน์ซ้ายจะสรุปว่า "α สูงไม่ดี" แล้วพลาดค่าที่ดีที่สุดไปเลย<br>
 เหตุผล: mixup ทำให้โจทย์ยากขึ้นโดยตั้งใจ โมเดลจึงต้องการ epoch มากขึ้นกว่าจะบรรจบ
</p>

<p>
 <b>7. early stopping patience คือพารามิเตอร์ที่ถูกมองข้ามมากที่สุด</b><br>
 ที่ epoch cap เท่ากัน : patience 5 → 0.8872 , 10 → 0.8964 , <b>20 → 0.9088</b> แล้วอิ่มตัว<br>
 ค่า default ที่ 5 กดผลของ <i>ทุกการทดลอง</i> ก่อนหน้านั้นไว้ราว 2-3 จุดพร้อมกันทั้งกระดาน
</p>

<p>
 <b>8. ขนาดโมเดลที่พอดีชนะขนาดใหญ่</b> — convnext_tiny (28M) 0.9243 ·
 convnext_small (50M) 0.9150 · convnext_base (89M) 0.9212<br>
 และ CNN ล้วนชนะ hybrid attention : convnext_tiny > maxvit_tiny > swin_tiny ทั้ง accuracy และความนิ่ง
</p>

<hr>

## ⚠️ เรื่องวิธีวัดผล

<p>
 ตอนเริ่มต้นเราวัดด้วย train/val split ครั้งเดียว ผลบอกว่า <b>ปิด clutter removal ดีกว่า</b> (0.839 vs 0.782)<br>
 พอเปลี่ยนเป็น 5-fold CV ตัวเลขเดียวกันเหลือ 0.794 ± 0.039 คือ <b>จมอยู่ใน noise</b><br>
 และเมื่อวัดบน encoder ที่ดีที่สุดจริง ๆ คำตอบกลับด้าน — ปิด clutter removal ทำให้ตกจาก 0.867 เหลือ 0.813
</p>

<p>
 <b>ด้วย validation แค่ ~130 ตัว ความคลาดเคลื่อนของ split เดียวอยู่ที่ราว 3 จุด</b><br>
 ตัวเลขที่ต่างกันน้อยกว่า std ไม่ควรถือว่าต่างกันจริง — ทุกตัวเลขในหน้านี้จึงเป็น 5-fold ทั้งหมด
</p>

### แล้ว test set ก็เล็กยิ่งกว่า

<p>
 test set มี 164 ตัว แบ่งเป็น public 82 / private 82 ดังนั้น <b>1 ตัวอย่าง = 1.22 คะแนน</b>
</p>

| submission | OOF | public | private |
|---|---|---|---|
| ensemble 3 runs | 0.9088 | 0.890 | 0.841 |
| maxvit mixup0.4 | 0.9196 | 0.878 | **0.902** |
| ensemble 4 runs | 0.9335 | 0.890 | **0.902** |
| ensemble 3 runs | 0.9382 | 0.890 | **0.902** |
| ensemble 7 runs | 0.9397 | 0.902 | **0.902** |

<p>
 <b>OOF ไต่ขึ้น 2 จุดผ่านโมเดล 5 แบบที่ต่างกันจริง แต่ private score ไม่ขยับเลย</b><br>
 การพัฒนาที่วัดได้จริงเกิดขึ้นในช่วงแรก (เปลี่ยน encoder, ขยาย schedule) ส่วนการปรับแต่งช่วงหลัง
 อยู่ต่ำกว่าความละเอียดที่ test set ขนาดนี้จะแยกออกได้
</p>

<p>
 <b>บทเรียน:</b> เมื่อ search space ใหญ่พอ เราจะหา combination ที่ "ชนะ" บน validation ได้เสมอ
 ไม่ว่าจะมี signal จริงหรือไม่ — ควรหยุดเมื่อส่วนต่างเล็กกว่า noise ของชุดที่ใช้วัด
</p>

<hr>

## 🚀 วิธีใช้งาน

**ต้องมีก่อน** : [`uv`](https://docs.astral.sh/uv/) และ Kaggle account

```bash
# 0) ติดตั้ง Kaggle CLI + login (ทำครั้งเดียวต่อเครื่อง)
uv tool install kaggle
kaggle auth login

# 1) สร้าง environment (.venv ถูก gitignore ไว้)
make setup

# 2) เช็คว่าทุกอย่างพร้อม โดยไม่ต้องโหลดข้อมูลจริง (ใช้เวลาไม่ถึงนาที)
uv run python scripts/smoke_test.py

# 3) โหลดข้อมูลจาก Kaggle ลง datasets/
#    ⚠️ ต้องกด "I Understand and Accept" ที่หน้า rules ของ competition ก่อน
#    https://www.kaggle.com/competitions/uwb-pose-prediction/rules
make data

# 4) เทรนชุดที่ดีที่สุด (5-fold)
uv run uwb-train --set train.folds=5

# 5) submission แบบ ensemble เฉลี่ยทั้ง 5 folds
uv run uwb-predict --set predict.checkpoint=models/<run_name>
```

<p>
 สคริปต์วิเคราะห์เพิ่มเติม :
</p>

```bash
uv run python scripts/ablate.py            # ทดลองเทียบ encoder ทั้งหมด
uv run python scripts/analyze.py models/<run>   # confusion matrix แบบ out-of-fold
uv run python scripts/ensemble.py models/a models/b --submit submissions/ens.csv
uv run python scripts/make_figure.py       # เรนเดอร์ภาพ encoder ทุกคลาส
```

ปรับค่าอะไรก็ได้จาก command line โดยไม่ต้องแก้ไฟล์ :

```bash
make train SET='--set train.epochs=40 --set model.name=maxvit_base_tf_384.in1k --set encode.img_size=384 --set train.batch_size=8'
make train SET='--set encode.method=cwt'
```

<hr>

## 🖥️ ข้อควรรู้เรื่อง VRAM (16 GB)

<p>
 backbone ตระกูล MaxViT / ViT <b>ล็อกขนาด input</b> ไว้ ถ้าเปลี่ยนรุ่นต้องเปลี่ยน <code>encode.img_size</code> ตามด้วย<br>
 (โค้ดจะเตือนให้เองถ้าตั้งไม่ตรง) ค่าที่พอดีกับเครื่องนี้เมื่อเปิด AMP :
</p>

| `encode.img_size` | `train.batch_size` ที่แนะนำ |
|:-:|:-:|
| 224 | 32 |
| 384 | 8 |
| 512 | 4 |

<p>
 ถ้าอยากได้ batch ใหญ่กว่านั้น ให้ใช้ <code>train.grad_accum</code> แทนการลดขนาดโมเดล<br>
 เช่น <code>--set train.batch_size=8 --set train.grad_accum=4</code> จะได้ผลเท่ากับ batch 32
</p>

<hr>

## 📁 โครงสร้างโฟลเดอร์

```
Hack_3_Ultrawideband_Pose_Prediction/
├── configs/default.yaml        # ค่าทั้งหมดอยู่ที่นี่ที่เดียว
├── src/uwb_pose/
│   ├── config.py               # YAML -> dataclass + override จาก CLI
│   ├── data.py                 # โหลดจาก Kaggle, อ่าน CSV, หา column ของสัญญาณ
│   ├── signal2image.py         # ⭐ หัวใจของ hack: encoder ทั้ง 4 แบบ
│   ├── dataset.py              # Dataset + cache ภาพลงดิสก์ + augmentation
│   ├── model.py                # timm backbone + save/load checkpoint
│   ├── engine.py               # train/eval loop, AMP, cosine schedule, early stop
│   ├── train.py / predict.py   # entry point ของแต่ละขั้น
│   └── cli.py                  # uwb-download / uwb-train / uwb-predict
├── scripts/smoke_test.py       # รัน pipeline ครบวงจรด้วยข้อมูลสังเคราะห์
├── notebooks/                  # EDA เท่านั้น — logic ไม่อยู่ในนี้
├── datasets/ models/ submissions/   # gitignored ทั้งหมด
└── Makefile
```

<hr>

## ⚡ ทำไมต้อง cache

<p>
 การ encode สัญญาณ (โดยเฉพาะ CWT) แพงกว่า forward pass เยอะ ถ้า encode ใหม่ทุก epoch จะกลายเป็นว่า GPU นั่งรอ CPU<br>
 รอบแรกจึงเขียนผลลง <b>float16 memmap</b> ไว้ที่ <code>datasets/cache/</code> — epoch ถัด ๆ ไปและ run ถัด ๆ ไป<br>
 ที่ใช้ค่า encode เดิม จะอ่านจาก cache ตรง ๆ (ชื่อไฟล์ hash มาจาก config เลยไม่ชนกัน)
</p>

<p>
 ล้าง cache ด้วย <code>make clean-cache</code>
</p>

<hr>

<p align="center">
 <a href="../../README.md">⬅️ กลับไปหน้าหลัก</a>
</p>
