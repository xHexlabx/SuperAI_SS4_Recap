# Research memo: Thai Image Captioning (SuperAI SS4 L2 Hack 1) — 2026-09-11

เป้าหมาย: เทรน "โมเดลบรรยายภาพภาษาไทยของตัวเอง" บนเครื่อง local (RTX 3080 Ti Laptop 16 GB) แล้วทำคะแนนให้เกินอันดับ 1 ของ leaderboard (52.20) จากคะแนนเดิมของเรา 45.97 public / 45.43 private (อันดับ 20 จาก 272 ทีม)

## 0. ข้อเท็จจริงของ competition นี้

- Kaggle `image-processing-thai-language-image-captioning` — "Super AI Engineer 2025 (Level 1) Image Processing Hackathon" แข่ง 31 ม.ค. – 2 ก.พ. 2568 (36 ชั่วโมง on-site) 272 ทีม, 4 submissions/วัน, public/private 50/50
- **Metric: หน้า Evaluation เขียนแค่ "Bleu-score"** แต่การแข่งรุ่นก่อน (`thai-language-image-captioning`, SS3 2024, ผู้จัดเดียวกัน) ระบุชัดว่า *sacrebleu BLEU, `references_per_prediction = 3`, ตัดคำด้วย sentencepiece flores101* (`tokenize="flores101"`) — คะแนน 45–52 บน LB สอดคล้องกับ BLEU-4×100 แบบนี้ ให้ใช้ scorer นี้เป็นค่า default แล้วยืนยันด้วยการเทียบ val-vs-LB หลังส่งครั้งแรก
- Leaderboard: #1 Magi 52.20, #2 50.65, #3 49.07, #4–#10 อยู่แถว 47.4–48.7 (น่าจะเป็นกลุ่มที่ใช้ notebook สาธารณะตัวเดียวกัน) — เราอยู่ 45.97 (#20)
- Notebook สาธารณะที่คนโหวตสูงสุด (93 votes): BLIP-2 opt-2.7b-coco + LoRA เทรนบน IPU24 10k รูป (caption แปลไป-กลับด้วย pythainlp en↔th) แล้ว *generate ภาษาอังกฤษแล้วแปลเป็นไทย* — เพดานของวิธีนี้คือคะแนน ~46–48. อีกอันใช้ unsloth + Qwen2-VL-7B 4-bit เทรนตรงเป็นภาษาไทย 30 steps (ไม่พอ)
- Late submission ยังเปิดอยู่: `submission-limits` บอก remaining today = 100 (ส่งได้เรื่อย ๆ ไม่ติดโควตา 4/วันแล้ว)

### ข้อมูล

| ส่วน | รูป | ที่มา | caption |
|---|---:|---|---|
| train JSON | 142,291 | COCO train2017 114,287 + IPU24 train (travel 16,311, food 11,693) | 430,285 (3/รูป 97.6%, บางรูป 4–5) |
| val JSON | 9,036 | COCO val2017 5,000 + IPU24 val (travel 2,368, food 1,668) | 3/รูป |
| test | 2,000 | `test/test/00000–01999.jpg` ไม่บอกที่มา (ตัวอย่างขนาด 640×472 แบบ COCO/IPU24) | ส่ง `image_id,caption` (sample มี 3 แถวเติมไว้เป็นตัวอย่าง) |

- **รูปที่ Kaggle ให้มา = IPU24 เท่านั้น** (train 28,004 รูป 1.55 GB + val food 1,668 + val travel **125 จาก 2,368** + test 2,000) รวม 1.82 GB — **รูป COCO 119k ต้องหาเอง** (train2017 ต้นฉบับ 18 GB, val2017 0.8 GB)
- val travel หายไป 2,243 รูป ⇒ ต้องสงสัยว่า test = ส่วนหนึ่งของ val ที่ถูกดึงออก (ถ้าใช่ caption ใน val JSON ของรูปพวกนั้นก็ใช้ไม่ได้อยู่ดีเพราะไม่มีรูป แต่ควรเช็ค perceptual hash ระหว่าง test กับ COCO val2017 ด้วย เผื่อ test มีรูป COCO ที่ caption อยู่ใน val JSON — เป็น leak ที่ต้องรู้ก่อน ไม่ใช่เพื่อโกง แต่เพื่อไม่ให้ val score หลอกเรา)
- caption ยาวเฉลี่ย 67 ตัวอักษร (median 63, p95 98) ≈ 12–20 คำ; 41% มีช่องว่างข้างใน; สไตล์ = ประโยคบรรยายเดียว ระบุสี/จำนวน/ตำแหน่ง เช่น "ผู้ชายใส่เสื้อสีน้ำเงินยืนอยู่ 1คน ด้านหน้ามี…"
- caption ทั้ง 3 ของรูปเดียวกันต่างกันมาก (คนละคนเขียน) ⇒ BLEU มีเพดานตามธรรมชาติ และรางวัลไปตกกับ caption ที่ "กลาง ๆ เหมือน consensus" ไม่ใช่ caption ที่สร้างสรรค์

## 1. ทรัพยากรที่มีตอนนี้ (สำรวจแล้ว)

| อย่าง | สถานะ |
|---|---|
| GPU | RTX 3080 Ti Laptop **16 GB** — ตอนนี้ Hack 6 (`liver-train`, 40 epochs) กินอยู่ 9 GB ⇒ Hack 1 ต้องรอรันนั้นจบก่อนจึงเทรนได้ |
| RAM / CPU | 31 GB / 20 threads |
| Disk | **เหลือ 20 GB** (ใช้ไป 96%) — HF cache 54 GB; ลบได้ถ้าไม่ใช้: CLIP-bigG 9.5 GB, Qwen3-14B-AWQ 9.4 GB, siglip2-giant 7 GB, dinov2-giant 4.3 GB |
| โมเดลที่ cache แล้ว | `google/siglip2-so400m-patch14-384` (4.3 GB) ✔, `Qwen/Qwen3.5-4B` (มีแค่ config/tokenizer 22 MB — weights ยังไม่โหลด) |
| toolchain | uv, kaggle CLI 2.2.4, CUDA driver 13.2, venv ของ Hack 6 มี torch 2.14 (ไม่มี transformers) — Hack 1 สร้าง venv ใหม่ |

## 2. โมเดลตัวเลือก (survey HF 2026-09)

### 2a. VLM ที่รู้ภาษาไทยอยู่แล้ว (fine-tune ทั้งก้อน)

| โมเดล | ขนาด | หมายเหตุ |
|---|---:|---|
| **Qwen/Qwen3.5-2B / 4B** | 2.3B / 4.7B | early-fusion VL, 201 ภาษา (ไทยด้วย), ViT 24 ชั้น patch16 merge 2×2; LLM เป็น Gated DeltaNet + attention ⇒ ต้องใช้ **transformers v5** และ Triton kernels (fla) ; unsloth รองรับ vision LoRA: VRAM 2B ≈ 5 GB, 4B ≈ 10 GB (bf16 LoRA); **unsloth เตือนว่า QLoRA 4-bit ไม่แนะนำสำหรับ Qwen3.5** |
| **Qwen/Qwen3-VL-2B / 4B / 8B-Instruct** | 2.1B / 4.4B / 8.8B | transformer ธรรมดา เทรนง่ายกว่า, รองรับไทยดี, ecosystem ครบ (unsloth, peft, vLLM) ; 8B ต้อง 4-bit บน 16 GB |
| typhoon-ai/typhoon2-qwen2vl-7b-vision-instruct | 8.3B | Thai-tuned Qwen2-VL-7B (SCB10X) — เก่ากว่า Qwen3-VL แต่เป็นตัวเดียวที่ tuned ไทยจริง; 4-bit เท่านั้นบน 16 GB |
| nectec/Pathumma-llm-vision-3.0.0-preview / -re | 2.2B / 2.4B | ต่อยอด Qwen3.5-2B / Qwen3-VL-2B แต่เทรนเพื่อ **Thai OCR** ไม่ใช่ captioning — ไม่ได้เปรียบ |
| google/gemma-3-4b-it | 4.3B | รู้ไทย, มี vision, gated license |

### 2a-ii. รุ่นใหม่ปี 2026 ที่สำรวจเพิ่ม (2026-09-11) — อะไรใช้บน 16 GB ได้จริง

| โมเดล (วันที่ออก) | ขนาด | ใช้ local ได้? | หมายเหตุ |
|---|---:|:-:|---|
| **nectec/Pathumma-llm-text-4.0.0** (ส.ค. 2026) | 4.54B = LM 4.21B + **vision tower 333M** (ตรวจจาก safetensors header แล้ว มี `model.visual.*` ครบ) | ✔ เหมือน Qwen3.5-4B | สถาปัตยกรรม Qwen3.5-4B ที่ **continual-pretrain ภาษาไทย (ThaiLLM)** แล้ว SFT+DPO แบบ thinking; IFEval-TH 71.7, code-switching 97.9 — เป็น "Qwen3.5-4B ที่ไทยแข็งกว่า" ตัวเดียวที่มีในตลาด แต่ต้องพิสูจน์ว่า alignment ภาพยังไม่พังหลัง CPT (ทดสอบ zero-shot caption บน val-1000 ก่อน) และต้องปิด thinking ตอน generate |
| **google/gemma-4-E4B-it / E2B-it** (มี.ค. 2026, Apache-2.0) | 4.5B effective (8B รวม PLE) / 2.3B (5.1B) | ✔ QLoRA 10 GB; LoRA bf16 17 GB ← **ไม่พอ** | 140+ ภาษา (มีไทย), vision encoder เล็กมาก ~150M (16 ชั้น, 768 hidden) → 280 image tokens; unsloth มี notebook E4B-Vision; MMMU-Pro E4B 52.6 vs Qwen3.5-4B (สูงกว่า) — encoder เล็กอาจเสียเปรียบเรื่องรายละเอียดสี/จำนวน |
| google/gemma-4-12B-it "Unified" (พ.ค. 2026) | 12B, ไม่มี encoder (patch → linear) | ✖ QLoRA ~ 8–9 GB weights + activations, เสี่ยง OOM/ช้ามาก | ไว้เป็น cloud option |
| Qwen/Qwen3.6-27B, 35B-A3B (เม.ย. 2026) · Qwen/Qwen3.8-27B, Flash-Next (ส.ค. 2026) | ≥27B | ✖ | ไม่มีรุ่นเล็ก; ตระกูล 3.6/3.8 ออกเฉพาะ 27B/MoE — รุ่นเล็กสุดที่ใหม่ที่สุดยังเป็น **Qwen3.5-0.8B/2B/4B/9B** |
| iapp/openthai2.0-qwen3.8-27b (ส.ค. 2026) | 27B | ✖ (NVFP4 ยังต้อง ~16 GB+ และ inference-only) | Thai-tuned Qwen3.8 มี vision — ใช้เป็น **teacher** สร้าง caption ไทยผ่าน cloud ได้ถ้าจะทำ distillation |
| typhoon-ai/typhoon-ocr1.5-2b (พ.ย. 2025) | 2B (Qwen3-VL-2B) | ✔ | Thai OCR เท่านั้น ไม่เกี่ยวกับ caption; Typhoon ไม่มี vision รุ่นใหม่กว่า typhoon2-qwen2vl-7b |
| LiquidAI/LFM2.5-VL-3B (ส.ค. 2026) | 3B | ✔ | edge model, ภาษาไทยไม่อยู่ในรายการที่รองรับ — ข้าม |
| InternVL3.5-2B/4B (ส.ค. 2025) | 2–4B | ✔ | Qwen3-based, multilingual แต่ไม่เน้นไทย; ไม่เห็นข้อได้เปรียบเหนือ Qwen3-VL/3.5 |
| DeepSeek-V4-Flash-Vision, GLM-5.3-Flash, Kimi-K3, Gemma-4-26B-A4B | ใหญ่มาก | ✖ | cloud/API เท่านั้น — ใช้เป็น teacher/pseudo-label ได้ |

vision encoder เดี่ยวสำหรับ Phase 3: ยังไม่มี SigLIP 3; ตัวเลือกที่ใหม่กว่า SigLIP2 คือ `facebook/PE-Core-L14-336 / G14-448` (Perception Encoder, 2025) และ DINOv3 (cache แล้ว) — SigLIP2-so400m ยังเป็น default ที่ปลอดภัยสุดสำหรับ caption

**สรุปผลกระทบต่อแผน:** ผู้ท้าชิงหลักบน 16 GB เหลือ 4 ตัว → ทำ **zero-shot bake-off บน val-1000 ก่อน fine-tune** (ครึ่งวัน): Qwen3-VL-2B/4B, Qwen3.5-2B/4B, Pathumma-llm-text-4.0.0, Gemma-4-E4B (QLoRA เท่านั้น) — เลือก 2 ตัวที่ BLEU zero-shot สูงสุดและ Thai อ่านลื่นสุดไป Phase 1/2 ส่วน Pathumma-4.0.0 น่าสนใจที่สุดในเชิง "โมเดลไทย" แต่เสี่ยงเรื่อง alignment ภาพหลัง CPT

### 2b. ประกอบเอง (ตามที่อยากทำ): image encoder + Thai LLM + projector (LLaVA-style)

| ชิ้น | ตัวเลือก |
|---|---|
| encoder | `siglip2-so400m-patch14-384` (cache แล้ว; 729 tokens → pool เหลือ 81–196), หรือ DINOv3 ViT-L (cache แล้ว; semantic น้อยกว่า SigLIP สำหรับ caption) |
| Thai LLM | `typhoon-ai/typhoon2.5-qwen3-4b` (Thai-tuned Qwen3-4B, apache), `typhoon2.1-gemma3-4b`, `Qwen/Qwen3-4B`, `Qwen3.5-2B` (ใช้เฉพาะส่วน text) |
| projector | MLP 2 ชั้น + 2×2/3×3 average pool (แบบ LLaVA-1.5 / PaliGemma) |
| สูตร | stage 1 เทรน projector อย่างเดียว (LLM freeze) บน caption ทั้งหมด → stage 2 LoRA LLM + projector |

ทำไมไม่ใช้ BLIP-2 ตรง ๆ: BLIP-2 = ViT-g + Q-Former (32 tokens) + OPT/Flan-T5 ซึ่งเป็นสถาปัตยกรรมเดียวกับ Qwen3-VL (ViT + MLP merge ~196 tokens + Qwen3 LLM) ต่างกันที่ (1) OPT ไม่รู้ภาษาไทย จึงต้อง generate อังกฤษแล้วแปล = เพดาน 46–48 บน LB, (2) Qwen3-VL align ภาพ↔ภาษามาแล้วด้วยข้อมูลหลายพันล้านคู่ ส่วน BLIP-2 ใช้ ~129M คู่อังกฤษ และถ้าสลับ LLM เป็นไทยต้อง align ใหม่ด้วย 430k caption ของเรา (= Phase 3), (3) Q-Former บีบเหลือ 32 tokens ทิ้งรายละเอียดสี/จำนวน/ตำแหน่งที่ reference ชุดนี้ชอบระบุ

ทางกลาง (เพิ่มเป็นตัวเลือกใน Phase 3): ใช้ ViT-g + Q-Former ของ BLIP-2 ที่ align แล้ว ต่อ linear projection ใหม่เข้า Typhoon/Qwen3 แล้ว LoRA — ถูกกว่าประกอบจาก SigLIP2 ตั้งแต่ศูนย์ แต่ยังติด 32 tokens

งานอ้างอิงสำหรับไทยที่มีอยู่: `Natthaphon/thaicapgen-clip-phayathai` / `-clip-gpt2` (CLIP encoder + PhayaThaiBERT/GPT-2 decoder 0.6B, เทรนบน Thai MSCOCO + IPU24 ชุดเดียวกับเรา) — ไม่รายงาน BLEU แต่ยืนยันว่า encoder-decoder เล็ก ๆ ก็เรียนข้อมูลชุดนี้ได้

## 3. ประเมินต้นทุนคำนวณ (บน 3080 Ti Laptop, ประมาณ 15 TFLOPS ใช้จริง)

สมมติ 1 sample = รูป 448 px (≈196 visual tokens) + prompt 20 + caption ≈ 35 tokens ≈ 260 tokens; LoRA ≈ 6·N·T FLOPs

| แผน | วินาที/sample | 28k IPU24 ×1 epoch | 142k ทั้งหมด ×1 epoch |
|---|---:|---:|---:|
| Qwen3-VL-2B / Qwen3.5-2B LoRA | ~0.3 | **~2.5 ชม.** | ~12 ชม. |
| Qwen3-VL-4B / Qwen3.5-4B LoRA | ~0.55 | ~4.5 ชม. | **~22 ชม.** |
| Qwen3-VL-8B QLoRA 4-bit | ~1.2 | ~9 ชม. | ~48 ชม. (ไม่คุ้มบน laptop) |
| ประกอบเอง SigLIP2 + 4B LLM (81 vis tokens) | ~0.25 ต่อ stage | ~2 ชม./stage | ~10 ชม./stage ×2 stage |

- Inference: test 2,000 รูป beam 4 ≈ 20–40 นาที; val เต็ม 9,036 ≈ 1.5–3 ชม. ⇒ ใช้ **val ย่อย 1,000 รูป (stratified COCO/travel/food)** ระหว่างพัฒนา, val เต็มเฉพาะตอนสรุป
- แต่ละ caption เพิ่ม (3/รูป) = คูณเวลา ⇒ epoch แรกใช้ 1 caption/รูป (สุ่ม หรือ "medoid" ดู §5)
- Disk ที่ต้องมี: Kaggle 1.8 GB + COCO ย่อเหลือด้านยาว 448 px ≈ **5 GB** (stream จาก parquet `detection-datasets/coco` 40 ไฟล์ × 480 MB ทีละไฟล์ ไม่ต้องเก็บ 19 GB) + weights (2B ≈ 4.5 GB, 4B ≈ 9 GB) ⇒ **ต้องเคลียร์ HF cache อย่างน้อย ~10 GB ก่อน** (แนะนำลบ CLIP-bigG + Qwen3-14B-AWQ = 19 GB)

### ถ้าเช่า cloud (ทางเลือก ไม่บังคับ)

| ที่ | ราคาโดยประมาณ | 4B LoRA 142k ×1 epoch |
|---|---|---|
| Kaggle T4×2 / P100 (ฟรี 30 ชม./สัปดาห์) | 0 | ช้ากว่า laptop, 16 GB, session 12 ชม. — ใช้ได้กับ 2B |
| RunPod / Vast RTX 4090 24 GB | ~$0.35–0.5/ชม. | ~8 ชม. ≈ $4 |
| A100 80 GB | ~$1.3–1.8/ชม. | ~4 ชม. ≈ $7 (เปิด batch ใหญ่ + full-FT 2B ได้) |
| H100 80 GB | ~$2.5–3/ชม. | ~2.5 ชม. ≈ $8 |

สรุป: งานทั้งหมดทำบน laptop ได้ภายใน ~3–5 วัน (ส่วนใหญ่รอเทรน) ; cloud ช่วยเฉพาะตอนอยากลอง 4B บนข้อมูลเต็มหลาย epoch หรือ 8B

## 4. กลไกของ BLEU ที่ต้องเล่นให้ถูก

1. **ทำ scorer ให้ตรงผู้จัด**: `sacrebleu.corpus_bleu(hyps, [ref1, ref2, ref3], tokenize="flores101")` (ต้อง `pip install sacrebleu[ja]`-style extra `sentencepiece`) — เก็บทั้ง corpus BLEU และ BLEU ต่อ domain (COCO / travel / food)
2. **Brevity penalty**: BLEU ลงโทษประโยคสั้นกว่า ref (เลือก ref ที่ยาวใกล้ที่สุด) แต่ไม่ลงโทษยาวเกิน (แค่ precision ตก) ⇒ จูน `max_new_tokens`, `length_penalty`, `num_beams`, `repetition_penalty` บน val ให้ BLEU สูงสุด — ค่า default ของ VLM ชอบพูดยาว/ขึ้นต้นด้วย "ภาพนี้แสดง…" ซึ่งกิน precision
3. **Consensus caption**: ref 3 อันเขียนต่างกันมาก ⇒ เทรนด้วย caption ที่เป็น "medoid" (อันที่ BLEU สูงสุดเมื่อเทียบกับอีก 2 อัน) แทนสุ่ม มักดัน BLEU ขึ้นเพราะโมเดลเรียน style กลาง; ทดลองเทียบ (i) สุ่ม 1/รูป (ii) medoid (iii) ทั้ง 3
4. **MBR decoding**: สุ่ม/beam ได้ N candidates (8–16) ต่อรูป แล้วเลือกอันที่ BLEU เฉลี่ยเทียบกับตัวอื่นสูงสุด — ปกติได้ +1–3 BLEU ฟรี ๆ และใช้รวมหลายโมเดล (ensemble ระดับ candidate) ได้โดยไม่ต้องรวม logits
5. **Domain prior**: test อาจเป็น IPU24 ล้วน (ชื่อไฟล์ `00000.jpg` แบบ IPU24) ⇒ น้ำหนักเทรนควรเอียงไป IPU24 (28k) แล้วใช้ COCO เป็นตัวเสริม; ยืนยันด้วย phash + ดูสัดส่วน BLEU ต่อ domain บน val vs public LB
6. **ความสะอาดข้อความ**: ใช้ tokenizer ที่ไม่ทำ NFKC แปลก ๆ กับสระไทย (สำคัญกับ Gemma/Qwen) ; strip ช่องว่างหัวท้าย, ไม่มี newline, ห้ามว่าง (sample_submission เตือน "at least 5 characters")

## 5. แผนที่แนะนำ (เรียงตามผลตอบแทน/ความเสี่ยง)

### Phase 0 — โครง + scorer + baseline (≈ ครึ่งวัน, ไม่ต้องรอ GPU ว่าง)
1. สร้าง package `thai_caption` แบบเดียวกับ Hack 4/6 (`pyproject.toml` + `src/` + `configs/` + `Makefile` + `scripts/`), venv ใหม่ (transformers v5, peft, unsloth, sacrebleu+sentencepiece, pythainlp)
2. `make data`: โหลด Kaggle zip 1.8 GB → `datasets/raw/`; stream COCO train2017/val2017 จาก parquet ย่อเหลือ 448 px → `datasets/coco448/` (~5 GB); ทำ `datasets/index.parquet` (path, domain, split, captions[3], medoid_idx, char_len)
3. `eval.py`: sacrebleu flores101 3-ref + per-domain + brevity-penalty report; unit test ด้วย caption ref-vs-ref (ควรได้ ~20–30) เพื่อรู้เพดาน
4. EDA notebook: ความยาว, คำที่พบบ่อยตาม domain, phash test↔COCO val2017 (leak check), ตัวอย่างสไตล์ caption
5. **Baseline zero-shot bake-off** (Qwen3-VL-2B/4B, Qwen3.5-2B/4B, Pathumma-llm-text-4.0.0, Gemma-4-E4B; prompt ไทยสั้น ๆ "บรรยายภาพนี้เป็นประโยคเดียว…") บน val-1000 → รู้ว่า style gap ใหญ่แค่ไหน และส่ง 1 submission เพื่อ calibrate val↔LB

### Phase 1 — LoRA VLM บน IPU24 (≈ 1 วัน, หลัง GPU ว่าง) — คาดว่าข้าม 50 ได้ตรงนี้
6. Qwen3-VL-2B-Instruct LoRA r=16 (vision + language + attn + mlp), lr 2e-4, 1 epoch บน IPU24 28k (medoid caption) ≈ 2.5 ชม. → val-1000 → ส่ง
7. จูน decode บน val: beams {1,3,5}, length_penalty {0.8,1.0,1.2}, max_new_tokens {40,64}, repetition_penalty ; แล้ว MBR N=8
8. เพิ่ม COCO: IPU24 ทั้งหมด + COCO 20–40k (สุ่ม) 1 epoch — ดูว่า COCO ช่วยหรือฉุด (BLEU ต่อ domain)

### Phase 2 — scale + ensemble (≈ 1–2 วัน)
9. Qwen3.5-4B (หรือ Qwen3-VL-4B ถ้า kernel ของ 3.5 มีปัญหา) LoRA บนชุดเดียวกัน ~5–10 ชม.; เทียบ 2B vs 4B
10. เทรนด้วย caption ทั้ง 3 (หรือ 2 epochs) กับตัวที่ดีที่สุด
11. MBR ข้ามโมเดล (2B + 4B + seeds) เป็น ensemble สุดท้าย; เลือก config จาก val เต็ม 9,036 ก่อนส่ง final

### Phase 3 — "โมเดลไทยของตัวเอง" (การทดลองที่ขอไว้ ≈ 1–2 วัน เทรน)
12. SigLIP2-so400m (freeze) + pool 3×3 → 81 tokens + MLP projector + `typhoon2.5-qwen3-4b`: stage 1 projector-only บน caption ทั้งหมด 430k (~10 ชม.) → stage 2 LoRA (~10 ชม.) → เทียบกับ Phase 1/2 บน val เต็ม. คาดว่าจะ **แพ้** Qwen3-VL/3.5 ที่ align vision มาแล้วด้วยข้อมูลพันล้านคู่ แต่เป็นงานที่คุ้มเขียนลง README เพราะเห็นชัดว่า "alignment ต้องใช้ข้อมูลเท่าไหร่" — ถ้าเวลาไม่พอ ตัดข้อนี้ก่อน
13. ตัวเลือก cheaper: ใช้ vision tower + projector จาก Qwen3-VL-2B แต่สลับ LLM เป็น Typhoon (ทั้งคู่เป็นสถาปัตยกรรม Qwen3 → embedding dim ต่าง ต้องเทรน projector ใหม่อยู่ดี) — ความเสี่ยงสูง ไม่แนะนำเป็นเส้นหลัก

## 6. ความเสี่ยงที่รู้แล้ว

- Qwen3.5 (Gated DeltaNet) ต้อง transformers v5 + `flash-linear-attention`/`causal-conv1d` (Triton) — ถ้า build บนเครื่องนี้มีปัญหา ให้ถอยไป Qwen3-VL-4B ทันที ไม่เสียเวลา debug
- GPU ถูก Hack 6 ใช้อยู่ 9 GB — ห้ามเริ่มเทรน Hack 1 จนกว่ารันนั้นจบ (2B LoRA ต้อง ~6–8 GB ตอน batch 4 + grad ckpt; 4B ~11–13 GB)
- Disk 20 GB: ต้องเคลียร์ cache ก่อนโหลด COCO + weights; เก็บ checkpoint เป็น LoRA adapter (หลักร้อย MB) ไม่ merge
- metric ไม่ยืนยัน 100% ว่าเป็น flores101 SPM — ถ้า val กับ LB ห่างกันผิดปกติในการส่งครั้งแรก ให้ลอง tokenize `char`/`newmm` เพื่อดูว่าตัวไหนสัมพันธ์กับ LB มากกว่า
- test อาจไม่มี COCO เลย ⇒ อย่าให้ COCO val ครอบงำการตัดสินใจ ให้ดู BLEU ของ travel/food เป็นหลัก

## Sources

- Kaggle competition pages (`kaggle competitions pages --content`): Evaluation "Bleu-score"; SS3 `thai-language-image-captioning` Evaluation: sacrebleu, 3 refs, sentencepiece flores101
- Kaggle leaderboard + our submissions (45.97 / 39.91 / 45.61 public)
- Public notebooks: `jamvessapoo/hack2-image-processing` (BLIP-2 LoRA + translate, 93 votes), `phumrapeecurrentone/at-least-5-characters-long` (unsloth Qwen2-VL-7B), SS3 `pattaratipaksorn/baseline` (GIT + translate), `besuto/zeroshot-blip2-infer`
- HF: Qwen/Qwen3.5-{2B,4B} README + config.json; Qwen/Qwen3-VL-{2B,4B,8B}-Instruct; typhoon-ai/typhoon2-qwen2vl-7b-vision-instruct; typhoon-ai/typhoon2.5-qwen3-4b; nectec/Pathumma-llm-vision-3.0.0-{preview,re}; Natthaphon/thaicapgen-clip-{phayathai,gpt2}; datasets detection-datasets/coco (parquet 19.2 GB train / 0.8 GB val), phiyodr/coco2017
- unsloth docs "Qwen3.5 Fine-tuning Guide" (VRAM 2B 5 GB / 4B 10 GB bf16 LoRA; QLoRA not recommended; transformers v5)
