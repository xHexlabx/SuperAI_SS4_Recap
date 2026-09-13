# Open-source object detection for a small grayscale ultrasound dataset — survey memo (verified 2026-09-11)

Target: ~1–5k grayscale US images, YOLO-txt labels, several lesion classes, one RTX 3080 Ti 16 GB, Python 3.11/3.12, PyTorch 2.x via `uv`. Everything below was checked against PyPI JSON, GitHub, HF Hub, docs or the papers on 2026-09-11; items that could not be verified are marked *(unverified)*.

## 0. Environment facts (PyPI, 2026-09-11)

| Package | Latest | Python | Uploaded | Note |
|---|---|---|---|---|
| torch / torchvision | 2.14.0 / 0.29.0 | >=3.10 | 2026-09-02 | |
| ultralytics | 8.4.147 | >=3.8 | 2026-09-11 | AGPL-3.0; releases almost daily |
| rfdetr | 1.10.1 | >=3.10 | 2026-09-07 | needs `torch>=2.2`, `transformers>=5.1,<6`, `supervision>=0.29` |
| transformers | 5.17.0 | >=3.10 | 2026-09-09 | has RT-DETRv2, D-FINE and DEIMv2 detection classes |
| ensemble-boxes | 1.0.9 | any | 2022-04-20 | last PyPI release 2022 (repo still pushed 2026-07) |
| sahi | 0.12.6 | >=3.8 | 2026-08-16 | backends: ultralytics, rfdetr, HF, mmdet, torchvision |
| faster-coco-eval | 1.8.0 | >=3.10 | 2026-08-24 | drop-in pycocotools |
| pycocotools | 2.0.11 | >=3.9 | 2025-12-15 | |
| torchmetrics | 1.9.0 | >=3.10 | 2026-03-09 | `MeanAveragePrecision` |
| mmdet / mmcv | 3.3.0 / 2.2.0 | – | 2024-01 / 2024-04 | no releases since; no prebuilt mmcv wheels for current torch / py3.12 |

`uv` note: `rfdetr` pins `transformers<6`; ultralytics has no transformers dependency, so one venv with `ultralytics rfdetr transformers ensemble-boxes faster-coco-eval sahi` resolves. DEIMv2 and RT-DETRv4 are git-clone repos (no PyPI), both tested on Python 3.11.

## 1. Ultralytics (8.4.147)

Model families: YOLO26 (Jan 2026, current flagship), YOLO12, YOLO11, YOLOv10/9/8, RT-DETR (`rtdetr-l`/`rtdetr-x`), YOLO-World, YOLOE, SAM 3/2/1, YOLO-NAS.

COCO val (docs):

| Model | mAP50-95 | Params | Notes |
|---|---|---|---|
| YOLO26n/s/m/l/x | 40.9 / 48.6 / 53.1 / 55.0 / 57.5 | 2.4 / 9.5 / 20.4 / 24.8 / 55.7 M | e2e-head mAP 0.6–0.8 lower; DFL removed, MuSGD, ProgLoss, STAL |
| YOLO12n…x | 40.6 / 48.0 / 52.5 / 53.7 / 55.2 | 2.6…59.1 M | docs warn "training instability, elevated memory" |
| RT-DETR-L / X | 53.0 / 54.8 | | `RTDETR("rtdetr-l.pt")`, same data yaml, `deterministic=False` |

YOLO26 behaviours: predict/val default `nms=None` uses one-to-many head + NMS (more accurate); `nms=False` = NMS-free one-to-one head. TTA `augment=True` = scales `[1, 0.83, 0.67]` + lr flip, skipped with a warning when `end2end` is active. No built-in multi-checkpoint ensembling in 8.x — do it externally with WBF. Grayscale: `channels: 1` in the data yaml, or 3-channel replicated grayscale (keeps COCO pretraining). Memory scales ~imgsz²; use `batch=0.7` (fraction autobatch) with `amp=True`.

Defaults: `epochs=100, patience=100, batch=16, imgsz=640, optimizer=auto, lr0=0.01, lrf=0.01, warmup_epochs=3, weight_decay=5e-4, cos_lr=False, close_mosaic=10, amp=True`. Aug: `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0, translate=0.1, scale=0.5, shear=0, perspective=0, flipud=0, fliplr=0.5, mosaic=1.0, mixup=0, cutmix=0, copy_paste=0 (segment only)`. Inference: `conf` predict=0.25, **val=0.001**; `iou=0.7`, `max_det=300`.

Suggested small-US recipe (recommendation, not a doc value): `epochs 150–300, patience 50, cos_lr=True, close_mosaic=15, imgsz=1024, mosaic=1.0, mixup=0.1–0.15, degrees=5–10, translate=0.1, scale=0.3–0.5, fliplr=0.5, flipud=0, hsv_s=0, hsv_v=0.3–0.4, perspective=0–0.0005, AdamW lr0=1e-3 or auto`. BUSI benchmarks show seed variance ±0.01–0.02 mAP.

## 2. Transformer detectors

| Framework | Repo / version | Weights | COCO AP | Custom-data ease | Small-data evidence |
|---|---|---|---|---|---|
| **RF-DETR** (Roboflow, ICLR 2026) | github.com/roboflow/rf-detr, `rfdetr 1.10.1`, Apache-2.0 (N/S/M/L); XL/2XL under PML | auto-download; DINOv2 backbone | N 48.4 (384), S 53.0 (512), M 54.7 (576), L 56.5 (704), XL 58.6, 2XL 60.1 | best: `train(dataset_dir=…)` auto-detects **YOLO (`data.yaml`) or COCO**; defaults `epochs=100, lr=1e-4, lr_encoder=1.5e-4, use_ema=True, early_stopping`; 16 GB → `batch_size=4, grad_accum_steps=4`, `gradient_checkpointing` | RF100-VL: RF-DETR-M 61.7 AP vs D-FINE-M 60.6; DINOv2 init "+2.4 AP on small datasets". Liver intra-op US (n=1,035): RF-DETR precision 80 %, mAP50 ≈70 %, vs YOLO11m sens 63 %/prec 62 % |
| **DEIMv2** (DINOv3) | github.com/Intellindust-AI-Lab/DEIMv2 (no PyPI; Py 3.11), non-commercial licence | HF `Intellindust/DEIMv2_DINOv3_{S,M,L,X}_COCO` | S 50.9, M 53.0, L 56.0, X 57.8 | COCO JSON only; also in HF transformers 5.17 | DEIM v1 on ultrasound: mixed (Thyroid-I 0.637 vs YOLO11 0.608; TN3K 0.467 vs 0.529; BUSI 0.435 vs YOLOv12 0.454) |
| **D-FINE** (ICLR 2025) | github.com/Peterande/D-FINE, Apache-2.0; HF `DFineForObjectDetection` | `ustc-community/dfine-{n,s,m,l,x}-{coco,obj365,obj2coco}` | L 54.0, X 55.8; Obj365→COCO: L 57.3, X 59.3 | COCO JSON / HF example script | strong on RF100-VL, slightly below RF-DETR; slow early convergence on small data |
| **RT-DETRv2** | HF `PekingU/rtdetr_v2_r{18,34,50,101}vd`; ultralytics `rtdetr-l/x` | | R50 ~53 | ultralytics path uses YOLO yaml directly | cholelithiasis US: RT-DETR 0.63/0.40 vs YOLOv8 0.61/0.38 |
| **RT-DETRv4** (ECCV 2026) | github.com/RT-DETRs/RT-DETRv4 | GDrive | S 49.8 … X 57.0 | COCO JSON | none on medical |
| **Co-DETR / DINO** | mmdet 3.x | Swin-L / ViT-L | up to 66.0 | mmcv has no wheels for torch 2.14 / py3.12; far beyond 16 GB | impractical here |
| Grounding DINO / OWLv2 | | | | mmdet pain / no training script | no advantage for closed-set lesions |

Honest picture: off-the-shelf DEIM/RT-DETR are roughly on par with YOLO11/12 and lose on TN3K/BUSI. RF-DETR is the only off-the-shelf DETR with systematic small-dataset evidence and an ultrasound result where it out-precised YOLO11. Conclusion: diversity, not replacement — YOLO gives recall, DINO-backbone DETRs give precision; that is what WBF rewards.

## 3. Foundation-model backbones for ultrasound

| Backbone | Status for detection | Ultrasound evidence |
|---|---|---|
| **DINOv2** | already the RF-DETR backbone | RF-DETR ablation: +2.4 AP on small data |
| **DINOv3** | in DEIMv2, RT-DETRv4 teacher | medical DINOv3 benchmark (arXiv 2509.06467): frozen DINOv3 *not* superior on CAMUS cardiac US |
| **UltraSam** (ViT-B SAM on 280k US masks; CC BY-NC-SA) | mmdet-based | BUS-BRA 60.7 vs 60.1 ResNet50-ImageNet — modest gain, heavy stack |
| USFM, MedSAM/2, BiomedCLIP, RadImageNet, EchoCLIP | classification/segmentation only | nothing plug-and-play for YOLO/DETR |

## 4. Ensembling and evaluation tooling

- **WBF**: `ensemble-boxes` 1.0.9: `weighted_boxes_fusion(boxes (normalized xyxy), scores, labels, weights, iou_thr=0.55–0.6, skip_box_thr=0.001, conf_type='avg')`. Also use it for hflip TTA where the framework has no `augment` (RF-DETR, DEIMv2, HF models).
- **sahi**: sliced inference; lesions are not tiny relative to the frame, so a maybe.
- **Local mAP**: build one evaluator on the competition's exact metric definition and score every model and every fusion with it. Ultralytics' own `val` AP integration has known small discrepancies vs pycocotools.

## 5. 2025–26 ultrasound / medical comparisons with numbers

| Study | Data | Result (AP50 / AP50-95) |
|---|---|---|
| Prior-Guided DETR (arXiv 2601.02212) | Thyroid-I 1,849; TN3K 3,493; BUSI 665 | Thyroid-I: YOLOv11 0.902/0.608, YOLOv12 0.933/0.596, DEIM 0.906/0.637, Faster R-CNN 0.865/0.438. TN3K: YOLOv11 0.848/0.529, DEIM 0.715/0.467. BUSI: YOLOv12 0.663/0.454, YOLOv11 0.587/0.365, DEIM 0.586/0.435 |
| Nodule-DETR (arXiv 2601.01908) | 7,301 thyroid US | YOLOv11 0.584/0.908, Faster R-CNN 0.424/0.848, Nodule-DETR 0.611/0.948 |
| Intra-operative liver US (Surg Endosc 2026) | 1,035 images, 1 class | YOLO11m sens 63 %/prec 62 %; RF-DETR prec 80 %, mAP50 ≈70 % |
| Cholelithiasis (CSBJ 2025) | 1,210 US | DETR-R50 0.65/0.43, RT-DETR 0.63/0.40, YOLOv8 0.61/0.38 |
| YOLOs on BUSI (QIMS 2026) | 647 imgs, 512 px, 8 seeds | mAP50: yolov5s 0.98, yolo11n 0.96, yolo11m 0.90, yolo11x 0.84 — larger is worse on 600 images |
| Auto-DFLLs liver (npj Digit Med 2026) | 5,258 videos | ResNet34-BiFPN AP50 0.777 |

## 6. Recommended shortlist for a 16 GB ensemble

1. **YOLO26m / YOLO26l** (ultralytics 8.4.147), imgsz 1024, `batch=0.7`, cos_lr, close_mosaic 15, flipud 0, hsv_s 0, mixup 0.1, 200–300 epochs; predict with `augment=True, conf=0.001`. Fallback YOLO11l.
2. **RF-DETR-M (and L)**, `rfdetr 1.10.1`, `train(dataset_dir=<yolo folder>)`, `batch_size=4, grad_accum_steps=4, epochs=100–150, lr=1e-4, lr_scheduler="cosine", early_stopping=True`. DINOv2 backbone = best documented small-data transfer.
3. **D-FINE-L from Objects365 weights** via HF transformers, 640 px, batch 4–8 with AMP. Or **DEIMv2-M/L** if the non-commercial licence is acceptable.
4. Optional: **RT-DETR-L through ultralytics**.

Fuse with WBF (`iou_thr≈0.55–0.6`, `skip_box_thr=0.001`, weights tuned on OOF), evaluate on the competition metric, keep hflip TTA for the DETRs via WBF. Avoid mmdet/Co-DETR/UltraSam.

## Sources

- PyPI JSON: ultralytics, rfdetr, ensemble-boxes, sahi, faster-coco-eval, torchmetrics, pycocotools, mmdet, mmcv, transformers, torch
- Ultralytics: https://github.com/ultralytics/ultralytics/releases · https://docs.ultralytics.com/models/yolo26/ · https://docs.ultralytics.com/models/rtdetr/ · https://docs.ultralytics.com/modes/train/ · https://docs.ultralytics.com/guides/end2end-detection/ · https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml · https://github.com/ultralytics/ultralytics/issues/23785 · https://arxiv.org/abs/2606.03748
- RF-DETR: https://github.com/roboflow/rf-detr · https://rfdetr.roboflow.com/latest/learn/train/training-parameters/ · https://github.com/roboflow/rf-detr/issues/507 · https://arxiv.org/abs/2511.09554
- DEIMv2: https://github.com/Intellindust-AI-Lab/DEIMv2 · https://huggingface.co/Intellindust/DEIMv2_DINOv3_L_COCO · https://arxiv.org/abs/2509.20787
- D-FINE: https://github.com/Peterande/D-FINE · https://huggingface.co/ustc-community/dfine-large-obj365 · https://arxiv.org/abs/2410.13842
- RT-DETRv2/v4: https://huggingface.co/docs/transformers/en/model_doc/rt_detr_v2 · https://github.com/RT-DETRs/RT-DETRv4 · https://arxiv.org/abs/2510.25257
- Co-DETR / mmdet: https://github.com/Sense-X/Co-DETR · https://github.com/open-mmlab/mmcv/issues/3319
- Foundation models: https://arxiv.org/html/2509.06467 · https://github.com/CAMMA-public/UltraSam · https://github.com/openmedlab/USFM · https://github.com/BMEII-AI/RadImageNet
- Tools: https://github.com/ZFTurbo/Weighted-Boxes-Fusion · https://github.com/obss/sahi · https://github.com/MiXaiLL76/faster_coco_eval · https://github.com/ultralytics/yolov5/issues/7658
- Ultrasound papers: https://arxiv.org/html/2601.02212 · https://arxiv.org/abs/2601.01908 · https://pubmed.ncbi.nlm.nih.gov/42414769/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC12595354/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC13350475/ · https://pubmed.ncbi.nlm.nih.gov/40381258/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC13260369/
