# Research memo: Liver Ultrasound Detection (SuperAI SS4 L2 Hack 6) — 2026-09-11

## 0. Facts about THIS competition

- Kaggle `liver-ultrasound-detection`, "Super AI Engineer 2023 (Level 2) – Image Processing Hackathon", supported by CU / Red Cross / AIAT (data very likely from King Chulalongkorn Memorial Hospital — the group of Tiyarattanachai et al., §1). Ran 27–31 May 2024; 39 teams, 415 submissions; 4 subs/day; 50/50 public/private. Discussion, Code and Models tabs are empty — no public notebooks or write-ups exist (Thai and English searches).
- Leaderboard: Pangpuriye-TA 0.22092, Kiddee-3 0.21476, … Optimizer-6 (our house) 0.19502 (#10); the organisers' "baseline" 0.18131.
- Data: train 14,448 images / 7,222 annotations; val 4,898 / 2,445; test 5,153. `mapping.xlsx` (19,346 rows): `machine` 17,998 (8,998 positive / 9,000 negative), `mobile` 1,348 (669 positive / 679 negative). ~53 % of train+val images contain NO lesion; ~7 % are phone photos. Test file sizes are bimodal (≈70–250 KB vs 1–4 MB), i.e. a substantial share of phone photos in test.
- 7 classes: FFC=0, FFS=1, HCC=2, cyst=3, hemangioma=4, dysplastic=5, CCA=6. Labels YOLO txt; submission is Pascal-VOC pixel boxes: `Image File,Annotation,Label`, `Annotation="[[x1,y1,x2,y2],...]"`, `Label="[c,...]"`, `[]` for no-lesion images. **There is NO confidence column.**
- Metric page says only "Mean Average Precision (mAP)"; IoU threshold and implementation are not published.

Implication of "no confidence column" (most important finding): if the host scores with pycocotools/torchmetrics with equal scores, detections are stable-sorted, so (a) every extra box is a permanent false positive on the PR curve — the usual "dump everything at conf 0.001" trick HURTS, and (b) tie order = file order, so writing boxes in descending model-confidence order may raise AP for free. Both must be probed with real submissions.

## 1. Published work on focal liver lesion (FLL) detection in ultrasound

| Study | Data | Detector | Numbers |
|---|---|---|---|
| Tiyarattanachai 2021, PLoS ONE (Chula) | 40,397 train imgs, 20,432 FLLs, 5 classes (HCC, cyst, hemangioma, FFS, FFI), 17 US machines | RetinaNet-R50, 1333×800, 25 ep; fan region cropped; aug: translate/rotate/scale/hflip/motion blur/contrast/brightness/HSV | detection rate 87.0 % internal, 75.0 % external; FPs = vessels 12.3 %, heterogeneous parenchyma 7.4 %, renal cysts 6.8 %; FNs mostly <1 cm |
| Tiyarattanachai 2022, Sci Rep (videos) | 446 videos | RetinaNet + hard-negative mining (28k difficult frames + 27k FP-structure frames) | detection HCC 100 %, FFS 96.7 %, hemangioma 85.2 %, cyst 82.4 %; FP rate 0.7 % |
| Two-stage HCC screening, J Assoc Med Sci (Thai) | 2,208 imgs (HCC 543, hemangioma 1,665) | YOLOv4 + ResNet-50 crop classifier | precision 0.74→0.88, HCC recall 0.68→0.72 vs single-stage |
| Radiology: AI 2022 | 2,551 imgs, 6 lesion types | DETR vs Faster R-CNN | DETR better on all tasks |
| npj Digit Med 2026 (multicenter video) | 5,258 videos | ResNet34-FPN + temporal fusion | AP50 0.777 |
| Surg Endosc 2026 (intraoperative) | 1,035 imgs | YOLO11m vs RF-DETR vs Cascade R-CNN | YOLO11m sens 63 % / prec 62 %; RF-DETR prec 80 %, mAP50 ≈70 % |
| JHC (Dove) YOLOR | 6,001 lesions, 8 classes | YOLOR-W6/D6 | benign-vs-malignant mAP 0.54–0.56; **8-class mAP only 0.30–0.31** |

Take-aways: (1) fine-grained class mAP is the bottleneck (0.30 for 8 classes vs 0.55 binary) — matches the 0.22 top score here; (2) DETR-family gives precision, YOLO gives recall — ensemble both; (3) dominant FP sources are vessels, parenchyma and renal cysts → hard-negative mining works; (4) two-stage detector + crop classifier improved precision on Thai HCC/hemangioma data.

Ultrasound augmentation evidence: Tupper 2025 (14 augs × 10 datasets): zoom, random crop, rotate ≈ +5 %, brightness +3 %; best = TrivialAugment over the top 3–5 augs, not all 14. UltraAugment (CVPRW 2024): fan-shape-preserving crop/elastic/noise. Noise-only augmentation hurt detectors (YOLO) in two studies — skip speckle injection for detectors.

## 2. Competition recipes that actually gave gains

- VinBigData CXR 1st: Detectron2-R101 + YOLOv5 + EffDet-D2, WBF (0.319→0.331); 2-class abnormal/normal image filter used by most teams. Failed: bbox re-classification filters, per-class models.
- SIIM-COVID 1st: YOLOv5x6 @768, EffDet-D7, Faster R-CNN; WBF; 8× TTA; repeated pseudo-labelling of test.
- Global Wheat 2021 1st: single YOLOv5 @800, out-of-domain validation fold, TTA, pseudo-labelling, WBF.
- Great Barrier Reef 1st: 6 YOLOv5, second-stage crop classifier CV 0.716→0.73.
- HuBMAP 2023 1st: RTMDet-x, rotation ±180°, scale 0.1–2.0, EMA "most crucial", WBF of 5 models.
- RSNA Breast 1st: YOLOX-nano ROI detector → crop → ConvNeXt classifier.
- WBF paper: NMS 0.544 → WBF 0.567; IoU 0.55 near-optimal. YOLOv5 TTA: 0.504→0.516.

## 3. mAP mechanics (what matters for this leaderboard)

- Standard COCO mAP ranks by score, so ultralytics val uses conf=0.001, max_det=300.
- Negative images: COCO-FP shows adding no-object images drops AP50 by 6.6–7.8 pts for YOLOv9-E/YOLOv10-X/RT-DETR; a score threshold of 0.3 raised YOLOv9-E's COCO-FP mAP 50.2→54.5. With 53 % negatives here, an image-level lesion/no-lesion gate is worth a lot.
- If the scorer ignores confidence: AP per class ≈ recall × envelope-precision at one operating point → choose per-class thresholds on val by simulating the exact metric; cap boxes per image; class-agnostic NMS; one label per box; write rows in descending confidence.

## 4. Domain shift: phone photos of the screen

- CheXphoto / CheXpedition / Kuo 2021: phone photos of X-ray screens cost 0.04–0.17 AUROC; training with simulated photo artifacts (glare, moiré, tilt, brightness/contrast, blur, JPEG, Poisson noise) recovered 53–85 % of the loss and beat fine-tuning on 175 real photos.
- Breast US via smartphone photo of the monitor (PMC10759682): YOLOv3 @416 worked; lights off to avoid reflections → expect glare in the wild. Moiré alone can degrade detectors up to 25 %.
- Practical: 1,348 real mobile images with labels exist — oversample, evaluate the `mobile` val slice separately, add Perspective / gamma / ISONoise / JPEG / MotionBlur / moiré / glare augmentation; consider a screen-ROI crop.

## 5. Ranked recommendations (expected gain, highest first)

1. Replicate the scorer. Local evaluator on val in the submission format (no scores). Diagnostic submissions: per-image caps, sorted vs shuffled rows, threshold sweep.
2. Image-level gate for negatives (53 % of data).
3. Strong single-model baseline: YOLO11/26 m–l at imgsz 1024, 100–150 epochs, mosaic, fliplr 0.5, flipud 0, scale 0.5, degrees ≤10, class-agnostic NMS, EMA.
4. Second-stage crop classifier for the 7 classes (context-padded crops) to relabel/rerank detector boxes.
5. Ensemble + WBF (IoU 0.55) + TTA; include one DETR (RF-DETR) for precision/diversity.
6. Mobile robustness: oversample mobile 3–5×, photo-artifact augmentation, separate mobile val metric.
7. Hard-negative mining on high-scoring FPs.
8. Pseudo-label the test images (high-confidence only), retrain 1–2 rounds.
9. Skip: speckle-noise injection, curriculum zoom, zero-shot Grounding DINO, per-class detectors.

## Sources

- Kaggle: https://www.kaggle.com/competitions/liver-ultrasound-detection
- Tiyarattanachai 2021: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252882 · 2022: https://pmc.ncbi.nlm.nih.gov/articles/PMC9095624/
- Two-stage HCC (J Assoc Med Sci): https://he01.tci-thaijo.org/index.php/bulletinAMS/article/view/264242
- Radiology AI DETR vs Faster R-CNN: https://pmc.ncbi.nlm.nih.gov/articles/PMC9152842/ · npj Digit Med 2026: https://pmc.ncbi.nlm.nih.gov/articles/PMC13260369/ · Surg Endosc 2026: https://link.springer.com/article/10.1007/s00464-026-13050-7 · JHC YOLOR: https://www.dovepress.com/automatic-real-time-detection-and-diagnosis-of-liver-tumor-with-ultras-peer-reviewed-fulltext-article-JHC
- SMC-LUD dataset: https://www.nature.com/articles/s41597-026-07023-7 · Kaggle liver US: https://www.kaggle.com/datasets/orvile/annotated-ultrasound-liver-images-dataset
- Augmentation: https://arxiv.org/html/2501.13193v1 · UltraAugment CVPRW 2024 · https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2026.1878351/full · https://pmc.ncbi.nlm.nih.gov/articles/PMC10325648/
- Competitions: VinBigData 1st write-up · https://github.com/dungnb1333/SIIM-COVID19-Detection · https://github.com/ksnxr/GWC_solution · HuBMAP 2023 1st (tascj) · https://github.com/dangnh0611/kaggle_rsna_breast_cancer · WBF https://ar5iv.labs.arxiv.org/html/1910.13302 · https://arxiv.org/abs/2309.05652
- mAP: COCO-FP https://arxiv.org/html/2409.07907 · pycocotools cocoeval.py · https://github.com/ultralytics/ultralytics/issues/24062
- Phone photos: CheXphoto https://ar5iv.labs.arxiv.org/html/2007.06199 · CheXpedition https://ar5iv.labs.arxiv.org/html/2002.11379 · https://pmc.ncbi.nlm.nih.gov/articles/PMC7884693/ · https://pmc.ncbi.nlm.nih.gov/articles/PMC10759682/
- Frameworks: https://docs.ultralytics.com/models/yolo26/ · https://docs.ultralytics.com/guides/yolo26-training-recipe/ · https://github.com/roboflow/rf-detr · https://github.com/Peterande/D-FINE · https://github.com/Intellindust-AI-Lab/DEIMv2 · https://github.com/RT-DETRs/RT-DETRv4
