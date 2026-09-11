# SuperAI SS4 Recap 🤖

<p>
 Repository นี้ จัดทำขึ้นมาเพื่อรวบรวม Code และแชร์เทคนิคต่าง ๆ ที่ได้รับจากค่าย SuperAI หวังว่าจะเป็นประโยชน์แก่ผู้เข้าชมทุกท่านให้ได้รับความรู้กลับไป<br>ไม่มากก็น้อย หากมีข้อผิดพลาดตัวผมขออภัยมา ณ ที่นี้ด้วยนะครับ (จาก HexTex Optimizer SS4 🔥🔥🔥)
</p>

<hr>

## ⭐ Special Thanks ⭐ 
 <p>
  <ul>
   <li>N'Jane 🐶 : https://github.com/jjentaa</li>
   <li>N'PP 🦆 : https://github.com/Makufff</li>
  </ul>
</p>

<hr>

## 🗺️ สารบัญ (Table of Contents)

<p>
 โดยใน Repository นี้ จะประกอบไปด้วย Code Hackathon จาก Level 1 , Level 2 และ Individual Hackathon แบ่งไปตามระดับนะครับ
</p>

### 🥇 Level 1

| # | Hackathon | Task | Tools |
|:-:|---|---|---|
| 0 | 🍰 [Pre-Hackathon Diabetes Prediction](./SuperAI_SS4_Level_1/Pre_Diabetes_Prediction) | Tabular Classification | AutoGluon |
| 1 | 🔎 [Image Search](./SuperAI_SS4_Level_1/Hack_1_Image_Search) | Zero-shot Retrieval (open-set) | SigLIP 2 · transformers · PyTorch |
| 2 | 🗃️ [ThaiJO Researcher](./SuperAI_SS4_Level_1/Hack_2_Thaijo_Researcher) | Data Cleaning | pandas · regex |
| 3 | 🔊 [UWB Pose Prediction](./SuperAI_SS4_Level_1/Hack_3_Ultrawideband_Pose_Prediction) | Signal Classification | timm (MaxViT) · PyTorch |
| 4 | 💧 [Soil Water Prediction](./SuperAI_SS4_Level_1/Hack_4_Soil_Water_Prediction) | Tabular Regression | AutoGluon |
| 5 | 📔 [Nithan Chadok NER](./SuperAI_SS4_Level_1/Hack_5_NithanChadok_NER) | Named Entity Recognition | PyThaiNLP · Spark NLP |
| 6 | 📷 [OCR 🚧](./SuperAI_SS4_Level_1/Hack_6_OCR) | Optical Character Recognition | — |

### 🥈 Level 2

| # | Hackathon | Task | Tools |
|:-:|---|---|---|
| 1 | 💬 [Image Captioning](./SuperAI_SS4_Level_2/Hack_1_Image_Captioning) | Vision-Language | BLIP-2 · transformers |
| 2 | 📄 [Table Question Answering](./SuperAI_SS4_Level_2/Hack_2_Table_Question_Answering) | LLM Agent | LangChain · GPT-4o |
| 3 | ⚖️ [Legal Act Classification](./SuperAI_SS4_Level_2/Hack_3_Legal_Act_Classification) | Rule Reasoning (LLM) | Qwen3.8 · OpenRouter · rule engine |
| 4 | 🧠 [Brain Motor Imagery](./SuperAI_SS4_Level_2/Hack_4_Brain_Motor_Imagery) | EEG Signal Classification (cross-subject) | MNE · pyRiemann · braindecode |
| 5 | 💸 [Home Credit Risk 🚧](./SuperAI_SS4_Level_2/Hack_5_Home_Credit_Risk) | Tabular Classification | — |
| 6 | 🏥 [Liver Ultrasound Detection](./SuperAI_SS4_Level_2/Hack_6_Liver_Ultrasound_Detection) | Medical Image Classification | Lightning ⚡ · MaxViT |
| 7 | 🌏 [Forest Type Classification](./SuperAI_SS4_Level_2/Hack_7_Forest_Type_Classification) | Tabular Classification | AutoGluon · CatBoost · XGBoost |

### 🎯 Individual Hackathon

| # | Hackathon | Task | Tools |
|:-:|---|---|---|
| 1 | 🪪 [License Plate Recognition 🚧](./SuperAI_SS4_Individual_Hackathon/Hack_1_License_Plate_Recognition) | OCR / Detection | — |
| 2 | 📔 [Nithan Chadok Hybrid OCR-NER 🚧](./SuperAI_SS4_Individual_Hackathon/Hack_2_Nithan_Chadok_Hybrid_OCR-NER) | OCR + NER | — |
| 3 | 🚗 [Car Prediction](./SuperAI_SS4_Individual_Hackathon/Hack_3_Car_Prediction) | Tabular Classification | AutoGluon |
| 4 | ♨️ [Temperature Prediction](./SuperAI_SS4_Individual_Hackathon/Hack_4_Temperature_Prediction) | Tabular Regression | CatBoost · XGBoost · AutoGluon |
| 5 | 😴 [Sleep Stages Classification](./SuperAI_SS4_Individual_Hackathon/Hack_5_Sleep_Stages_Classification) | Signal Classification | FFT · AutoGluon |

<p>
 <i>หมายเหตุ : 🚧 คือ Notebook ที่ยังเป็น Placeholder อยู่ครับ เดี๋ยวจะทยอยเติม Code ให้ครบนะครับ 🙏</i>
</p>

<hr>

### SuperAI Season 4 | Level 1 

<p>
 โดยในรอบนี้ จะทำการคัดผู้เข้าร่วมโครงการจากการให้ทำ Hackathon ทั้งหมด 6 ครั้ง (มี Pre-Hackathon 1 ครั้งด้วย) ประกอบไปด้วย 
 <ul>
  <li>1.Pre-Hackathon Diabetes Prediction</li> 
  <li>2.Image Search</li>
  <li>3.Thaijo Researcher</li>
  <li>4.UWB Pose Prediction</li>
  <li>5.Soil Water Prediction</li>
  <li>6.Nithan Chadok NER</li>
  <li>7.OCR</li>
 </ul>
</p>

#### Pre-Hackathon Diabetes Prediction
<p>
 🍰 ใน Hackathon นี้ ไม่มีอะไรเป็นพิเศษมาก เนื่องจากเป็น Hackathon ในคาบทดลอง Kaggle เพื่อเตรียมพร้อมสำหรับการทำ Hackathon ต่อ ๆ ไป โดยในโจทย์นี้มีจุดประสงค์เพื่อให้ทำ Machine Learning เพื่อแก้ปัญหาในการคัดแยกผู้ป่วยที่มีความเสี่ยงเป็นโรคเบาหวาน (Classification) โดยสามารถติดตามวิธีการได้ใน Repository นี้ 
</p>

> **Task** : Tabular Classification &nbsp;|&nbsp; **Tools** : AutoGluon , pandas

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Pre_Diabetes_Prediction

#### Image Search

<p>
 🔎 ใน Hackathon นี้ เป็น Hackathon เรื่อง <b>Image Search</b> หรือให้อธิบายง่าย ๆ ก็คือ มีข้อมูลที่เป็น <b>รูปภาพ</b> เข้ามาแล้วให้ทายว่ารูปภาพที่เข้ามานั้น <b>คล้ายกับ</b> รูปภาพใดใน folder queries ที่มีให้ คล้าย ๆ งาน Classification แต่เราจะใช้วิธีการที่เรียกว่า <b>Similarity Search</b> ในการตอบคำถามนี้<br>
 รอบนี้เขียนใหม่เป็น <b>Python package</b> ที่ benchmark encoder แบบ zero-shot 16 ตัวบน validation ที่สร้างเองจาก <code>train/</code> ประเด็นหลักคือ <b>คอขวดไม่ได้อยู่ที่ backbone แต่อยู่ที่การตัดสินว่า "ไม่ตรงสักอัน"</b> — public 0.936 → <b>0.961</b> โดยสามารถติดตามวิธีการได้ใน Repository นี้
 </p>

> **Task** : Zero-shot Logo Retrieval (open-set) &nbsp;|&nbsp; **Tools** : SigLIP 2 (`google/siglip2-base-patch16-384`) , transformers , open_clip , PyTorch

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_1_Image_Search

#### Thaijo Researcher
<p>
 🗃️ ใน Hackathon นี้ นับเป็น Hackathon ที่ <b>เหนื่อยที่สุด</b> เลยก็เป็นได้ เนื่องจาก Hackathon นี้ จัดทำขึ้นมาเพื่อให้ <b>Clean Dataset!!!</b> คุณอ่านไม่ผิดครับ ใช่ครับ คุณจะได้เทคนิคในการ clean data จาก Thaijo แบบ <b>จะเป็นจะตาย</b> รับรองความสนุกเป็นทุกข์ไปด้วยกัน โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Data Cleaning / Text Preprocessing &nbsp;|&nbsp; **Tools** : pandas , regex

 https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_2_Thaijo_Researcher

#### UWB Pose Prediction
<p>
 🔊 ใน Hackathon นี้ จะเป็นการทำ <b>Signal Classification</b> จาก สัญญาณ <b>Ultra-Wide-Band</b> โดยจะให้ทำการ Classified ท่า Pose ต่าง ๆ จากตัวสัญญาณ และ มีเทคนิคในการทำ <u>signal to image</u> เข้ามาเพิ่มเติมด้วย โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Signal Classification (signal → image) &nbsp;|&nbsp; **Tools** : timm (`maxvit_base_tf_384`) , PyTorch , torchvision

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_3_Ultrawideband_Pose_Prediction

#### Soil Water Prediction
<p>
 💧 ใน Hackathon นี้ จะเป็นการทำ <b>Regression</b> ความชื้นในดินด้วยข้อมูล tabular ที่ได้รับมาในวันอื่น ๆ เป็น Hackathon ที่เน้นในการทำ Machine Learning และการวิเคราะห์ features โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Tabular Regression &nbsp;|&nbsp; **Tools** : AutoGluon , pandas

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_4_Soil_Water_Prediction

#### Nithan Chadok NER
<p>
 📔 ใน Hackathon นี้ จะเป็นการทำ Named-Entity-Recognition ใน <b>นิทานชาดก</b> โดยความยากคือเราจะได้ dataset <b>สุดมหัศจรรย์</b> โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Named Entity Recognition (Token Classification) &nbsp;|&nbsp; **Tools** : PyThaiNLP , transformers , Spark NLP , seqeval

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_5_NithanChadok_NER

#### OCR
<p>
 📷 ใน Hackathon นี้ จะเป็นการทำ OCR โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Optical Character Recognition &nbsp;|&nbsp; **Tools** : 🚧 กำลังจัดทำ

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_1/Hack_6_OCR

<hr>

### SuperAI Season 4 | Level 2 

<p>
 โดยในรอบนี้ จะทำการคัดผู้เข้าร่วมโครงการจากการให้ทำ Hackathon ทั้งหมด 7 ครั้ง ประกอบไปด้วย 
 <ul>
  <li>1.Image Captioning</li> 
  <li>2.Table Question Answering</li>
  <li>3.Legal Act Classification</li>
  <li>4.Brain Motor Imagery</li>
  <li>5.Home Credit Risk</li>
  <li>6.Liver Ultrasound Detection</li>
  <li>7.Forest Type Classification</li>
 </ul>
</p>

#### Image Captioning

<p>
 💬 ใน Hackathon นี้ นับเป็น Hackathon แรกสุดจากทางโครงการ โดยเป็นการแข่งแบบ On-Site โดยเป็นการแข่งบรรยายรูปภาพเป็นข้อความจาก Coco dataset และ IPU-24 ของทางโครงการ โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Image Captioning (Vision-Language) &nbsp;|&nbsp; **Tools** : BLIP-2 (`Salesforce/blip2-opt-2.7b`) , transformers , PyTorch

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_1_Image_Captioning

#### Table Question Answering

<p>
 📄 ใน Hackathon นี้ นับเป็น Hackathon ที่กิน Resource มากที่สุด เพราะต้องทำงาน Table-Question-Answering บน Data ขนาดใหญ่ และยังเป็นสัปดาห์ที่มีโอกาสได้ใช้ในตัวของ Lanta อีกด้วย โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Table Question Answering (LLM Agent) &nbsp;|&nbsp; **Tools** : LangChain , GPT-4o , pandas

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_2_Table_Question_Answering

#### Legal Act Classification

<p>
  ⚖️ ใน Hackathon นี้ นับเป็น Hackathon ที่ต้องมีการพึ่งพา Domain Expert สูงมาก เพราะเป็นโมเดลทางด้านกฎหมาย โดยในสัปดาห์นี้เราจะต้องให้โมเดลตอบคำถามทางกฎหมายว่า หากมีเงื่อนไขตามโจทย์ การทำสัญญาเช่นนี้ ๆ สามารถทำได้หรือไม่ ? <br> โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Rule Reasoning (text → boolean) &nbsp;|&nbsp; **Tools** : Qwen3.8 , OpenRouter , bipartite matching

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_3_Legal_Act_Classification

#### Brain Motor Imagery

<p>
 🧠 ใน Hackathon นี้ นับเป็น Hackathon ที่ยากลำบากที่สุด เนื่องจาก Data ที่ได้รับมาใช้งานยากมาก มี Noise และมีรูปแบบที่มองยาก นับเป็น Hackathon ที่มีความยากด้าน Technical มากที่สุด โดยในงานนี้เราได้รับบทเป็นคุณหมอที่จะต้อง Classified ว่าหากมีสัญญาณสมองที่ได้รับมาตามนี้ ๆ ผู้ทดสอบกำลังคิดว่าเคลื่อนที่ไปทาง ซ้าย - ขวา หรือ Resting โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : EEG Signal Classification (3 คลาส, cross-subject) &nbsp;|&nbsp; **Tools** : MNE , pyRiemann , braindecode , PyTorch<br>
> **รอบเดิม** : `hack4_preprocess.ipynb` + `hack4_visualize-2.ipynb` (Colab, EEGNet) ได้ 0.315 บน leaderboard ขณะ baseline ผู้จัด 0.565<br>
> **รอบใหม่ (2026)** : rebuild เป็น Python package (MNE · pyRiemann · braindecode) — EDA พบว่า test คือคนใหม่ 2 คน และเครื่องคือ Unicorn Hybrid Black 8 ช่อง · Riemannian covariance + alignment ต่อ session ได้ **0.638 public / 0.607 private** ด้วย EEG ล้วน ชนะ baseline ผู้จัด 0.565 และทุกทีมในรอบเดิม · รายละเอียด ablation 56 แบบใน README ของโฟลเดอร์

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_4_Brain_Motor_Imagery

#### Home Credit Risk

<p>
 💸 ใน Hackathon นี้ นับเป็น Hackathon ที่มี Data ขนาดใหญ่ เป็นงานที่จะต้องมองความเสี่ยงในการอนุมัติสินเชื่อ โดยให้ Classified ว่าบุคคลคนที่สามารถจ่ายหนี้ได้หรือไม่ ? โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Tabular Classification &nbsp;|&nbsp; **Tools** : 🚧 กำลังจัดทำ

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_5_Home_Credit_Risk

#### Liver Ultrasound Detection

<p>
 🏥 ใน Hackathon นี้ นับเป็น Hackathon ที่ต้องทำงานร่วมกับ Medical Image โดยทำการ Detection เนื้องอกชนิดต่าง ๆ ในตับ รวมถึง <b>มะเร็งตับ</b> อีกด้วย โดยใน Hackathon จำเป็นต้องทำการ Augment Data ค่อนข้างมากเพราะจำเป็นต้องต่อยอดผลลัพธ์เพื่อให้คุณหมอสามารถใช้งานโมเดลผ่านรูปถ่าย Ultrasound จากโทรศัพท์ได้ โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Medical Image Detection / Classification &nbsp;|&nbsp; **Tools** : PyTorch Lightning ⚡ , timm (`maxvit_tiny_tf_224`) , torchmetrics

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_6_Liver_Ultrasound_Detection


#### Forest Type Classification

<p>
 🌏 ใน Hackathon นี้ นับเป็น Hackathon ที่สามารถทำได้ง่ายที่สุดเพราะออกแบบมาเพื่อให้เป็น Individual Hackathon โดยในงานนี้เราจะได้รับบทเป็นนักวิเคราะห์เซนเซอร์จากดาวเทียม เพื่อวิเคราะห์ว่าจากข้อมูลที่ได้รับมานั้น เป็นข้อมูลของป่าเบญจพรรณ (MDF) ป่าเต็งรัง (DDF) หรือป่าดิบแล้ง (DEF) เป็นโจทย์ <b>Tabular Data Classification</b> นั่นเอง โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Tabular Classification &nbsp;|&nbsp; **Tools** : AutoGluon , CatBoost , XGBoost , SMOTE (imblearn) , PyTorch Lightning<br>
> **ไฟล์ในโฟลเดอร์** : `Forest_Type_Classification.ipynb` (Baseline 3 คลาสรวด) , `Forest_Type_Double_Classifier.ipynb` (แตกเป็น 2 ชั้น DEF/NON-DEF → MDF/DDF) , `Forest_Type_Double_Classifier_MLP.ipynb` (เวอร์ชัน MLP)

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Level_2/Hack_7_Forest_Type_Classification

<hr>

### SuperAI Season 4 | Individual Hackathon

<p>
 โดยในรอบนี้ จะทำการคัดผู้เข้าร่วมโครงการจากการให้ทำ Hackathon ทั้งหมด 5 ครั้ง ประกอบไปด้วย 
 <ul>
  <li>1.License Plate Recognition</li> 
  <li>2.Nithan Chadok Hybrid OCR-NER</li>
  <li>3.Car Prediction</li>
  <li>4.Temperature Prediction</li>
  <li>5.Sleep Stages Classification</li>
 </ul>
</p>

#### License Plate Recognition
<p>
 🪪 ใน Hackathon นี้ จะเป็นการอ่านป้ายทะเบียนรถจากรูปภาพ โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : License Plate Recognition &nbsp;|&nbsp; **Tools** : 🚧 กำลังจัดทำ

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Individual_Hackathon/Hack_1_License_Plate_Recognition

#### Nithan Chadok Hybrid OCR-NER
<p>
 📔 ใน Hackathon นี้ จะเป็นการนำงาน OCR และ NER มารวมร่างกันเป็น Pipeline เดียว คือ อ่านตัวอักษรจาก <b>นิทานชาดก</b> ออกมาก่อน แล้วค่อยหา Entity ต่อ โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : OCR + Named Entity Recognition &nbsp;|&nbsp; **Tools** : 🚧 กำลังจัดทำ

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Individual_Hackathon/Hack_2_Nithan_Chadok_Hybrid_OCR-NER

#### Car Prediction
<p>
 🚗 ใน Hackathon นี้ จะเป็นการทำนายผลจากข้อมูล <b>แบบสอบถาม</b> (Questionnaire) ที่มีทั้งอาชีพ พฤติกรรมการใช้ Social Media ต่าง ๆ ความสนุกจะอยู่ที่การจัดการ Missing Value ที่ถูกเข้ารหัสมาเป็น <b>888 / 999</b> และการตัด features ที่ไม่จำเป็นทิ้ง โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Tabular Classification &nbsp;|&nbsp; **Tools** : AutoGluon , pandas

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Individual_Hackathon/Hack_3_Car_Prediction

#### Temperature Prediction
<p>
 ♨️ ใน Hackathon นี้ จะเป็นการทำ <b>Regression</b> ทำนายอุณหภูมิจากข้อมูลสถานีตรวจวัดอากาศ โดยจะได้เล่นกับการทำ Feature Engineering จาก <u>เวลา</u> และการตัด column ที่เป็นรหัสพื้นที่ต่าง ๆ ทิ้ง แล้วเทียบผลระหว่าง Gradient Boosting กับ AutoGluon โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Tabular Regression &nbsp;|&nbsp; **Tools** : CatBoost , XGBoost , AutoGluon , scikit-learn

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Individual_Hackathon/Hack_4_Temperature_Prediction

#### Sleep Stages Classification
<p>
 😴 ใน Hackathon นี้ จะเป็นการ Classified <b>ระยะการนอนหลับ</b> จากสัญญาณชีวภาพ โดยจะต้องซอยสัญญาณออกเป็นช่วงละ <b>30 วินาที (1920 rows)</b> แล้วทำ Feature Engineering ด้วย <u>FFT</u> เพื่อดึงข้อมูลด้านความถี่ออกมาก่อนโยนเข้าโมเดล นับเป็นงานที่ได้ใช้ทั้งความรู้ด้าน Signal Processing และ Machine Learning โดยสามารถติดตามวิธีการได้ใน Repository นี้
</p>

> **Task** : Signal Classification &nbsp;|&nbsp; **Tools** : numpy (FFT) , AutoGluon , scikit-learn , seaborn

https://github.com/xHexlabx/SuperAI_SS4_Recap/tree/main/SuperAI_SS4_Individual_Hackathon/Hack_5_Sleep_Stages_Classification

<hr>

### 📁 โครงสร้างของ Repository

```
SuperAI_SS4_Recap
├── assets                           # รูป banner ของแต่ละ Hackathon
├── SuperAI_SS4_Level_1              # Hackathon รอบคัดเลือก Level 1 (+ Pre-Hackathon)
├── SuperAI_SS4_Level_2              # Hackathon รอบคัดเลือก Level 2
└── SuperAI_SS4_Individual_Hackathon # Individual Hackathon
```

<p>
 โดยในแต่ละ Hackathon จะมี <code>README.md</code> (รูปโจทย์ + สรุปโจทย์ + รายละเอียดไฟล์) และ Notebook ของงานนั้น ๆ อยู่ข้างในนะครับ
</p>

<hr>

<p align="center">
 ⭐ ถ้า Repository นี้มีประโยชน์ ฝากกด Star เป็นกำลังใจให้กันด้วยนะครับ ⭐
</p>
