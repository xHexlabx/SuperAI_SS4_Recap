# 🌏 Forest Type Classification

<p>
 <b>SuperAI Season 4 | Level 2 | Hackathon 7</b>
</p>

<p>
 ใน Hackathon นี้ นับเป็น Hackathon ที่สามารถทำได้ง่ายที่สุดเพราะออกแบบมาเพื่อให้เป็น Individual Hackathon โดยในงานนี้เราจะได้รับบทเป็นนักวิเคราะห์เซนเซอร์จากดาวเทียม เพื่อวิเคราะห์ว่าจากข้อมูลที่ได้รับมานั้น เป็นข้อมูลของป่าเบญจพรรณ (MDF) ป่าเต็งรัง (DDF) หรือป่าดิบแล้ง (DEF) เป็นโจทย์ <b>Tabular Data Classification</b> นั่นเอง โดยสามารถติดตามวิธีการได้ใน Notebook ด้านล่างนี้เลยครับ
</p>

> **Task** : Tabular Classification &nbsp;|&nbsp; **Tools** : AutoGluon , CatBoost , XGBoost , SMOTE (imblearn) , PyTorch Lightning

<hr>

## 📓 ไฟล์ในโฟลเดอร์นี้

- [`Forest_Type_Classification.ipynb`](./Forest_Type_Classification.ipynb) — Baseline : Feature Engineer + SMOTE → CatBoost / XGBoost / AutoGluon แยก 3 คลาสรวดเดียว
- [`Forest_Type_Double_Classifier.ipynb`](./Forest_Type_Double_Classifier.ipynb) — แตกเป็น 2 ชั้น : DEF vs NON-DEF ก่อน แล้วค่อยแยก MDF vs DDF
- [`Forest_Type_Double_Classifier_MLP.ipynb`](./Forest_Type_Double_Classifier_MLP.ipynb) — เวอร์ชัน MLP ด้วย PyTorch Lightning ของแนวคิด Double Classifier

<hr>

<p align="center">
 <a href="../../README.md">⬅️ กลับไปหน้าหลัก</a>
</p>
