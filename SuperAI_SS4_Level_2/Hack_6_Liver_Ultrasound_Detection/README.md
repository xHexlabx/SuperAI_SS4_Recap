# 🏥 Liver Ultrasound Detection

<p>
 <b>SuperAI Season 4 | Level 2 | Hackathon 6</b>
</p>

<p>
 ใน Hackathon นี้ นับเป็น Hackathon ที่ต้องทำงานร่วมกับ Medical Image โดยทำการ Detection เนื้องอกชนิดต่าง ๆ ในตับ รวมถึง <b>มะเร็งตับ</b> อีกด้วย โดยใน Hackathon จำเป็นต้องทำการ Augment Data ค่อนข้างมากเพราะจำเป็นต้องต่อยอดผลลัพธ์เพื่อให้คุณหมอสามารถใช้งานโมเดลผ่านรูปถ่าย Ultrasound จากโทรศัพท์ได้ โดยสามารถติดตามวิธีการได้ใน Notebook ด้านล่างนี้เลยครับ
</p>

> **Task** : Medical Image Detection / Classification &nbsp;|&nbsp; **Tools** : PyTorch Lightning ⚡ , timm (`maxvit_tiny_tf_224`) , torchmetrics

<hr>

## 📓 ไฟล์ในโฟลเดอร์นี้

- [`Liver_Ultrasound_Detection_Classification_Lightning.ipynb`](./Liver_Ultrasound_Detection_Classification_Lightning.ipynb) — สร้าง Annotation file → Dataloader แยก lesion (1) / non-lesion (0) → เทรน MaxViT ด้วย Torch Lightning ⚡

<hr>

<p align="center">
 <a href="../../README.md">⬅️ กลับไปหน้าหลัก</a>
</p>
