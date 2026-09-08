# 😴 Sleep Stages Classification

<p>
 <b>SuperAI Season 4 | Individual Hackathon 5</b>
</p>

<p>
 ใน Hackathon นี้ จะเป็นการ Classified <b>ระยะการนอนหลับ</b> จากสัญญาณชีวภาพ โดยจะต้องซอยสัญญาณออกเป็นช่วงละ <b>30 วินาที (1920 rows)</b> แล้วทำ Feature Engineering ด้วย <u>FFT</u> เพื่อดึงข้อมูลด้านความถี่ออกมาก่อนโยนเข้าโมเดล นับเป็นงานที่ได้ใช้ทั้งความรู้ด้าน Signal Processing และ Machine Learning โดยสามารถติดตามวิธีการได้ใน Notebook ด้านล่างนี้เลยครับ
</p>

> **Task** : Signal Classification &nbsp;|&nbsp; **Tools** : numpy (FFT) , AutoGluon , scikit-learn , seaborn

<hr>

## 📓 ไฟล์ในโฟลเดอร์นี้

- [`Sleep_Stages_Classification.ipynb`](./Sleep_Stages_Classification.ipynb) — Plot สัญญาณ + FFT ดูย่านความถี่ → ซอย 30 วินาที → Feature Engineering → AutoGluon

<hr>

<p align="center">
 <a href="../../README.md">⬅️ กลับไปหน้าหลัก</a>
</p>
