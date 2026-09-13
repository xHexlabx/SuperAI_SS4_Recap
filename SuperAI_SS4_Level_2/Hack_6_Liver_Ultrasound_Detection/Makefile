# Liver Ultrasound Detection — one entry point per step. ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data prepare eda train predict submit lab clean

help:
	@echo "make setup    - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data     - โหลด zip จาก Kaggle แล้ว stream-extract ลง datasets/raw (ย่อรูปมือถือให้ด้านยาว 2048)"
	@echo "make prepare  - สร้าง datasets/index.csv และ symlink layout สำหรับ ultralytics"
	@echo "make eda      - รัน notebooks/01_eda.ipynb ใหม่"
	@echo "make train    - เทรนโมเดล  เช่น: make train SET='--set model.weights=yolo26l.pt'"
	@echo "make predict  - ทำนาย val+test, จูน threshold, เขียน submissions/submission.csv"
	@echo "make submit   - ส่ง submissions/submission.csv ขึ้น Kaggle (MSG='...')"

setup:
	uv sync --extra notebook

data:
	uv run liver-download

prepare:
	uv run liver-prepare --config $(CONFIG) $(SET)

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train:
	uv run liver-train --config $(CONFIG) $(SET)

predict:
	uv run liver-predict --config $(CONFIG) $(SET)

MSG ?= liver_us
submit:
	uv run kaggle competitions submit -c liver-ultrasound-detection -f submissions/submission.csv -m "$(MSG)"

lab:
	uv run jupyter lab notebooks/

clean:
	rm -rf models submissions .venv
