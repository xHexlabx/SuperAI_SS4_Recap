# Brain Motor Imagery — one entry point per step.
# ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data eda train predict lab clean

help:
	@echo "make setup    - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data     - โหลด dataset จาก Kaggle ลง datasets/ (gitignored)"
	@echo "make eda      - รัน notebooks/01_eda.ipynb ใหม่ (สร้างรูปลง ../../assets)"
	@echo "make train    - เทรนโมเดล  เช่น: make train SET='--set model.name=eegnet'"
	@echo "make predict  - สร้าง submissions/submission.csv"
	@echo "make lab      - เปิด JupyterLab สำหรับ notebooks/"

setup:
	uv sync --extra notebook

data:
	uv run kaggle competitions download -c brain-motor-imagery-classification -p datasets
	unzip -oq datasets/brain-motor-imagery-classification.zip -d datasets

eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train:
	uv run mi-train --config $(CONFIG) $(SET)

predict:
	uv run mi-predict --config $(CONFIG) $(SET)

lab:
	uv run jupyter lab notebooks/

clean:
	rm -rf models submissions .venv
