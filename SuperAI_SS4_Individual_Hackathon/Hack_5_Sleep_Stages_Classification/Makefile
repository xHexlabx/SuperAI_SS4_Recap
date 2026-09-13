# Sleep Stages Classification — one entry point per step.
# ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data cache restore eda train predict submit lab clean

help:
	@echo "make setup    - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data     - โหลด dataset จาก Kaggle ลง datasets/ (gitignored)"
	@echo "make cache    - สกัด feature ต่อคืน ลง datasets/cache/ (ทำครั้งเดียว ~5 นาที)"
	@echo "make restore  - คืน probability ของ run ที่ archive ไว้ใน results/saved_runs/ กลับเข้า models/"
	@echo "make eda      - รัน notebooks/01_eda.ipynb ใหม่ (สร้างรูปลง ../../assets)"
	@echo "make train    - เทรน + CV  เช่น: make train SET='--set model.name=lgbm'"
	@echo "make predict  - สร้าง submissions/submission.csv"
	@echo "make submit   - ส่งไฟล์ขึ้น Kaggle"
	@echo "make lab      - เปิด JupyterLab สำหรับ notebooks/"

setup:
	uv sync --extra notebook

data:
	uv run sleep-download --config $(CONFIG)

cache:
	uv run sleep-cache --config $(CONFIG) $(SET)

restore:
	mkdir -p models
	cp -r results/saved_runs/* models/


eda:
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

train:
	uv run sleep-train --config $(CONFIG) $(SET)

predict:
	uv run sleep-predict --config $(CONFIG) $(SET)

submit:
	uv run kaggle competitions submit -c signal-processing-sleep-staging-classification \
	  -f submissions/submission.csv -m "$(MSG)"

lab:
	uv run jupyter lab notebooks/

clean:
	rm -rf models submissions .venv
