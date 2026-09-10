# UWB Pose Prediction — one entry point per step.
# ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data train predict clean-cache clean lab

help:
	@echo "make setup    - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data     - โหลด dataset จาก Kaggle ลง datasets/ (gitignored)"
	@echo "make train    - เทรนโมเดล  เช่น: make train SET='--set train.epochs=5'"
	@echo "make predict  - สร้าง submissions/submission.csv"
	@echo "make lab      - เปิด JupyterLab สำหรับ notebooks/ (EDA เท่านั้น)"
	@echo "make clean-cache - ลบ cache ภาพที่ encode ไว้"

setup:
	uv sync --extra notebook

data:
	uv run uwb-download --config $(CONFIG) $(SET)

train:
	uv run uwb-train --config $(CONFIG) $(SET)

predict:
	uv run uwb-predict --config $(CONFIG) $(SET)

lab:
	uv run jupyter lab notebooks/

clean-cache:
	rm -rf datasets/cache

clean: clean-cache
	rm -rf models submissions .venv
