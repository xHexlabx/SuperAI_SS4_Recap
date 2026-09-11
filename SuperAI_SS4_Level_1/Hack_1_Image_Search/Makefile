# Image Search — one entry point per step.
# ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data embed map benchmark predict all lab clean-cache clean

help:
	@echo "make setup      - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data       - โหลด dataset จาก Kaggle ลง datasets/ (gitignored)"
	@echo "make embed      - สกัด embedding ของทุกภาพแล้ว cache ไว้"
	@echo "make map        - จับคู่โลโก้ 22 รูป กับโฟลเดอร์แบรนด์ใน train/ -> configs/query_map.yaml"
	@echo "make benchmark  - วัดทุก encoder บน validation ที่สร้างจาก train/"
	@echo "make predict    - สร้าง submissions/submission.csv"
	@echo "make all        - data -> map -> benchmark -> predict"
	@echo "make lab        - เปิด JupyterLab สำหรับ notebooks/ (EDA เท่านั้น)"
	@echo ""
	@echo "ตัวอย่าง override:  make predict SET=\"--set embed.models=[siglip2-so400m,dinov2-l]\""

setup:
	uv sync --extra notebook

data:
	uv run img-download --config $(CONFIG) $(SET)

embed:
	uv run img-embed --config $(CONFIG) $(SET)

map:
	uv run img-map --config $(CONFIG) $(SET)

benchmark:
	uv run img-benchmark --config $(CONFIG) $(SET)

predict:
	uv run img-predict --config $(CONFIG) $(SET)

all: data map benchmark predict

lab:
	uv run jupyter lab notebooks/

smoke:
	uv run python scripts/smoke_test.py

clean-cache:
	rm -rf datasets/emb

clean: clean-cache
	rm -rf models submissions .venv
