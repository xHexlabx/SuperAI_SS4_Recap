# Legal Act Classification — one entry point per step.
# ทุกคำสั่งรันผ่าน uv จึงไม่ต้อง activate venv เอง

CONFIG ?= configs/default.yaml
SET    ?=

.PHONY: help setup data serve compile eval predict clean-cache clean lab

help:
	@echo "make setup    - สร้าง .venv และติดตั้ง dependencies (uv sync)"
	@echo "make data     - โหลด dataset จาก Kaggle ลง datasets/ (gitignored)"
	@echo "make serve    - เปิด Qwen เป็น OpenAI-compatible server ด้วย vLLM"
	@echo "make compile  - ให้ LLM คอมไพล์ข้อความอำนาจกรรมการเป็นกฎ JSON"
	@echo "make eval     - วัด accuracy ของกฎบน train + ดู clause ที่ยังพลาด"
	@echo "make predict  - สร้าง submissions/submission.csv"
	@echo "make lab      - เปิด JupyterLab สำหรับ notebooks/ (EDA เท่านั้น)"
	@echo "make clean-cache - ลบ cache คำตอบของ LLM"

setup:
	uv sync --extra notebook

data:
	uv run legal-download --config $(CONFIG) $(SET)

serve:
	uv run --extra serve vllm serve $(shell uv run python -c "from legal_act.config import load_config; print(load_config('$(CONFIG)').llm.model)") \
		--max-model-len 12288 --gpu-memory-utilization 0.93 --port 8000

compile:
	uv run legal-compile --config $(CONFIG) $(SET)

eval:
	uv run legal-eval --config $(CONFIG) $(SET)

predict:
	uv run legal-predict --config $(CONFIG) $(SET)

lab:
	uv run jupyter lab notebooks/

clean-cache:
	rm -rf datasets/llm_cache

clean: clean-cache
	rm -rf models submissions .venv
