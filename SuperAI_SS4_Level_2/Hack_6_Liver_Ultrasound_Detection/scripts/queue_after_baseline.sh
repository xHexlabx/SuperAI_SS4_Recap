#!/bin/bash
# Waits for the running liver-train to exit, then launches the synth run. Detached from the session.
D=/home/hextex/Documents/Github/SuperAI_SS4_Recap/SuperAI_SS4_Level_2/Hack_6_Liver_Ultrasound_Detection
cd "$D" || exit 1
while pgrep -f "bin/liver-train" > /dev/null; do sleep 60; done
echo "baseline finished at $(date)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec uv run liver-train --set data.mobile_oversample=4 --set data.synth_dir=datasets/synth_v1 --set model.imgsz=640 --set train.batch=16 --set train.epochs=30 --set train.patience=12
