#!/usr/bin/env bash
set -euo pipefail

# Example launch script for HIS-Guard training on Qwen2.5-VL-7B.
# Adjust paths and GPU count for the local environment.

MODEL_PATH=${MODEL_PATH:-models/Qwen2.5-VL-7B}
DATA_PATH=${DATA_PATH:-data/preference_data.json}
OUTPUT_DIR=${OUTPUT_DIR:-checkpoints/his_guard_qwen7b}
NUM_GPUS=${NUM_GPUS:-4}
MASTER_PORT=${MASTER_PORT:-29501}

mkdir -p "$(dirname "$OUTPUT_DIR")" outputs

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" train.py \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs 3 \
    --batch_size 1 \
    --grad_accum 8 \
    --lr 1e-6 \
    --warmup_ratio 0.1 \
    --beta 0.1 \
    --gamma_visual 0.2 \
    --gamma_anchor 0.1 \
    --anchor_value 0.0 \
    --mask_ratio 0.3 \
    --mask_method random \
    --lora_rank 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --save_steps 200

