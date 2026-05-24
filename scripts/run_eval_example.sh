#!/usr/bin/env bash
set -euo pipefail

# Example POPE + CHAIR evaluation script.
# Usage:
#   bash scripts/run_eval_example.sh <model_path> [lora_path] [output_prefix]

MODEL_PATH=${1:-models/Qwen2.5-VL-3B}
LORA_PATH=${2:-}
OUTPUT_PREFIX=${3:-outputs/eval}
GPU_START=${GPU_START:-0}

LORA_ARG=()
if [ -n "$LORA_PATH" ]; then
    LORA_ARG=(--lora_path "$LORA_PATH")
fi

mkdir -p "$(dirname "$OUTPUT_PREFIX")"

CUDA_VISIBLE_DEVICES=$((GPU_START)) python eval/eval_pope.py \
    --model_path "$MODEL_PATH" "${LORA_ARG[@]}" \
    --pope_file data/POPE/coco_pope_random.json \
    --image_dir data/coco/val2014 \
    --output_file "${OUTPUT_PREFIX}_pope_random.jsonl"

CUDA_VISIBLE_DEVICES=$((GPU_START)) python eval/eval_pope.py \
    --model_path "$MODEL_PATH" "${LORA_ARG[@]}" \
    --pope_file data/POPE/coco_pope_popular.json \
    --image_dir data/coco/val2014 \
    --output_file "${OUTPUT_PREFIX}_pope_popular.jsonl"

CUDA_VISIBLE_DEVICES=$((GPU_START)) python eval/eval_pope.py \
    --model_path "$MODEL_PATH" "${LORA_ARG[@]}" \
    --pope_file data/POPE/coco_pope_adversarial.json \
    --image_dir data/coco/val2014 \
    --output_file "${OUTPUT_PREFIX}_pope_adversarial.jsonl"

CUDA_VISIBLE_DEVICES=$((GPU_START)) python eval/eval_chair.py \
    --model_path "$MODEL_PATH" "${LORA_ARG[@]}" \
    --coco_ann_file data/coco/annotations/instances_val2014.json \
    --image_dir data/coco/val2014 \
    --num_samples 500 \
    --output_file "${OUTPUT_PREFIX}_chair_results.json"
