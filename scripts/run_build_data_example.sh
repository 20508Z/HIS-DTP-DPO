#!/usr/bin/env bash
set -euo pipefail

# Example preference-data construction script.
# Adjust annotation and image paths to the local dataset layout.

MODEL_PATH=${MODEL_PATH:-models/Qwen2.5-VL-3B}
VG_OBJECTS=${VG_OBJECTS:-data/visual_genome/objects.json}
VG_IMAGE_DIR=${VG_IMAGE_DIR:-data/visual_genome/images}
OUTPUT_FILE=${OUTPUT_FILE:-data/preference_data.json}

mkdir -p "$(dirname "$OUTPUT_FILE")"

python scripts/build_preference_data.py \
    --model_path "$MODEL_PATH" \
    --vg_objects_file "$VG_OBJECTS" \
    --vg_image_dir "$VG_IMAGE_DIR" \
    --output_path "$OUTPUT_FILE"
