#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the Video-3D-LLM_data directory}"
CKPT_PATH="${CKPT_PATH:?Set CKPT_PATH to the released Video-3D-LLM checkpoint}"
OUT_DIR="${OUT_DIR:-results/hisc}"
N_GPU="${N_GPU:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
mkdir -p "$OUT_DIR"

for task in scanrefer scanqa sqa3d scan2cap multi3drefer; do
    case "$task" in
        scanrefer) question_file="scanrefer_vg_val_llava_style.json" ;;
        *) question_file="${task}_val_llava_style.json" ;;
    esac
    answer_file="$OUT_DIR/${task}.jsonl"
    if [[ -e "$answer_file" ]]; then
        printf "Output already exists: %s\n" "$answer_file" >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="$GPU_IDS" PYTHONPATH=. "$PYTHON_BIN" "llava/eval/model_${task}.py" \
        --model-path "$CKPT_PATH" \
        --video-folder "$DATA_ROOT" \
        --embodiedscan-folder "$DATA_ROOT/embodiedscan" \
        --n_gpu "$N_GPU" \
        --frame_sampling_strategy uniform \
        --max_frame_num 32 \
        --question-file "$DATA_ROOT/processed/$question_file" \
        --conv-mode qwen_1_5 \
        --answer-file "$answer_file" \
        --overwrite_cfg true
    "$PYTHON_BIN" "llava/eval/eval_${task}.py" --input-file "$answer_file"
done
