#!/usr/bin/env bash
set -e

GPU_IDS="0,1,2,3"
NUM_DEVICES=$(echo "$GPU_IDS" | tr ',' '\n' | wc -l)

# 4卡训练
BATCH_PER_DEVICE=1
GLOBAL_BATCH_SIZE=4

export PYTHONPATH=$PYTHONPATH:$(pwd)

BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
# BASE_MODEL="/home/ma-user/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct/snapshots/cc594898137f460bfe9f0759e9844b3ce807cfb5"

OUT_DIR_STAGE1="output7b-6000_3000_3000_4000_stage4_nodrop_vqa8000/lora_vision_test/lora_stage1"
MERGED_STAGE1_MODEL="output7b-6000_3000_3000_4000_stage4_nodrop_vqa8000/lora_merged/lora_stage1_merged"

OUT_DIR_STAGE234="output7b-6000_3000_3000_4000_stage4_nodrop_vqa8000/lora_vision_test/lora_stage234"
FINAL_MERGED_MODEL="output7b-6000_3000_3000_4000_stage4_nodrop_vqa8000/lora_merged/lora_stage234_merged"

DATA_PATH="/home/ma-user/work/lbx/CoVT/CoVT_data/data.json"
IMAGE_FOLDER="/home/ma-user/work/lbx/CoVT/CoVT_data/images"
# /home/ma-user/work/lbx/CoVT/CoVT-Dataset
VISUAL_MODEL_ID="['sam', 'depth', 'dino']"


# echo "==== [1/4] First stage training: max_steps=6000 ===="
# MODEL_NAME="$BASE_MODEL" \
# MODEL_PATH="$BASE_MODEL" \
# GPU_IDS="$GPU_IDS" \
# NUM_DEVICES="$NUM_DEVICES" \
# BATCH_PER_DEVICE="$BATCH_PER_DEVICE" \
# GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
# OUTPUT_DIR="$OUT_DIR_STAGE1" \
# RUN_NAME="stage1_train" \
# STAGE_0_STEP=6000 \
# STAGE_1_STEP=6000 \
# STAGE_2_STEP=6000 \
# VQA_ONLY_STAGE=6000 \
# MAX_STEPS=6000 \
# DATA_PATH="$DATA_PATH" \
# IMAGE_FOLDER="$IMAGE_FOLDER" \
# VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
# bash scripts/run.sh


# STAGE_0_STEP=0 \
# STAGE_1_STEP=1500 \
# STAGE_2_STEP=4500 \
# VQA_ONLY_STAGE=4500 \
# MAX_STEPS=4500 \

echo "==== [2/4] First merge LoRA ===="

MODEL_NAME="$BASE_MODEL" \
MODEL_PATH="$OUT_DIR_STAGE1" \
SAVE_MODEL_PATH="$MERGED_STAGE1_MODEL" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/merge_lora.sh


echo "==== [3/4] Joint training of stage 2/3/4: max_steps=10000 ===="

MODEL_NAME="$MERGED_STAGE1_MODEL" \
MODEL_PATH="$MERGED_STAGE1_MODEL" \
GPU_IDS="$GPU_IDS" \
NUM_DEVICES="$NUM_DEVICES" \
BATCH_PER_DEVICE="$BATCH_PER_DEVICE" \
GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
OUTPUT_DIR="$OUT_DIR_STAGE234" \
RUN_NAME="stage234_train" \
STAGE_0_STEP=0 \
STAGE_1_STEP=3000 \
STAGE_2_STEP=6000 \
VQA_ONLY_STAGE=8000 \
MAX_STEPS=10000 \
DATA_PATH="$DATA_PATH" \
IMAGE_FOLDER="$IMAGE_FOLDER" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/run.sh


# STAGE_0_STEP=0 \
# STAGE_1_STEP=3000 \
# STAGE_2_STEP=6000 \
# VQA_ONLY_STAGE=8000 \
# MAX_STEPS=10000 \

echo "==== [4/4] Second merge LoRA (final) ===="

MODEL_NAME="$MERGED_STAGE1_MODEL" \
MODEL_PATH="$OUT_DIR_STAGE234" \
SAVE_MODEL_PATH="$FINAL_MERGED_MODEL" \
VISUAL_MODEL_ID="$VISUAL_MODEL_ID" \
bash scripts/merge_lora.sh

echo "==== All processes completed, final model: $FINAL_MERGED_MODEL ===="
