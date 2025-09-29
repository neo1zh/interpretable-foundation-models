#!/bin/bash

# Simple SAE Evaluation Script
# Usage: ./eval_sae.sh [task] [gpu_id]

# Set default values
TASK=${1:-"flower102"}
GPU_ID=${2:-"7"}

# Set GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Evaluating SAE on task: $TASK (GPU: $GPU_ID)"

# Run evaluation
python main.py \
    --task $TASK \
    --save_path ../ckpt/ViT-SAE \
    --load_path ../ckpt \
    --use_wandb \
    --wandb_project "sae-vit-evaluation"
