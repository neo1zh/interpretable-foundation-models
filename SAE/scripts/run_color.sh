#!/bin/bash

# SAE Training Script for Color Dataset
# Usage: ./run_color.sh [gpu_id] [expansion_factor] [epochs]

# Set default values
GPU_ID=${1:-7}
EXPANSION_FACTOR=${2:-4}
SAE_EPOCHS=${3:-1000}
SAE_LR=${4:-1e-3}
BATCH_SIZE=${5:-256}

# Assign GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "🎨 SAE Training on Color Dataset"
echo "================================"
echo "GPU ID: $GPU_ID"
echo "Expansion Factor: $EXPANSION_FACTOR"
echo "SAE Epochs: $SAE_EPOCHS"
echo "SAE Learning Rate: $SAE_LR"
echo "Batch Size: $BATCH_SIZE"
echo "================================"

# Create timestamp for unique run identification
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="color_${EXPANSION_FACTOR}x_${SAE_EPOCHS}ep_${TIMESTAMP}"

echo "🚀 Starting training with run name: $RUN_NAME"

# Run the training
cd /common/home/zz1009/workspace/interpretable-foundation-models/SAE/src

python main.py \
    --task Color \
    --expansion_factor $EXPANSION_FACTOR \
    --sae_epochs $SAE_EPOCHS \
    --sae_lr $SAE_LR \
    --sae_batch_size $BATCH_SIZE \
    --device cuda \
    --train \
    --use_augmentation \
    --use_wandb \
    --wandb_project "sae-color" \
    --wandb_run_name "$RUN_NAME" \
    --wandb_tags "color" "sae" "training" \
    --wandb_notes "SAE training on Color dataset with expansion factor $EXPANSION_FACTOR" \
    --eval_interval 50 \
    --save_interval 100

echo "✅ Color SAE training completed!"
echo "📊 Check your Wandb dashboard for detailed metrics"
echo "💾 Model saved to: ../ckpt/ViT-SAE/"
