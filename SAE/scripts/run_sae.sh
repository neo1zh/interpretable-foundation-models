#!/bin/bash

# Assign GPU
export CUDA_VISIBLE_DEVICES=7

# SAE Training Script for ViT Interpretability

# Set default values
TASK="flower102"
EXPANSION_FACTOR=4
SAE_EPOCHS=1000
SAE_LR=1e-3
BATCH_SIZE=256
DEVICE="cuda"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --expansion_factor)
            EXPANSION_FACTOR="$2"
            shift 2
            ;;
        --sae_epochs)
            SAE_EPOCHS="$2"
            shift 2
            ;;
        --sae_lr)
            SAE_LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --eval_only)
            EVAL_ONLY="--eval_only"
            shift
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

echo "Running SAE with the following parameters:"
echo "Task: $TASK"
echo "Expansion Factor: $EXPANSION_FACTOR"
echo "SAE Epochs: $SAE_EPOCHS"
echo "SAE Learning Rate: $SAE_LR"
echo "Batch Size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "Eval Only: ${EVAL_ONLY:-False}"

# Run the training/evaluation
python main.py \
    --task $TASK \
    --expansion_factor $EXPANSION_FACTOR \
    --sae_epochs $SAE_EPOCHS \
    --sae_lr $SAE_LR \
    --sae_batch_size $BATCH_SIZE \
    --device $DEVICE \
    --train \
    --use_augmentation \
    $EVAL_ONLY
