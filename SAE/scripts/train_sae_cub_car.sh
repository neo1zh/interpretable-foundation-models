#!/bin/bash

# 简单SAE训练脚本 - Flower102和Color数据集

echo "开始SAE训练..."

export CUDA_VISIBLE_DEVICES=1

DATA_PATH="../PACE/dataset"
SAVE_PATH="./ckpt"

# 训练CUB2011数据集
echo "训练CUB2011数据集..."
python src/main.py \
    --task cub2011 \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --train \
    --sae_epochs 100 \
    --sae_lr 1e-3 \
    --expansion_factor 4 \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "CUB2011训练完成!"

# 训练Cars数据集
echo "训练Cars数据集..."
python src/main.py \
    --task cars \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --train \
    --sae_epochs 100 \
    --sae_lr 1e-3 \
    --expansion_factor 4 \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "Cars训练完成!"

echo "所有训练完成!"