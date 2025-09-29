#!/bin/bash

# 简单SAE训练脚本 - Flower102和Color数据集

echo "开始SAE训练..."

export CUDA_VISIBLE_DEVICES=1

DATA_PATH="../PACE/dataset"
SAVE_PATH="./ckpt"

# 训练Flower102数据集
echo "训练Flower102数据集..."
python src/main.py \
    --task flower102 \
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

echo "Flower102训练完成!"

# 训练Color数据集  
echo "训练Color数据集..."
python src/main.py \
    --task Color \
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

echo "Color训练完成!"

echo "所有训练完成!"