#!/bin/bash

# 简单SAE训练脚本 - Flower102和Color数据集

echo "开始SAE评估..."

export CUDA_VISIBLE_DEVICES=1

DATA_PATH="../PACE/dataset"
SAVE_PATH="./ckpt"

# 评估Color数据集  
echo "评估Color数据集..."
python src/main.py \
    --task Color \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --eval_only \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "Color评估完成!"

# 评估Flower102数据集
echo "评估Flower102数据集..."
python src/main.py \
    --task flower102 \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --eval_only \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "Flower102评估完成!"

# 评估CUB2011数据集
echo "评估CUB2011数据集..."
python src/main.py \
    --task cub2011 \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --eval_only \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "CUB2011评估完成!"

# 评估Cars数据集
echo "评估Cars数据集..."
python src/main.py \
    --task cars \
    --data_path $DATA_PATH \
    --save_path $SAVE_PATH \
    --eval_only \
    --use_wandb \
    --wandb_project "sae" \
    --use_finetuned_vit \
    --load_path ./ckpt \
    --pretrain_epoch 5 \
    --use_augmentation

echo "Cars评估完成!"    

echo "所有评估完成!"