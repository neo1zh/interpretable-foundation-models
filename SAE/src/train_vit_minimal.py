#!/usr/bin/env python3
"""
Minimal ViT training script for individual datasets
Trains ViT on Color and Flower102 datasets separately (5 epochs each)
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import parser
from model import ViTClassify
from utils import load_dataset_by_task, create_data_loaders, set_seed


def train_vit_on_dataset(task, data_path, save_path, epochs=5):
    """Train ViT on a single dataset"""
    print(f"🚀 Training ViT on {task} dataset for {epochs} epochs...")
    
    # Set random seed
    set_seed(2021)
    
    # Load dataset
    train_dataset, test_dataset, out_dim = load_dataset_by_task(task, data_path)
    train_loader, test_loader = create_data_loaders(
        train_dataset, test_dataset, 32, 64, 4
    )
    
    print(f"Dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
    print(f"Output classes: {out_dim}")
    
    # Initialize ViT model
    model = ViTClassify(out_dim=out_dim, layer=-2)
    model = model.to('cuda')
    
    # Unfreeze ViT parameters for training
    for param in model.parameters():
        param.requires_grad = True
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            images = batch['encodings'].to('cuda')
            labels = batch['labels'].to('cuda')
            
            # Forward pass
            optimizer.zero_grad()
            logits, embeddings, attention = model(images)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100.*correct/total:.2f}%"
            })
        
        # Print epoch summary
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
    
    # Save model
    os.makedirs(save_path, exist_ok=True)
    model_path = os.path.join(save_path, f'{task}_epoch{epochs}.pt')
    torch.save(model.state_dict(), model_path)
    print(f"✅ Saved {task} model to: {model_path}")
    
    return model_path


def main():
    """Main function"""
    args = parser.parse_args()
    
    # Configuration
    data_path = args.data_path
    save_path = args.load_path  # Use load_path as the base save directory
    
    print("🎯 Minimal ViT Training on Individual Datasets")
    print("="*50)
    print(f"Data path: {data_path}")
    print(f"Save path: {save_path}")
    print("="*50)
    
    # # Train on Color dataset
    # print("\n📊 Training on Color dataset...")
    # color_model_path = train_vit_on_dataset(
    #     'Color', data_path, save_path, epochs=5
    # )
    
    # # Train on Flower102 dataset
    # print("\n🌸 Training on Flower102 dataset...")
    # flower_model_path = train_vit_on_dataset(
    #     'flower102', data_path, save_path, epochs=5
    # )

    # Train on CUB2011 dataset
    print("\n🐦 Training on CUB2011 dataset...")
    cub2011_model_path = train_vit_on_dataset(
        'cub2011', data_path, save_path, epochs=5
    )

    # Train on Cars dataset
    print("\n🚗 Training on Cars dataset...")
    cars_model_path = train_vit_on_dataset(
        'cars', data_path, save_path, epochs=5
    )

    print("\n🎉 Training completed!")
    # print(f"Color model: {color_model_path}")
    print(f"CUB2011 model: {cub2011_model_path}")
    print(f"Cars model: {cars_model_path}")
    # print(f"Flower102 model: {flower_model_path}")
    print("\nTo use these models:")
    # print(f"python src/main.py --task Color --train --pretrain_epoch 5")
    # print(f"python src/main.py --task flower102 --train --pretrain_epoch 5")
    print(f"python src/main.py --task cub2011 --train --pretrain_epoch 5")
    print(f"python src/main.py --task cars --train --pretrain_epoch 5")


if __name__ == "__main__":
    main()
