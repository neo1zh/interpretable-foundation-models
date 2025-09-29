#!/usr/bin/env python3
"""
Main script for SAE training and evaluation
"""

import os
import sys
import argparse
import torch
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import parser
from train_sae import train_sae, evaluate_sae
from model import SAE
from utils import set_seed, load_dataset_by_task, extract_embeddings, extract_augmented_activations, create_data_loaders


def main():
    """Main function"""
    args = parser.parse_args()
    
    print("="*60)
    print("SAE for ViT Interpretability")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"SAE expansion factor: {args.expansion_factor}")
    print(f"SAE activation dimension: {args.d_activation if args.d_activation else args.d_embedding * args.expansion_factor}")
    print(f"Device: {args.device}")
    print("="*60)
    
    if args.train:
        print("🚀 Starting SAE training...")
        sae_model, train_embeddings, train_labels, test_embeddings, test_labels, train_embeddings_aug = train_sae(args)
        
        print("📊 Evaluating SAE...")
        
        metrics = evaluate_sae(args, sae_model, train_embeddings, train_labels, test_embeddings, test_labels, train_embeddings_aug)
        
        print("✅ Training and evaluation completed!")
        
    else:
        print("📊 Evaluation only mode...")
        
        # Set random seed
        set_seed(args.seed)
        
        # Create save directory path
        args.save_path = os.path.normpath(os.path.join(args.save_path, "SAE", args.task))
        
        # Load pretrained SAE model
        sae_model = SAE(
            d_embedding=args.d_embedding,
            d_activation=args.d_activation if args.d_activation else args.d_embedding * args.expansion_factor,
            expansion_factor=args.expansion_factor,
            sparsity_penalty=args.sparsity_penalty
        )
        
        model_path = os.path.join(args.save_path, "best_sae_model.pt")
        if os.path.exists(model_path):
            sae_model.load_state_dict(torch.load(model_path, map_location=args.device))
            sae_model = sae_model.to(args.device)  # Move model to device after loading
            print(f"✅ Loaded pretrained SAE model from: {model_path}")
        else:
            print(f"❌ SAE model not found at: {model_path}")
            print("Please train the model first with --train flag")
            return
        
        # Load or extract embeddings
        train_embeddings_aug = None  # Initialize for evaluation mode
        if args.load_activations:
            print("📁 Loading pre-computed activations...")
            try:
                train_activations = np.load(os.path.join(args.save_path, f'activations_epoch_{args.sae_epochs}.npy'))
                train_labels = np.load(os.path.join(args.save_path, f'labels_epoch_{args.sae_epochs}.npy'))
                test_embeddings = torch.tensor(train_activations)  # Use activations as embeddings for evaluation
                test_labels = torch.tensor(train_labels)
                train_embeddings = test_embeddings  # Same for simplicity
                train_labels = test_labels
                print("✅ Loaded pre-computed activations")
            except FileNotFoundError:
                print("❌ Pre-computed activations not found, extracting embeddings...")
                args.load_activations = False
        
        if not args.load_activations:
            print("🔄 Extracting ViT embeddings...")
            # Load dataset and extract embeddings
            train_dataset, test_dataset, out_dim = load_dataset_by_task(args.task, args.data_path)
            train_loader, test_loader = create_data_loaders(
                train_dataset, test_dataset, 
                args.train_batch_size, args.eval_batch_size, args.num_workers
            )
            
            # Load ViT model
            from model import ViTClassify
            model = ViTClassify(
                out_dim=out_dim, 
                layer=args.layer
            )
            model = model.to(args.device)
            
            # Load pretrained ViT weights
            pretrain_path = os.path.join(args.load_path, 'ViT-pretrain', f'{args.task}_epoch{args.pretrain_epoch}.pt')
            if os.path.exists(pretrain_path):
                model.load_state_dict(torch.load(pretrain_path, map_location=args.device))
                print(f"✅ Loaded pretrained ViT from: {pretrain_path}")
            else:
                print("⚠️  Using HuggingFace pretrained ViT")
            
            # Extract embeddings
            train_embeddings, train_labels, train_preds = extract_embeddings(model, train_loader, args.device, args.layer)
            test_embeddings, test_labels, test_preds = extract_embeddings(model, test_loader, args.device, args.layer)
            
            # Extract augmented embeddings for stability evaluation (if enabled)
            train_embeddings_aug = None
            if args.use_augmentation:
                print("🔄 Extracting augmented ViT embeddings for stability evaluation...")
                train_embeddings_aug = extract_augmented_activations(
                    model, train_loader, args.device, args.layer, args.augmentation_strength
                )
                print("✅ Extracted augmented ViT embeddings")
            
            print("✅ Extracted ViT embeddings")
        
        # Evaluate SAE
        print("📊 Evaluating SAE...")
        metrics = evaluate_sae(args, sae_model, train_embeddings, train_preds, test_embeddings, test_preds, train_embeddings_aug)
        
        print("✅ Evaluation completed!")


if __name__ == "__main__":
    main()
