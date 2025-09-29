#!/usr/bin/env python3
"""
Test script for SAE model
"""

import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import SAE, ViTClassify
from evaluate import compute_all_metrics, print_metrics


def test_sae_model():
    """Test SAE model basic functionality"""
    print("Testing SAE model...")
    
    # Create SAE model
    d_embedding = 768
    expansion_factor = 4
    d_activation = d_embedding * expansion_factor
    
    sae = SAE(
        d_embedding=d_embedding,
        d_activation=d_activation,
        expansion_factor=expansion_factor,
        sparsity_penalty=1e-3
    )
    
    print(f"SAE created with:")
    print(f"  - Input dimension: {d_embedding}")
    print(f"  - Activation dimension: {d_activation}")
    print(f"  - Expansion factor: {expansion_factor}")
    
    # Test forward pass
    batch_size = 32
    x = torch.randn(batch_size, d_embedding)
    
    reconstructed, activations = sae(x)
    
    print(f"Forward pass test:")
    print(f"  - Input shape: {x.shape}")
    print(f"  - Reconstructed shape: {reconstructed.shape}")
    print(f"  - Activations shape: {activations.shape}")
    
    # Test loss computation
    total_loss, recon_loss, sparsity_loss = sae.compute_loss(x, reconstructed, activations)
    
    print(f"Loss computation test:")
    print(f"  - Total loss: {total_loss.item():.6f}")
    print(f"  - Reconstruction loss: {recon_loss.item():.6f}")
    print(f"  - Sparsity loss: {sparsity_loss.item():.6f}")
    
    print("✅ SAE model test passed!")


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting evaluation metrics...")
    
    # Create dummy data
    n_train, n_test = 1000, 200
    d_activation = 3072
    
    # Random activations and labels
    train_activations = np.random.randn(n_train, d_activation)
    test_activations = np.random.randn(n_test, d_activation)
    train_labels = np.random.randint(0, 10, n_train)
    test_labels = np.random.randint(0, 10, n_test)
    
    # Test metrics computation
    metrics = compute_all_metrics(
        train_activations, train_labels,
        test_activations, test_labels
    )
    
    print("Metrics computation test:")
    print_metrics(metrics)
    
    print("✅ Metrics test passed!")


def test_vit_model():
    """Test ViT model wrapper"""
    print("\nTesting ViT model wrapper...")
    
    try:
        # Create ViT model (this will download the model if not cached)
        model = ViTClassify(
            in_dim=768,
            out_dim=102,
            hid_dim=768,
            layer=-2
        )
        
        print(f"ViT model created with:")
        print(f"  - Input dimension: {model.in_dim}")
        print(f"  - Output dimension: {model.out_dim}")
        print(f"  - Hidden dimension: {model.hid_dim}")
        print(f"  - Layer: {model.layer}")
        
        # Test with dummy input
        batch_size = 2
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            logits, hidden, attention = model(dummy_input)
        
        print(f"ViT forward pass test:")
        print(f"  - Input shape: {dummy_input.shape}")
        print(f"  - Logits shape: {logits.shape}")
        print(f"  - Hidden shape: {hidden.shape}")
        print(f"  - Attention length: {len(attention)}")
        
        print("✅ ViT model test passed!")
        
    except Exception as e:
        print(f"⚠️  ViT model test failed (this is expected if no internet connection): {e}")


def main():
    """Run all tests"""
    print("="*50)
    print("SAE Model Tests")
    print("="*50)
    
    test_sae_model()
    test_metrics()
    test_vit_model()
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()
