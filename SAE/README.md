# SAE for ViT Interpretability

This project implements a Sparse Autoencoder (SAE) for interpreting Vision Transformer (ViT) activations, following the same evaluation framework as PACE but using SAE instead of probabilistic topic modeling.

## Overview

The SAE approach:
- Uses the same Google ViT-base model to extract activations from the -2 layer
- Trains a sparse autoencoder with expansion factor (default: 4x, so 768 → 3072 dimensions)
- Evaluates using the same metrics as PACE: faithfulness, stability, sparsity, and parsimony

## Key Differences from PACE

| Aspect | PACE | SAE |
|--------|------|-----|
| **Method** | Probabilistic topic model | Sparse autoencoder |
| **Activation Dimension** | K=100 topics | 4×d_embedding = 3072 |
| **Architecture** | Probabilistic model | Encoder-decoder |
| **Hierarchical** | Yes | No |

## Metrics

1. **Faithfulness**: Train a linear classifier on SAE activations to predict labels
2. **Stability**: Compare activations between original and augmented images
3. **Sparsity**: Fraction of activations below threshold (0.1/d_activation)
4. **Parsimony**: Number of activation dimensions (3072 for 4x expansion)

## Installation

```bash
# Install dependencies
pip install torch torchvision transformers datasets scikit-learn numpy matplotlib tqdm

# Or use the environment file from PACE
conda env create -f ../PACE/environment_PACE.yml
```

## Usage

### Training SAE

```bash
# Basic training
python src/main.py --task flower102 --train

# With custom parameters
python src/main.py \
    --task flower102 \
    --expansion_factor 4 \
    --sae_epochs 1000 \
    --sae_lr 1e-3 \
    --sae_batch_size 256 \
    --train

# Using the shell script
./src/run_sae.sh --task flower102 --expansion_factor 4 --sae_epochs 1000
```

### Evaluation Only

```bash
# Evaluate trained SAE
python src/main.py --task flower102 --eval_only

# Using shell script
./src/run_sae.sh --task flower102 --eval_only
```

### Available Tasks

- `flower102`: Oxford Flowers-102 dataset
- `Color`: Custom color classification dataset
- `cub2011`: CUB-200-2011 dataset (requires manual setup)
- `cars`: Stanford Cars dataset (requires manual setup)

## Configuration

Key parameters in `config.py`:

```python
# SAE model
--d_embedding 768          # ViT embedding dimension
--expansion_factor 4       # SAE expansion factor
--d_activation 3072        # SAE activation dimension (auto-calculated)
--sparsity_penalty 1e-3    # L1 sparsity penalty

# Training
--sae_epochs 1000         # SAE training epochs
--sae_lr 1e-3            # SAE learning rate
--sae_batch_size 256     # SAE batch size
--sae_patience 50        # Early stopping patience
```

## Project Structure

```
SAE/
├── src/
│   ├── main.py              # Main training/evaluation script
│   ├── train_sae.py         # SAE training logic
│   ├── model.py             # SAE and ViT model definitions
│   ├── evaluate.py          # Evaluation metrics
│   ├── utils.py             # Utility functions
│   ├── config.py            # Configuration
│   └── run_sae.sh           # Training script
├── ckpt/                    # Model checkpoints
└── README.md
```

## Expected Output

The training will output metrics like:

```
==================================================
SAE EVALUATION METRICS
==================================================
Faithfulness (MLP):     0.8542
Faithfulness (Linear):  0.8234
Sparsity:               0.7234
Parsimony:              3072
Stability:              0.1234
==================================================
```

## Comparison with PACE

To compare SAE with PACE results:

1. **Faithfulness**: SAE should achieve similar or better classification performance
2. **Sparsity**: SAE typically achieves higher sparsity due to L1 penalty
3. **Parsimony**: SAE has more dimensions (3072 vs 100) but may be more interpretable
4. **Stability**: Both should show similar stability under augmentation

## Notes

- The SAE does not have hierarchical structure like PACE
- SAE activations are deterministic (not probabilistic)
- The expansion factor can be adjusted (2x, 4x, 8x) to balance sparsity vs. reconstruction quality
- Early stopping is used to prevent overfitting
- All metrics are computed on the same test set for fair comparison
