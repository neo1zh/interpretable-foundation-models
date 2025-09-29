#!/bin/bash

# Example script demonstrating Wandb integration for SAE training
# This script shows different ways to use Wandb with the SAE project

echo "🚀 SAE Wandb Integration Examples"
echo "================================="

# Check if wandb is installed
if ! python -c "import wandb" 2>/dev/null; then
    echo "❌ Wandb not installed. Please run: pip install wandb"
    exit 1
fi

# Check if user is logged in to wandb
if ! wandb status >/dev/null 2>&1; then
    echo "⚠️  Not logged in to Wandb. Please run: wandb login"
    echo "   You can get your API key from: https://wandb.ai/authorize"
    exit 1
fi

echo "✅ Wandb is ready!"

# Example 1: Basic training with Wandb
echo ""
echo "📊 Example 1: Basic Training with Wandb"
echo "--------------------------------------"
echo "Command:"
echo "python src/train_sae.py --train --use_wandb --task flower102"
echo ""
echo "This will:"
echo "- Train SAE on flower102 dataset"
echo "- Log all metrics to Wandb"
echo "- Use default project name: sae-vit-interpretability"
echo "- Auto-generate run name with timestamp"
echo ""

# Example 2: Advanced configuration
echo "📊 Example 2: Advanced Configuration"
echo "-----------------------------------"
echo "Command:"
echo "python src/train_sae.py \\"
echo "    --train \\"
echo "    --use_wandb \\"
echo "    --wandb_project 'sae-vit-experiments' \\"
echo "    --wandb_entity 'your-team' \\"
echo "    --wandb_run_name 'flower102_experiment_1' \\"
echo "    --wandb_tags 'baseline' 'flower102' 'sae' \\"
echo "    --wandb_notes 'Initial SAE training on flower102 dataset' \\"
echo "    --task flower102 \\"
echo "    --sae_epochs 1000 \\"
echo "    --sparsity_penalty 1e-3 \\"
echo "    --expansion_factor 4"
echo ""

# Example 3: Evaluation only
echo "📊 Example 3: Evaluation Only"
echo "-----------------------------"
echo "Command:"
echo "python src/train_sae.py \\"
echo "    --use_wandb \\"
echo "    --wandb_project 'sae-vit-evaluation' \\"
echo "    --task flower102 \\"
echo "    --save_path ../ckpt/ViT-SAE"
echo ""

# Example 4: Hyperparameter sweep (conceptual)
echo "📊 Example 4: Hyperparameter Sweep"
echo "----------------------------------"
echo "You can create a sweep configuration in Python:"
echo ""
echo "import wandb"
echo ""
echo "sweep_config = {"
echo "    'method': 'bayes',"
echo "    'metric': {'name': 'eval/total_loss', 'goal': 'minimize'},"
echo "    'parameters': {"
echo "        'sparsity_penalty': {'min': 1e-4, 'max': 1e-2, 'distribution': 'log_uniform'},"
echo "        'expansion_factor': {'values': [2, 4, 8, 16]},"
echo "        'sae_lr': {'min': 1e-4, 'max': 1e-2, 'distribution': 'log_uniform'}"
echo "    }"
echo "}"
echo ""

# Interactive mode
echo "🎯 Interactive Mode"
echo "==================="
echo "Would you like to run a quick example? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Choose an example to run:"
    echo "1. Basic training (flower102, 50 epochs)"
    echo "2. Advanced training (flower102, custom config)"
    echo "3. Evaluation only"
    echo "4. Skip"
    echo ""
    read -p "Enter choice (1-4): " choice
    
    case $choice in
        1)
            echo "🚀 Running basic training example..."
            python src/train_sae.py \
                --train \
                --use_wandb \
                --wandb_project "sae-examples" \
                --wandb_run_name "basic_example_$(date +%Y%m%d_%H%M%S)" \
                --wandb_tags "example" "basic" "flower102" \
                --task flower102 \
                --sae_epochs 50 \
                --eval_interval 10
            ;;
        2)
            echo "🚀 Running advanced training example..."
            python src/train_sae.py \
                --train \
                --use_wandb \
                --wandb_project "sae-examples" \
                --wandb_run_name "advanced_example_$(date +%Y%m%d_%H%M%S)" \
                --wandb_tags "example" "advanced" "flower102" "custom-config" \
                --wandb_notes "Advanced example with custom hyperparameters" \
                --task flower102 \
                --sae_epochs 100 \
                --sparsity_penalty 5e-4 \
                --expansion_factor 8 \
                --sae_lr 5e-4 \
                --eval_interval 20
            ;;
        3)
            echo "🚀 Running evaluation example..."
            echo "Note: This requires a trained model. Make sure you have run training first."
            python src/train_sae.py \
                --use_wandb \
                --wandb_project "sae-examples" \
                --wandb_run_name "evaluation_example_$(date +%Y%m%d_%H%M%S)" \
                --wandb_tags "example" "evaluation" "flower102" \
                --task flower102
            ;;
        4)
            echo "Skipping examples."
            ;;
        *)
            echo "Invalid choice. Skipping examples."
            ;;
    esac
else
    echo "Skipping interactive examples."
fi

echo ""
echo "📚 Additional Resources:"
echo "- Wandb Integration Guide: WANDB_INTEGRATION.md"
echo "- Wandb Documentation: https://docs.wandb.ai/"
echo "- SAE Project README: README.md"
echo ""
echo "🎉 Happy experimenting with Wandb and SAE!"
