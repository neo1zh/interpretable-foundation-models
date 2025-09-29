#!/bin/bash

# Minimal ViT training script
# Trains ViT on Color and Flower102 datasets separately (5 epochs each)

set -e

echo "🚀 Minimal ViT Training"
echo "========================"

# Default paths
DATA_PATH="../PACE/dataset"
SAVE_PATH="./ckpt/ViT-pretrain"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --save_path)
            SAVE_PATH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Minimal ViT Training Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --data_path PATH    Path to dataset directory (default: ../dataset)"
            echo "  --save_path PATH    Path to save models (default: ../ckpt)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "This script will:"
            echo "1. Train ViT on Color dataset for 5 epochs"
            echo "2. Train ViT on Flower102 dataset for 5 epochs"
            echo "3. Save models as: ViT-Base/Color_epoch5.pt and ViT-Base/flower102_epoch5.pt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Data path: $DATA_PATH"
echo "Save path: $SAVE_PATH"
echo ""

# Run training
python src/train_vit_minimal.py \
    --data_path "$DATA_PATH" \
    --load_path "$SAVE_PATH"

echo ""
echo "✅ Training completed!"
echo "Models saved in: $SAVE_PATH/ViT-Base/"
echo ""
echo "Generated files:"
echo "- $SAVE_PATH/ViT-Base/Color_epoch5.pt"
echo "- $SAVE_PATH/ViT-Base/flower102_epoch5.pt"
