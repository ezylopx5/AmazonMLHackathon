#!/bin/bash
set -e

echo "🚀 Amazon ML Challenge 2025 - Training Pipeline"
echo "================================================"

# Use the new simplified training command
TRAIN_CMD="python -m src"

# Add arguments for Lightning AI paths
TRAIN_ARGS="--folds 5 --model both --features dataset/features --output output"

echo "Executing: $TRAIN_CMD $TRAIN_ARGS"
echo ""

# Run training
$TRAIN_CMD $TRAIN_ARGS

echo ""
echo "✅ Checking for checkpoint files..."
CHECKPOINT_DIR="output/checkpoints"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "📁 Checkpoint directory exists"
    
    # Check for pickle files (model checkpoints)
    PKL_FILES=$(ls $CHECKPOINT_DIR/*.pkl 2>/dev/null | wc -l)
    if [ $PKL_FILES -gt 0 ]; then
        echo "✅ Found $PKL_FILES .pkl checkpoint files:"
        ls -la $CHECKPOINT_DIR/*.pkl
    else
        echo "⚠ No .pkl checkpoint files found"
    fi
    
    # Check for PyTorch files
    PT_FILES=$(ls $CHECKPOINT_DIR/*.pt 2>/dev/null | wc -l)
    if [ $PT_FILES -gt 0 ]; then
        echo "✅ Found $PT_FILES .pt checkpoint files:"
        ls -la $CHECKPOINT_DIR/*.pt
    else
        echo "ℹ No .pt checkpoint files found (expected for XGBoost/LightGBM)"
    fi
    
    # Check for prediction files
    NPY_FILES=$(ls $CHECKPOINT_DIR/*.npy 2>/dev/null | wc -l)
    if [ $NPY_FILES -gt 0 ]; then
        echo "✅ Found $NPY_FILES .npy prediction files:"
        ls -la $CHECKPOINT_DIR/*.npy
    fi
    
else
    echo "❌ Checkpoint directory not found!"
    exit 1
fi

echo ""
echo "================================================"
echo "🎉 Training pipeline completed successfully!"
echo "================================================"