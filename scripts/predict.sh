#!/bin/bash
# Prediction script for Amazon ML Challenge 2025

set -e  # Exit on error

# Default values - Use dynamic path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_DIR/configs/config.yaml"
OUTPUT_DIR="$PROJECT_DIR/output"
CHECKPOINT_DIR=""
SUBMISSION_DIR="$PROJECT_DIR/submissions"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --submission-dir)
            SUBMISSION_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH           Path to config file (default: <project>/configs/config.yaml)"
            echo "  --output-dir PATH       Output directory (default: <project>/output)"
            echo "  --checkpoint-dir PATH   Checkpoint directory (default: from config)"
            echo "  --submission-dir PATH   Submission directory (default: <project>/submissions)"
            echo "  --help                  Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Set paths relative to project root
cd "$PROJECT_ROOT"

echo "============================================"
echo "Amazon ML Challenge 2025 - Prediction Pipeline"
echo "============================================"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Submission: $SUBMISSION_DIR"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SUBMISSION_DIR"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Running on CPU."
    echo ""
fi

# Check for trained models
if [ ! -z "$CHECKPOINT_DIR" ]; then
    CKPT_DIR="$CHECKPOINT_DIR"
else
    CKPT_DIR="$OUTPUT_DIR/checkpoints"
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory not found at $CKPT_DIR"
    echo "Please run training first or specify --checkpoint-dir"
    exit 1
fi

# Count available checkpoints
CHECKPOINT_COUNT=$(find "$CKPT_DIR" -name "fold*_best.pt" | wc -l)
if [ "$CHECKPOINT_COUNT" -eq 0 ]; then
    echo "Error: No trained model checkpoints found in $CKPT_DIR"
    echo "Please run training first"
    exit 1
fi

echo "Found $CHECKPOINT_COUNT model checkpoints"
echo ""

# Prediction command
PREDICT_CMD="python -c \"
import sys
sys.path.append('src')
from config import Config
from infer import generate_predictions

# Load config
config = Config.from_yaml('$CONFIG_PATH')

# Override paths if specified
config.output_dir = '$OUTPUT_DIR'
config.submission_dir = '$SUBMISSION_DIR'
if '$CHECKPOINT_DIR':
    config.ckpt_dir = '$CKPT_DIR'

# Generate predictions
generate_predictions(config)
\""

echo "Running prediction pipeline..."
echo ""

# Create log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/predict_${TIMESTAMP}.log"

# Run prediction with logging
if eval "$PREDICT_CMD" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "============================================"
    echo "Prediction completed successfully!"
    echo "============================================"
    
    # Find submission file
    SUBMISSION_FILE=$(find "$SUBMISSION_DIR" -name "submission.csv" -type f | head -1)
    
    if [ -f "$SUBMISSION_FILE" ]; then
        echo "Submission file: $SUBMISSION_FILE"
        
        # Show submission statistics
        echo ""
        echo "Submission statistics:"
        python -c "
import pandas as pd
import numpy as np
df = pd.read_csv('$SUBMISSION_FILE')
prices = df['price'].values
print(f'  Samples: {len(prices):,}')
print(f'  Mean price: \${np.mean(prices):.2f}')
print(f'  Median price: \${np.median(prices):.2f}')
print(f'  Min price: \${np.min(prices):.2f}')
print(f'  Max price: \${np.max(prices):.2f}')
print(f'  Std price: \${np.std(prices):.2f}')
"
        echo ""
        echo "Submission ready for upload!"
    else
        echo "Warning: Submission file not found"
    fi
    
else
    echo ""
    echo "============================================"
    echo "Prediction failed!"
    echo "============================================"
    echo "Check log file: $LOG_FILE"
    echo ""
    exit 1
fi