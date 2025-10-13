#!/bin/bash
# Training script for Amazon ML Challenge 2025

set -e  # Exit on error

# Default values - Use dynamic path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_DIR/configs/config.yaml"
OUTPUT_DIR="$PROJECT_DIR/output"
EXPERIMENT_NAME="amazon-ml-$(date +%Y%m%d_%H%M%S)"
RESUME=""
WANDB_PROJECT="amazon-ml-2025"

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
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --resume)
            RESUME="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH           Path to config file (default: <project>/configs/config.yaml)"
            echo "  --output-dir PATH       Output directory (default: <project>/output)"
            echo "  --experiment-name NAME  Experiment name (default: amazon-ml-YYYYMMDD_HHMMSS)"
            echo "  --resume PATH           Resume from checkpoint"
            echo "  --wandb-project NAME    W&B project name (default: amazon-ml-2025)"
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
echo "Amazon ML Challenge 2025 - Training Pipeline"
echo "============================================"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Experiment: $EXPERIMENT_NAME"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set environment variables
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"
export WANDB_PROJECT="$WANDB_PROJECT"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Running on CPU."
    echo ""
fi

# Training command
TRAIN_CMD="python -m src"

# Use new simplified arguments for the working version
TRAIN_CMD="$TRAIN_CMD --folds 5 --model both --features dataset/features --output $OUTPUT_DIR"

echo "Running command: $TRAIN_CMD"
echo ""

# Create log file
LOG_FILE="$OUTPUT_DIR/${EXPERIMENT_NAME}_train.log"
mkdir -p "$(dirname "$LOG_FILE")"

# Run training with logging
if eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "============================================"
    echo "Training completed successfully!"
    echo "============================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $LOG_FILE"
    echo ""
    
    # Show final results if available
    RESULTS_FILE="$OUTPUT_DIR/cv_results.json"
    if [ -f "$RESULTS_FILE" ]; then
        echo "Cross-validation results:"
        python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
if 'summary' in results:
    print(f\"  Mean CV SMAPE: {results['summary']['mean_cv_smape']:.2f}% ± {results['summary']['std_cv_smape']:.2f}%\")
    print(f\"  Best Fold: {results['summary']['best_fold_smape']:.2f}%\")
"
    fi
    
else
    echo ""
    echo "============================================"
    echo "Training failed!"
    echo "============================================"
    echo "Check log file: $LOG_FILE"
    echo ""
    exit 1
fi