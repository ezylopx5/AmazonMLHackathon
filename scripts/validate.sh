#!/bin/bash
# Validation script for Amazon ML Challenge 2025

set -e  # Exit on error

# Default values - Use dynamic path resolution
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_DIR/configs/config.yaml"
OUTPUT_DIR="$PROJECT_DIR/output" 
VALIDATION_DIR=""

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
        --validation-dir)
            VALIDATION_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --config PATH           Path to config file (default: <project>/configs/config.yaml)"
            echo "  --output-dir PATH       Output directory (default: <project>/output)"
            echo "  --validation-dir PATH   Validation output directory (default: output/validation)"
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
echo "Amazon ML Challenge 2025 - Model Validation"
echo "============================================"
echo "Config: $CONFIG_PATH"
echo "Output: $OUTPUT_DIR"
echo "Project Root: $PROJECT_ROOT"
echo ""

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Set validation directory
if [ -z "$VALIDATION_DIR" ]; then
    VALIDATION_DIR="$OUTPUT_DIR/validation"
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$VALIDATION_DIR"

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
CKPT_DIR="$OUTPUT_DIR/checkpoints"
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory not found at $CKPT_DIR"
    echo "Please run training first"
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

# Validation command
VALIDATE_CMD="python -c \"
import sys
sys.path.append('src')
from config import Config
from validate import validate_model_performance

# Load config
config = Config.from_yaml('$CONFIG_PATH')

# Override paths
config.output_dir = '$OUTPUT_DIR'

# Run validation
validate_model_performance(config, '$VALIDATION_DIR')
\""

echo "Running validation pipeline..."
echo ""

# Create log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$VALIDATION_DIR/validation_${TIMESTAMP}.log"

# Run validation with logging
if eval "$VALIDATE_CMD" 2>&1 | tee "$LOG_FILE"; then
    echo ""
    echo "============================================"
    echo "Validation completed successfully!"
    echo "============================================"
    echo "Results saved to: $VALIDATION_DIR"
    echo ""
    
    # Show validation report if available
    REPORT_FILE="$VALIDATION_DIR/validation_report.json"
    if [ -f "$REPORT_FILE" ]; then
        echo "Validation Summary:"
        python -c "
import json
with open('$REPORT_FILE', 'r') as f:
    report = json.load(f)

if 'summary' in report:
    summary = report['summary']
    if summary['mean_smape'] is not None:
        print(f'  Mean SMAPE: {summary[\"mean_smape\"]:.2f}% ± {summary[\"std_smape\"]:.2f}%')
        print(f'  Best Fold: {summary[\"best_fold_smape\"]:.2f}%')

if 'cross_validation_analysis' in report:
    cv = report['cross_validation_analysis']
    if cv:
        print(f'  CV Stability: {cv[\"cv_stability\"]:.3f}')

print(f'  Individual fold results: {len(report[\"individual_fold_results\"])} folds')
"
        echo ""
        
        # List generated files
        echo "Generated files:"
        find "$VALIDATION_DIR" -name "*.png" -o -name "*.json" -o -name "*.log" | sort | sed 's/^/  /'
        echo ""
        
    else
        echo "Warning: Validation report not found"
    fi
    
else
    echo ""
    echo "============================================"
    echo "Validation failed!"
    echo "============================================"
    echo "Check log file: $LOG_FILE"
    echo ""
    exit 1
fi