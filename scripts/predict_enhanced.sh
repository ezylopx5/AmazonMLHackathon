#!/bin/bash
# Enhanced prediction script for Amazon ML Challenge 2025

echo "🔮 Starting enhanced prediction..."

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Run predictions using the trained models
python -m src.infer --config configs/config.yaml --output submissions/

echo "✅ Enhanced prediction completed!"
echo "📁 Results saved in submissions/ directory"