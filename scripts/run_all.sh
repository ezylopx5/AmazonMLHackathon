#!/bin/bash
# Complete ML pipeline execution script

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_PATH="$PROJECT_DIR/configs/config.yaml"

echo "🚀 Starting Amazon ML Challenge 2025 Pipeline"
echo "================================================"
echo "Project directory: $PROJECT_DIR"
echo "Config path: $CONFIG_PATH"

# Check dependencies
echo "📋 Checking dependencies..."
python -c "import torch, transformers, timm, pandas, numpy, sklearn, scipy, psutil" || {
    echo "❌ Missing dependencies. Run: pip install -r requirements.txt"
    exit 1
}

# Check GPU availability
if python -c "import torch; print('GPU available:', torch.cuda.is_available())"; then
    echo "✅ Dependencies check passed"
else
    echo "⚠️ CUDA not available, using CPU"
fi

# Download images if needed
echo ""
echo "📥 Downloading images..."
cd "$PROJECT_DIR"
python scripts/download_images.py || {
    echo "❌ Image download failed"
    exit 1
}

# Feature extraction
echo ""
echo "🔧 Extracting features..."
python -c "
import sys
sys.path.append('src')
from features import FeatureExtractor
from config import Config

config = Config('$CONFIG_PATH')
extractor = FeatureExtractor(config)
extractor.extract_all_features()
print('✅ Feature extraction completed')
"

# Training with cross-validation
echo ""
echo "🎯 Training models..."
bash scripts/train.sh || {
    echo "❌ Training failed"
    exit 1
}

# Validation
echo ""
echo "🔍 Validating models..."
bash scripts/validate.sh || {
    echo "❌ Validation failed"
    exit 1
}

# Final predictions
echo ""
echo "🔮 Generating predictions..."
bash scripts/predict.sh || {
    echo "❌ Prediction failed"
    exit 1
}

# Validation of submission
echo ""
echo "✅ Validating submission..."
python -c "
import sys
sys.path.append('src')
from infer import validate_submission
from config import Config

config = Config('$CONFIG_PATH')
submission_path = f'{config.submission_dir}/submission.csv'
validate_submission(submission_path)
"

echo ""
echo "🎉 Pipeline completed successfully!"
echo "📁 Check results in: submissions/"
echo "📊 Model checkpoints in: checkpoints/"
echo "📈 Logs in: logs/"