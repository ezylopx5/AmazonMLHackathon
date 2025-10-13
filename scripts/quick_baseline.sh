#!/bin/bash
# Quick baseline model for rapid prototyping

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "⚡ Quick Baseline Pipeline"
echo "========================="

cd "$PROJECT_DIR"

# Quick feature extraction (reduced set)
echo "🔧 Quick feature extraction..."
python -c "
import sys, os
sys.path.append('src')
from features import FeatureExtractor
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

config = Config('configs/config.yaml')

# Override for quick run
config.text_max_length = 128  # Reduced from 512
config.img_size = 224         # Reduced from 384
config.num_epochs = 2         # Reduced from 10
config.batch_size = 16        # Reduced batch size

extractor = FeatureExtractor(config)
extractor.extract_all_features()
print('✅ Quick features extracted')
"

# Quick training (1 fold only)
echo "🎯 Quick training..."
python -c "
import sys
sys.path.append('src')
from train import train_single_fold
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

config = Config('configs/config.yaml')
config.num_epochs = 2
config.batch_size = 16

# Train only fold 0 for quick baseline
print('Training fold 0 only...')
val_smape = train_single_fold(config, fold=0)
print(f'✅ Quick training completed. Validation SMAPE: {val_smape:.4f}')
"

# Quick prediction
echo "🔮 Quick prediction..."
python -c "
import sys, os
sys.path.append('src')
from infer import generate_predictions
from config import Config
import logging

logging.basicConfig(level=logging.INFO)

config = Config('configs/config.yaml')
config.post_process = False  # Skip post-processing for speed

submission_df = generate_predictions(config)
print(f'✅ Quick predictions generated: {len(submission_df)} samples')
"

echo ""
echo "⚡ Quick baseline completed!"
echo "📁 Results in: submissions/submission.csv"
echo "⏱️ This was a quick run - use run_all.sh for full pipeline"