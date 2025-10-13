#!/usr/bin/env python3
"""
Amazon ML Challenge 2025 - Pipeline Controller
Easy-to-use Python script for running the complete ML pipeline
"""

import argparse
import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PipelineController:
    """Control the complete ML pipeline execution"""
    
    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).parent
        self.scripts_dir = self.project_dir / "scripts"
        self.src_dir = self.project_dir / "src"
        
        # Add src to Python path
        sys.path.insert(0, str(self.src_dir))
    
    def run_command(self, command: str, shell: bool = True) -> bool:
        """Run a shell command and return success status"""
        try:
            logger.info(f"🚀 Running: {command}")
            result = subprocess.run(
                command, 
                shell=shell, 
                cwd=self.project_dir,
                check=True,
                capture_output=False
            )
            logger.info("✅ Command completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Command failed with exit code {e.returncode}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        logger.info("📋 Checking dependencies...")
        
        try:
            import torch
            import transformers
            import timm
            import pandas
            import numpy
            import sklearn
            import scipy
            import psutil
            logger.info("✅ All core dependencies available")
            return True
        except ImportError as e:
            logger.error(f"❌ Missing dependency: {e}")
            logger.info("💡 Run: pip install -r requirements.txt")
            return False
    
    def download_images(self) -> bool:
        """Download required images"""
        logger.info("📥 Downloading images...")
        
        # Set up paths for the download script
        train_csv = self.project_dir / "dataset" / "train.csv"
        test_csv = self.project_dir / "dataset" / "test.csv"
        train_output = self.project_dir / "dataset" / "images" / "train"
        test_output = self.project_dir / "dataset" / "images" / "test"
        log_file = self.project_dir / "logs" / "download_images.log"
        
        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(exist_ok=True)
        
        # First attempt with standard settings
        command = (
            f"python scripts/download_images_simple.py "
            f"--train-csv {train_csv} "
            f"--test-csv {test_csv} "
            f"--train-output {train_output} "
            f"--test-output {test_output} "
            f"--num-workers 6 "
            f"--log-file {log_file} "
            f"--skip-existing"
        )
        
        result = self.run_command(command)
        
        # After download, check results and potentially retry
        if result:
            train_count = len(list(train_output.glob("*.jpg"))) if train_output.exists() else 0
            test_count = len(list(test_output.glob("*.jpg"))) if test_output.exists() else 0
            
            logger.info(f"📊 Download Results: {train_count:,} train, {test_count:,} test images")
            
            # Calculate success rates (assuming original CSV has 75,000 each)
            train_success_rate = (train_count / 75000) * 100 if train_count > 0 else 0
            test_success_rate = (test_count / 75000) * 100 if test_count > 0 else 0
            
            logger.info(f"📈 Success Rates: Train {train_success_rate:.1f}%, Test {test_success_rate:.1f}%")
            
            # If test images have very low success rate, try a retry with different settings
            if test_count < 20000:  # Less than ~25% success rate
                logger.warning("⚠️  Low test image success rate detected!")
                logger.info("🔄 Attempting retry with optimized settings...")
                
                # Retry with fewer workers and longer timeout
                retry_command = (
                    f"python scripts/download_images_simple.py "
                    f"--test-csv {test_csv} "
                    f"--test-output {test_output} "
                    f"--num-workers 3 "
                    f"--log-file {log_file} "
                    f"--skip-existing"
                )
                
                retry_result = self.run_command(retry_command)
                
                if retry_result:
                    new_test_count = len(list(test_output.glob("*.jpg"))) if test_output.exists() else 0
                    logger.info(f"� Retry Results: {new_test_count:,} test images")
                    test_count = new_test_count
            
            # Final assessment
            if train_count < 10000 or test_count < 5000:
                logger.error("❌ Very low image count - may not be sufficient for training")
                logger.info("💡 Consider checking internet connection or trying again later")
                return False
            elif train_count < 50000 or test_count < 30000:
                logger.warning("⚠️  Moderate image count - training will continue but performance may be affected")
                logger.info("💡 You can retry download later to get more images")
            else:
                logger.info("✅ Good image count for training")
            
        return result
    
    def check_download_progress(self) -> bool:
        """Check the progress of image download"""
        train_dir = self.project_dir / "dataset" / "images" / "train"
        test_dir = self.project_dir / "dataset" / "images" / "test"
        
        try:
            train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
            test_count = len(list(test_dir.glob("*.jpg"))) if test_dir.exists() else 0
            
            logger.info("📊 Download Progress:")
            logger.info(f"   🔸 Training images: {train_count:,}/75,000 ({train_count/75000*100:.1f}%)")
            logger.info(f"   🔸 Test images: {test_count:,}/75,000 ({test_count/75000*100:.1f}%)")
            logger.info(f"   🔸 Total downloaded: {train_count + test_count:,}/150,000")
            
            # Check if download is complete
            if train_count >= 75000 and test_count >= 75000:
                logger.info("✅ Download complete! Ready for training.")
                return True
            elif train_count + test_count > 0:
                logger.info("🔄 Download in progress...")
                estimated_remaining = (150000 - train_count - test_count) / 2  # Assuming 2 images/sec
                logger.info(f"⏱️  Estimated time remaining: {estimated_remaining/60:.0f} minutes")
                return True
            else:
                logger.info("❌ No images found. Download may not have started.")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error checking download progress: {e}")
            return False
    
    def has_sufficient_images(self, min_train: int = 1000, min_test: int = 1000) -> bool:
        """Check if we have sufficient images to proceed"""
        train_dir = self.project_dir / "dataset" / "images" / "train"
        test_dir = self.project_dir / "dataset" / "images" / "test"
        
        train_count = len(list(train_dir.glob("*.jpg"))) if train_dir.exists() else 0
        test_count = len(list(test_dir.glob("*.jpg"))) if test_dir.exists() else 0
        
        if train_count >= min_train and test_count >= min_test:
            logger.info(f"✅ Sufficient images available: {train_count:,} train, {test_count:,} test")
            return True
        else:
            logger.info(f"⚠️  Insufficient images: {train_count:,} train, {test_count:,} test")
            logger.info(f"   Required: {min_train:,} train, {min_test:,} test")
            return False
    
    def retry_download(self) -> bool:
        """Retry download with optimized settings for failed images"""
        logger.info("🔄 Retrying download with optimized settings...")
        
        # Set up paths
        train_csv = self.project_dir / "dataset" / "train.csv"
        test_csv = self.project_dir / "dataset" / "test.csv"
        train_output = self.project_dir / "dataset" / "images" / "train"
        test_output = self.project_dir / "dataset" / "images" / "test"
        log_file = self.project_dir / "logs" / "retry_download.log"
        
        # Create logs directory if it doesn't exist
        log_file.parent.mkdir(exist_ok=True)
        
        # Retry with conservative settings
        retry_command = (
            f"python scripts/download_images_simple.py "
            f"--train-csv {train_csv} "
            f"--test-csv {test_csv} "
            f"--train-output {train_output} "
            f"--test-output {test_output} "
            f"--num-workers 3 "
            f"--log-file {log_file} "
            f"--skip-existing"
        )
        
        result = self.run_command(retry_command)
        
        if result:
            train_count = len(list(train_output.glob("*.jpg"))) if train_output.exists() else 0
            test_count = len(list(test_output.glob("*.jpg"))) if test_output.exists() else 0
            
            logger.info(f"📊 Final Results: {train_count:,} train, {test_count:,} test images")
            
        return result
    
    def extract_features(self) -> bool:
        """Extract features from images and text"""
        logger.info("🔧 Extracting features...")
        
        # Check if we have sufficient images before feature extraction
        if not self.has_sufficient_images(min_train=100, min_test=100):
            logger.error("❌ Not enough images for feature extraction")
            logger.info("💡 Download more images first: python pipeline.py download")
            return False
        
        try:
            from features import build_feature_cache
            from config import Config
            
            # Use correct path with project directory
            config_path = os.path.join(self.project_dir, "configs", "config.yaml")
            config = Config.from_yaml(config_path)
            logger.info("🔄 Starting feature cache build...")
            build_feature_cache(config)
            logger.info("✅ Feature extraction completed")
            return True
        except Exception as e:
            logger.error(f"❌ Feature extraction failed: {e}")
            logger.error("💡 Try running individually: python pipeline.py features")
            return False
    
    def train_models(self, quick: bool = False) -> bool:
        """Train models with cross-validation"""
        logger.info("🎯 Training models...")
        
        if quick:
            return self.run_command("bash scripts/quick_baseline.sh")
        else:
            return self.run_command("bash scripts/train.sh")
    
    def validate_models(self) -> bool:
        """Validate trained models"""
        logger.info("🔍 Validating models...")
        return self.run_command("bash scripts/validate.sh")
    
    def generate_predictions(self, enhanced: bool = True) -> bool:
        """Generate final predictions"""
        logger.info("🔮 Generating predictions...")
        
        script = "predict_enhanced.sh" if enhanced else "predict.sh"
        return self.run_command(f"bash scripts/{script}")
    
    def run_monitoring(self, background: bool = True) -> bool:
        """Start system monitoring"""
        logger.info("📊 Starting system monitoring...")
        
        if background:
            return self.run_command("bash scripts/monitor.sh &")
        else:
            return self.run_command("bash scripts/monitor.sh")
    
    def validate_submission(self) -> bool:
        """Validate the final submission"""
        logger.info("✅ Validating submission...")
        
        try:
            from infer import validate_submission
            
            submission_path = self.project_dir / "submissions" / "submission.csv"
            return validate_submission(str(submission_path))
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            return False
    
    def run_complete_pipeline(self, quick: bool = False, monitoring: bool = True) -> bool:
        """Run the complete ML pipeline"""
        logger.info("🚀 Starting complete Amazon ML Challenge 2025 pipeline")
        logger.info("=" * 60)
        
        steps = [
            ("Check Dependencies", self.check_dependencies),
        ]
        
        # Check if we have sufficient images, if not, download automatically
        if not self.has_sufficient_images(min_train=1000, min_test=1000):
            logger.info("🔍 Insufficient images detected, starting automatic download...")
            steps.append(("Download Images", self.download_images))
        else:
            logger.info("✅ Sufficient images found, skipping download")
        
        steps.extend([
            ("Extract Features", self.extract_features),
            ("Train Models", lambda: self.train_models(quick=quick)),
        ])
        
        if not quick:
            steps.append(("Validate Models", self.validate_models))
        
        steps.extend([
            ("Generate Predictions", lambda: self.generate_predictions(enhanced=not quick)),
            ("Validate Submission", self.validate_submission)
        ])
        
        # Start monitoring if requested
        if monitoring and not quick:
            self.run_monitoring(background=True)
        
        # Execute all steps
        for step_name, step_func in steps:
            logger.info(f"\n🔄 Step: {step_name}")
            logger.info("-" * 40)
            
            if not step_func():
                logger.error(f"❌ Pipeline failed at step: {step_name}")
                return False
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📁 Check results in:")
        logger.info("   • submissions/submission.csv")
        logger.info("   • checkpoints/")
        logger.info("   • logs/")
        
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2025 Pipeline Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ONE COMMAND DOES EVERYTHING (RECOMMENDED):
  python pipeline.py                    # Runs complete pipeline: download→features→train→predict
  python pipeline.py complete           # Same as above (explicit)
  python pipeline.py complete --quick   # Fast baseline mode
  
  # INDIVIDUAL STEPS (if needed):
  python pipeline.py download           # Download images only
  python pipeline.py retry-download     # Retry download with optimized settings (for low success rates)
  python pipeline.py features           # Extract features only
  python pipeline.py train              # Train models only
  python pipeline.py predict            # Generate predictions only
        """
    )
    
    parser.add_argument(
        'action',
        nargs='?',
        default='complete',
        choices=['complete', 'download', 'retry-download', 'features', 'train', 'validate', 'predict', 'monitor', 'check'],
        help='Action to perform (default: complete - runs everything in one go)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick/baseline settings (faster but lower quality)'
    )
    
    parser.add_argument(
        '--no-monitoring',
        action='store_true',
        help='Disable system monitoring during training'
    )
    
    parser.add_argument(
        '--project-dir',
        type=str,
        help='Project directory path (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Initialize controller
    controller = PipelineController(args.project_dir)
    
    # Execute requested action
    success = False
    
    if args.action == 'complete':
        success = controller.run_complete_pipeline(
            quick=args.quick,
            monitoring=not args.no_monitoring
        )
    elif args.action == 'download':
        success = controller.download_images()
    elif args.action == 'retry-download':
        success = controller.retry_download()
    elif args.action == 'features':
        success = controller.extract_features()
    elif args.action == 'train':
        success = controller.train_models(quick=args.quick)
    elif args.action == 'validate':
        success = controller.validate_models()
    elif args.action == 'predict':
        success = controller.generate_predictions(enhanced=not args.quick)
    elif args.action == 'monitor':
        success = controller.run_monitoring(background=False)
    elif args.action == 'check':
        success = controller.check_dependencies()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()