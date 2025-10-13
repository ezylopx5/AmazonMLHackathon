#!/usr/bin/env python3
"""
Pre-flight check script for Amazon ML Challenge 2025 Pipeline
Verifies all prerequisites before running the main pipeline
"""

import os
import sys
import subprocess
import pandas as pd
from pathlib import Path
import shutil

class PreFlightChecker:
    """Comprehensive pre-flight checks"""
    
    def __init__(self, project_dir: str = None):
        self.project_dir = Path(project_dir) if project_dir else Path(__file__).parent
        self.errors = []
        self.warnings = []
        self.success_count = 0
        self.total_checks = 0
    
    def check_step(self, name: str, check_func):
        """Execute a check step and track results"""
        self.total_checks += 1
        print(f"\n{'='*50}")
        print(f"🔍 {name}")
        print('='*50)
        
        try:
            result = check_func()
            if result:
                print(f"✅ {name} - PASSED")
                self.success_count += 1
                return True
            else:
                print(f"❌ {name} - FAILED")
                self.errors.append(name)
                return False
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            self.errors.append(f"{name}: {e}")
            return False
    
    def check_dependencies(self) -> bool:
        """Check Python dependencies"""
        # Map package names to import names
        package_imports = {
            'torch': 'torch',
            'transformers': 'transformers', 
            'timm': 'timm',
            'pandas': 'pandas',
            'numpy': 'numpy',
            'sklearn': 'sklearn',
            'scipy': 'scipy',
            'psutil': 'psutil',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm'
        }
        
        missing = []
        for package, import_name in package_imports.items():
            try:
                __import__(import_name)
                print(f"✓ {package}")
            except ImportError:
                # Double check with pip list to see if it's installed but not importable
                try:
                    import subprocess
                    result = subprocess.run(['pip', 'show', package], 
                                          capture_output=True, text=True, check=False)
                    if result.returncode == 0:
                        print(f"⚠ {package} - INSTALLED but import failed")
                    else:
                        missing.append(package)
                        print(f"✗ {package} - MISSING")
                except:
                    missing.append(package)
                    print(f"✗ {package} - MISSING")
        
        if missing:
            print(f"\n💡 Install missing packages: pip install {' '.join(missing)}")
            return False
        
        return True
    
    def check_cuda(self) -> bool:
        """Check CUDA availability"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                print(f"✓ CUDA Available: {cuda_available}")
                print(f"✓ GPU Count: {gpu_count}")
                print(f"✓ GPU Name: {gpu_name}")
                
                # Check for A100 specifically
                if "A100" in gpu_name:
                    print(f"🚀 Excellent! A100 GPU detected - perfect for training!")
                
                # Check GPU memory
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
                
                return True
            else:
                print("⚠ CUDA not available - will use CPU (slower)")
                self.warnings.append("CUDA not available")
                return True  # Not critical, can run on CPU
        except Exception as e:
            print(f"✗ CUDA check failed: {e}")
            return False
    
    def check_directories(self) -> bool:
        """Check and create required directories"""
        required_dirs = [
            'dataset/images/train',
            'dataset/images/test', 
            'dataset/images/val',
            'dataset/features',
            'output',
            'output/checkpoints',
            'output/logs',
            'submissions',
            'logs'
        ]
        
        for dir_path in required_dirs:
            full_path = self.project_dir / dir_path
            if not full_path.exists():
                print(f"📁 Creating directory: {dir_path}")
                full_path.mkdir(parents=True, exist_ok=True)
            else:
                print(f"✓ Directory exists: {dir_path}")
        
        return True
    
    def check_data_files(self) -> bool:
        """Check required data files"""
        required_files = ['dataset/train.csv', 'dataset/test.csv']
        
        for file_path in required_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                print(f"✗ Missing file: {file_path}")
                return False
            
            # Check CSV structure
            try:
                df = pd.read_csv(full_path)
                print(f"✓ {file_path}: {len(df)} rows, {len(df.columns)} columns")
                
                if 'train.csv' in file_path:
                    required_cols = ['sample_id', 'catalog_content', 'image_link', 'price']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    if missing_cols:
                        print(f"✗ Missing columns in train.csv: {missing_cols}")
                        return False
                    print(f"✓ Required columns present: {required_cols}")
                
            except Exception as e:
                print(f"✗ Error reading {file_path}: {e}")
                return False
        
        return True
    
    def check_images(self) -> bool:
        """Check if images are downloaded"""
        train_img_dir = self.project_dir / 'dataset/images/train'
        test_img_dir = self.project_dir / 'dataset/images/test'
        
        train_count = len(list(train_img_dir.glob('*'))) if train_img_dir.exists() else 0
        test_count = len(list(test_img_dir.glob('*'))) if test_img_dir.exists() else 0
        
        print(f"📸 Training images: {train_count}")
        print(f"📸 Test images: {test_count}")
        
        if train_count == 0 or test_count == 0:
            print("ℹ️  No images found - pipeline will download them automatically")
            print("💡 Images will be downloaded when pipeline runs: python pipeline.py complete")
            self.warnings.append("Images will be downloaded during pipeline execution")
            # Return True instead of False - this is not a blocking error
            return True
        else:
            print("✓ Images already downloaded and ready")
        
        return True
    
    def check_disk_space(self) -> bool:
        """Check available disk space"""
        try:
            total, used, free = shutil.disk_usage(self.project_dir)
            free_gb = free // (1024**3)
            print(f"💾 Free disk space: {free_gb} GB")
            
            if free_gb < 5:
                print("⚠ Low disk space! Need at least 5GB for images and models")
                self.warnings.append("Low disk space")
                return False
            
            return True
        except Exception as e:
            print(f"✗ Disk space check failed: {e}")
            return True  # Not critical
    
    def check_config_file(self) -> bool:
        """Check configuration file"""
        config_path = self.project_dir / 'configs/config.yaml'
        
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        print(f"✓ Config file exists: {config_path}")
        
        # Check if config is readable
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print("✓ Config file is valid YAML")
            return True
        except ImportError:
            print("⚠ PyYAML not installed, cannot validate config")
            return True
        except Exception as e:
            print(f"✗ Config file error: {e}")
            return False
    
    def check_scripts(self) -> bool:
        """Check if required scripts exist"""
        required_scripts = [
            'scripts/download_images.py',
            'scripts/train.sh',
            'scripts/predict_enhanced.sh',
            'src/config.py',
            'src/data.py',
            'src/models.py',
            'src/train.py',
            'src/infer.py'
        ]
        
        for script in required_scripts:
            script_path = self.project_dir / script
            if not script_path.exists():
                print(f"✗ Missing script: {script}")
                return False
            print(f"✓ Script exists: {script}")
        
        return True
    
    def run_all_checks(self) -> bool:
        """Run comprehensive pre-flight checks"""
        print("🚀 Amazon ML Challenge 2025 - Pre-Flight Check")
        print("=" * 60)
        
        checks = [
            ("Python Dependencies", self.check_dependencies),
            ("CUDA/GPU Setup", self.check_cuda),
            ("Directory Structure", self.check_directories),
            ("Data Files", self.check_data_files),
            ("Configuration File", self.check_config_file),
            ("Required Scripts", self.check_scripts),
            ("Product Images", self.check_images),
            ("Disk Space", self.check_disk_space),
        ]
        
        for check_name, check_func in checks:
            self.check_step(check_name, check_func)
        
        # Final summary
        print("\n" + "=" * 60)
        print("📋 PRE-FLIGHT CHECK SUMMARY")
        print("=" * 60)
        
        print(f"✅ Passed: {self.success_count}/{self.total_checks}")
        
        if self.errors:
            print(f"❌ Failed: {len(self.errors)}")
            print("Errors that must be fixed:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print(f"⚠ Warnings: {len(self.warnings)}")  
            print("Issues to consider:")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        all_passed = len(self.errors) == 0
        
        if all_passed:
            print("\n🎉 ALL CHECKS PASSED! Ready to run pipeline.")
            print("\n🚀 Next steps:")
            print("   python pipeline.py complete        # Full pipeline")
            print("   python pipeline.py complete --quick # Quick test")
        else:
            print("\n🚨 FIX ERRORS BEFORE RUNNING PIPELINE!")
            print("\n💡 Common fixes:")
            print("   pip install -r requirements.txt")
            print("   # Images will be downloaded automatically during pipeline execution")
        
        return all_passed

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-flight check for Amazon ML pipeline")
    parser.add_argument('--project-dir', help='Project directory path')
    args = parser.parse_args()
    
    checker = PreFlightChecker(args.project_dir)
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()