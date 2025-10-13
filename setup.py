"""
Setup script for Amazon ML Challenge 2025
Multi-modal Price Prediction System
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
def read_requirements(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        requirements = []
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Remove version constraints for basic setup
                if '>=' in line:
                    package = line.split('>=')[0]
                    requirements.append(package)
                elif '==' in line:
                    package = line.split('==')[0]  
                    requirements.append(package)
                else:
                    requirements.append(line)
        return requirements

# Core requirements (essential for basic functionality)
install_requires = [
    'torch>=2.0.0',
    'torchvision>=0.15.0', 
    'transformers>=4.30.0',
    'accelerate>=0.20.0',
    'peft>=0.4.0',
    'pandas>=2.0.0',
    'numpy>=1.24.0',
    'scikit-learn>=1.3.0',
    'scipy>=1.10.0',
    'Pillow>=9.5.0',
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'tqdm>=4.65.0',
    'PyYAML>=6.0',
    'requests>=2.31.0',
    'pyarrow>=12.0.0'
]

# Optional extras
extras_require = {
    'gpu': [
        'bitsandbytes>=0.41.0',
        'gpustat>=1.1.0',
        'py3nvml>=0.2.7'
    ],
    'wandb': [
        'wandb>=0.15.0'
    ],
    'optimization': [
        'optimum>=1.12.0',
        'onnx>=1.14.0', 
        'onnxruntime>=1.15.0',
        'optuna>=3.3.0'
    ],
    'development': [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'black>=23.0.0',
        'flake8>=6.0.0',
        'isort>=5.12.0',
        'jupyter>=1.0.0',
        'notebook>=6.5.0'
    ],
    'serving': [
        'fastapi>=0.100.0',
        'uvicorn>=0.23.0'
    ],
    'distributed': [
        'deepspeed>=0.10.0',
        'ray[tune]>=2.6.0'
    ],
    'nlp': [
        'nltk>=3.8.0',
        'textblob>=0.17.0',
        'wordcloud>=1.9.0'
    ],
    'vision': [
        'opencv-python>=4.7.0'
    ]
}

# All extras combined
extras_require['all'] = list(set(sum(extras_require.values(), [])))

setup(
    name="amazon-ml-2025",
    version="1.0.0",
    author="Amazon ML Challenge Team",
    author_email="",
    description="Multi-modal Price Prediction System for Amazon ML Challenge 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/amazon-ml-2025",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    entry_points={
        "console_scripts": [
            "amazon-ml-train=src.train:main",
            "amazon-ml-predict=src.infer:main", 
            "amazon-ml-validate=src.validate:main",
            "amazon-ml-download=scripts.download_images:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/your-username/amazon-ml-2025/issues",
        "Source": "https://github.com/your-username/amazon-ml-2025",
        "Documentation": "https://github.com/your-username/amazon-ml-2025/wiki",
    },
    keywords=[
        "machine learning",
        "deep learning", 
        "computer vision",
        "natural language processing",
        "multimodal",
        "price prediction",
        "e-commerce",
        "pytorch",
        "transformers",
        "amazon ml challenge"
    ],
    zip_safe=False,
)

# Post-install message
print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    Amazon ML Challenge 2025 - Setup Complete                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🚀 Installation successful! Next steps:                                     ║
║                                                                              ║
║  1. Download data:                                                           ║
║     • Place train.csv and test.csv in dataset/ folder                       ║
║     • Run: amazon-ml-download --train-csv dataset/train.csv \\               ║
║              --test-csv dataset/test.csv \\                                 ║
║              --train-output dataset/images/train \\                         ║
║              --test-output dataset/images/test                              ║
║                                                                              ║
║  2. Configure training:                                                      ║
║     • Edit configs/config.yaml as needed                                    ║
║                                                                              ║
║  3. Train model:                                                             ║
║     • Run: amazon-ml-train --config configs/config.yaml                     ║
║     • Or use: bash scripts/train.sh                                         ║
║                                                                              ║
║  4. Generate predictions:                                                    ║
║     • Run: amazon-ml-predict --config configs/config.yaml                   ║
║     • Or use: bash scripts/predict.sh                                       ║
║                                                                              ║
║  5. Validate model:                                                          ║
║     • Run: amazon-ml-validate --config configs/config.yaml                  ║
║     • Or use: bash scripts/validate.sh                                      ║
║                                                                              ║
║  📖 For detailed documentation, see README.md                               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")