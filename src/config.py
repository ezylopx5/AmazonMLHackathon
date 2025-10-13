import os
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Paths
    root: str = "."
    project_dir: str = None
    train_csv: str = None
    test_csv: str = None
    img_train: str = None
    img_test: str = None
    img_val: str = None
    feats_dir: str = None
    ckpt_dir: str = None
    output_dir: str = None
    submission_dir: str = None
    
    # Model
    text_model: str = "mistralai/Mistral-7B-v0.1"
    vision_model: str = "openai/clip-vit-large-patch14"
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    quantization: str = "4bit"
    
    # Training
    seed: int = 42
    folds: int = 5
    epochs: int = 10
    batch_size: int = 16
    gradient_accumulation: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    mixed_precision: bool = True
    
    # Features
    max_text_length: int = 256
    image_size: int = 224
    cache_features: bool = True
    
    # Inference
    ensemble_weights: str = "inverse_smape"
    post_process: bool = True
    tta_augments: int = 0
    
    # System
    num_workers: int = 4
    pin_memory: bool = True
    log_interval: int = 10
    save_best_only: bool = False
    upload_to_hub: bool = False
    hub_repo: Optional[str] = None
    
    def __post_init__(self):
        # Set project directory
        if self.project_dir is None:
            self.project_dir = self.root if self.root != "." else os.getcwd()
        
        # Set default paths
        if self.train_csv is None:
            self.train_csv = os.path.join(self.project_dir, "dataset/train.csv")
        if self.test_csv is None:
            self.test_csv = os.path.join(self.project_dir, "dataset/test.csv")
        if self.img_train is None:
            self.img_train = os.path.join(self.project_dir, "dataset/images/train")
        if self.img_test is None:
            self.img_test = os.path.join(self.project_dir, "dataset/images/test")
        if self.img_val is None:
            self.img_val = os.path.join(self.project_dir, "dataset/images/val")
        if self.feats_dir is None:
            self.feats_dir = os.path.join(self.project_dir, "dataset/features")
        if self.ckpt_dir is None:
            self.ckpt_dir = os.path.join(self.project_dir, "output/checkpoints")
        if self.output_dir is None:
            self.output_dir = os.path.join(self.project_dir, "output")
        if self.submission_dir is None:
            self.submission_dir = os.path.join(self.project_dir, "submissions")
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Get the parent directory of the config file as the project root
        # (config is in configs/ subdirectory, so go up one level)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(path)))
        
        # Lightning AI compatibility: detect and handle Lightning AI environment
        current_dir = os.getcwd()
        is_lightning_ai = '/teamspace/studios/' in current_dir or '/teamspace/studios/' in project_root
        
        if is_lightning_ai:
            # Lightning AI uses /teamspace/studios/this_studio/ as project root
            # Find the correct Lightning AI project root
            cwd_parts = current_dir.split('/')
            for i, part in enumerate(cwd_parts):
                if part == 'teamspace' and i + 2 < len(cwd_parts) and cwd_parts[i + 1] == 'studios':
                    lightning_root = '/'.join(cwd_parts[:i + 3])  # /teamspace/studios/this_studio
                    project_root = lightning_root
                    print(f"🔧 Lightning AI detected - using project root: {project_root}")
                    break
        
        # Flatten nested structure to match dataclass fields
        flat_data = {'project_dir': project_root}
        
        # Map nested YAML to flat dataclass fields
        if 'data' in data:
            # Convert relative paths to absolute paths
            def make_absolute(rel_path):
                if rel_path and not os.path.isabs(rel_path):
                    return os.path.join(project_root, rel_path)
                return rel_path
            
            flat_data.update({
                'train_csv': make_absolute(data['data'].get('train_csv')),
                'test_csv': make_absolute(data['data'].get('test_csv')),
                'img_train': make_absolute(data['data'].get('img_train')),
                'img_test': make_absolute(data['data'].get('img_test')),
                'img_val': make_absolute(data['data'].get('img_val')),
                'feats_dir': make_absolute(data['data'].get('feats_dir')),
                'output_dir': make_absolute(data['data'].get('output_dir')),
                'ckpt_dir': make_absolute(data['data'].get('ckpt_dir')),
                'submission_dir': make_absolute(data['data'].get('submission_dir')),
            })
        
        if 'model' in data:
            flat_data.update({
                'text_model': data['model'].get('text_model'),
                'vision_model': data['model'].get('image_model'),  # Note: YAML uses image_model
                'use_lora': data['model'].get('text_use_lora', True) or data['model'].get('image_use_lora', True),
                'lora_r': data['model'].get('text_lora_r', 16),
                'lora_alpha': data['model'].get('text_lora_alpha', 32),
                'lora_dropout': data['model'].get('text_lora_dropout', 0.05),
                'quantization': '4bit' if data['model'].get('text_use_4bit') else 'none',
            })
        
        if 'training' in data:
            flat_data.update({
                'seed': data['training'].get('fold_seed', 42),
                'folds': data['training'].get('num_folds', 5),
                'epochs': data['training'].get('num_epochs', 10),
                'batch_size': data['training'].get('batch_size', 16),
                'gradient_accumulation': data['training'].get('gradient_accumulation_steps', 2),
                'learning_rate': data['training'].get('learning_rate', 2e-5),
                'weight_decay': data['training'].get('weight_decay', 0.01),
                'warmup_ratio': data['training'].get('warmup_ratio', 0.1),
                'max_grad_norm': data['training'].get('max_grad_norm', 1.0),
                'mixed_precision': data['training'].get('mixed_precision', True),
            })
            
        # Remove None values to use defaults
        flat_data = {k: v for k, v in flat_data.items() if v is not None}
        
        return cls(**flat_data)
    
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)

def get_config(config_path: Optional[str] = None) -> Config:
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return Config()