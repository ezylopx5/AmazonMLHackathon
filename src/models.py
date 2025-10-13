import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    CLIPVisionModel, CLIPImageProcessor,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
import logging
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

class MistralTextEncoder:
    """Mistral-7B text encoder with 4-bit quantization and LoRA"""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-v0.1", 
                 device: str = 'cuda',
                 use_lora: bool = True,
                 lora_r: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.05):
        
        self.device = device
        self.model_name = model_name
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        logger.info(f"Loading {model_name} with 4-bit quantization...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        if use_lora:
            # Apply LoRA for efficient fine-tuning
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )
            
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        
        self.hidden_size = 4096  # Mistral-7B hidden size
    
    def encode(self, texts: List[str], max_length: int = 256) -> torch.Tensor:
        """Encode texts to embeddings"""
        # Prepare prompts for better understanding
        prompts = [f"Product description: {text}\nAnalysis:" for text in texts]
        
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.cuda.amp.autocast():
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                output_hidden_states=True
            )
        
        # Use last hidden state of last token as embedding
        hidden_states = outputs.hidden_states[-1]
        
        # Get last non-padding token for each sequence
        sequence_lengths = inputs.attention_mask.sum(dim=1) - 1
        batch_size = hidden_states.shape[0]
        
        embeddings = hidden_states[range(batch_size), sequence_lengths]
        
        return embeddings

class CLIPImageEncoder:
    """CLIP vision encoder for image understanding"""
    
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14",
                 device: str = 'cuda'):
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading {model_name}...")
        self.model = CLIPVisionModel.from_pretrained(model_name).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        
        # Freeze CLIP
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        self.hidden_size = 1024  # CLIP-ViT-L projection dim
    
    @torch.no_grad()
    def encode(self, images) -> torch.Tensor:
        """Encode images to embeddings"""
        # Process images
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        
        # Get embeddings
        outputs = self.model(**inputs)
        embeddings = outputs.pooler_output
        
        return embeddings

class MultiModalFusionModel(nn.Module):
    """Advanced multi-modal fusion model"""
    
    def __init__(self, 
                 text_dim: int = 4096,
                 image_dim: int = 1024,
                 num_features: int = 100,
                 hidden_dims: List[int] = [2048, 1024, 512, 256],
                 dropout_rate: float = 0.3,
                 use_attention: bool = True):
        
        super().__init__()
        
        self.text_dim = text_dim
        self.image_dim = image_dim
        self.num_features = num_features
        self.use_attention = use_attention
        
        # Feature projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dims[0] // 2),
            nn.LayerNorm(hidden_dims[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dims[0] // 4),
            nn.LayerNorm(hidden_dims[0] // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        self.feature_proj = nn.Sequential(
            nn.Linear(num_features, hidden_dims[0] // 4),
            nn.LayerNorm(hidden_dims[0] // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.5)
        )
        
        # Cross-modal attention (optional)
        if use_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[0] // 4,
                num_heads=8,
                dropout=dropout_rate * 0.5,
                batch_first=True
            )
        
        # Fusion network
        fusion_layers = []
        input_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:]):
            fusion_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate * (0.9 - i * 0.1))
            ])
            input_dim = hidden_dim
        
        self.fusion = nn.Sequential(*fusion_layers)
        
        # Multiple regression heads for ensemble
        self.price_head_main = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        self.price_head_aux = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.price_head_simple = nn.Linear(hidden_dims[-1], 1)
        
        # Learnable ensemble weights
        self.head_weights = nn.Parameter(torch.tensor([0.5, 0.3, 0.2]))
        
        # Residual connection from features to output
        self.residual_proj = nn.Linear(num_features, 1)
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, text_emb: torch.Tensor, 
                image_emb: torch.Tensor, 
                features: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        
        # Project modalities
        text_proj = self.text_proj(text_emb)
        image_proj = self.image_proj(image_emb)
        feat_proj = self.feature_proj(features)
        
        # Optional cross-attention between modalities
        if self.use_attention:
            # Reshape for attention
            feat_proj_attn = feat_proj.unsqueeze(1)
            image_proj_attn = image_proj.unsqueeze(1)
            
            # Cross-attend features with image
            feat_attended, _ = self.cross_attention(
                feat_proj_attn, image_proj_attn, image_proj_attn
            )
            feat_proj = feat_attended.squeeze(1)
        
        # Concatenate all projections
        combined = torch.cat([text_proj, image_proj, feat_proj], dim=1)
        
        # Pass through fusion network
        fused = self.fusion(combined)
        
        # Get predictions from multiple heads
        price_main = self.price_head_main(fused)
        price_aux = self.price_head_aux(fused)
        price_simple = self.price_head_simple(fused)
        
        # Weighted ensemble of heads
        weights = F.softmax(self.head_weights, dim=0)
        price = (weights[0] * price_main + 
                weights[1] * price_aux + 
                weights[2] * price_simple)
        
        # Add residual connection from raw features
        residual = self.residual_proj(features)
        price = price + self.residual_weight * residual
        
        # Ensure positive prices with softplus
        price = F.softplus(price).squeeze(-1)
        
        return price

class SimpleTabularModel(nn.Module):
    """Simple MLP for tabular features baseline"""
    
    def __init__(self, num_features: int, 
                 hidden_dims: List[int] = [512, 256, 128],
                 dropout_rate: float = 0.3):
        
        super().__init__()
        
        layers = []
        input_dim = num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.model(features)
        return F.softplus(output).squeeze(-1)

def create_model(config, num_features: int):
    """Factory function to create models"""
    
    if config.text_model and config.vision_model:
        # Full multi-modal model
        text_encoder = MistralTextEncoder(
            model_name=config.text_model,
            use_lora=config.use_lora,
            lora_r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout
        )
        
        image_encoder = CLIPImageEncoder(
            model_name=config.vision_model
        )
        
        model = MultiModalFusionModel(
            text_dim=text_encoder.hidden_size,
            image_dim=image_encoder.hidden_size,
            num_features=num_features,
            dropout_rate=0.3
        )
        
        return model, text_encoder, image_encoder
    
    else:
        # Tabular only baseline
        model = SimpleTabularModel(num_features=num_features)
        return model, None, None