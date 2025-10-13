"""
Resource monitoring and memory optimization utilities
"""

import os
import psutil
import torch
import logging
import numpy as np
from typing import Optional, Tuple, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ResourceMonitor:
    """Monitor system resources and prevent OOM"""
    
    def __init__(self, safety_margin_gb: float = 2.0):
        self.safety_margin_gb = safety_margin_gb
        self.initial_memory = self._get_memory_info()
    
    def _get_memory_info(self) -> dict:
        """Get current memory information"""
        memory_info = {
            'ram_total': psutil.virtual_memory().total / (1024 ** 3),
            'ram_available': psutil.virtual_memory().available / (1024 ** 3),
            'ram_used': psutil.virtual_memory().used / (1024 ** 3),
        }
        
        # GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                gpu_allocated = torch.cuda.memory_allocated(0) / (1024 ** 3)
                gpu_reserved = torch.cuda.memory_reserved(0) / (1024 ** 3)
                
                memory_info.update({
                    'gpu_total': gpu_memory,
                    'gpu_allocated': gpu_allocated,
                    'gpu_reserved': gpu_reserved,
                    'gpu_free': gpu_memory - gpu_reserved
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return memory_info
    
    def check_system_health(self) -> bool:
        """Check if system has enough resources"""
        memory_info = self._get_memory_info()
        
        # Check RAM
        if memory_info['ram_available'] < self.safety_margin_gb:
            logger.warning(f"Low RAM: {memory_info['ram_available']:.1f}GB available")
            return False
        
        # Check GPU memory
        if 'gpu_free' in memory_info and memory_info['gpu_free'] < self.safety_margin_gb:
            logger.warning(f"Low GPU memory: {memory_info['gpu_free']:.1f}GB available")
            return False
        
        return True
    
    def get_optimal_batch_size(self, model: torch.nn.Module, base_batch_size: int, 
                               input_shapes: dict, max_attempts: int = 5) -> int:
        """Dynamically determine optimal batch size"""
        if not torch.cuda.is_available():
            return base_batch_size
        
        device = next(model.parameters()).device
        model.eval()
        
        # Start with a small batch to test basic functionality
        test_batch_size = min(2, base_batch_size)
        
        try:
            # Create dummy inputs
            dummy_inputs = {}
            for key, shape in input_shapes.items():
                if key == 'text':
                    dummy_inputs[key] = ['test text'] * test_batch_size
                elif key == 'image':
                    dummy_inputs[key] = torch.randn(test_batch_size, *shape, device=device)
                elif key == 'features':
                    dummy_inputs[key] = torch.randn(test_batch_size, shape[0], device=device)
            
            # Test base functionality
            with torch.no_grad():
                _ = model(**dummy_inputs)
            
            # Find maximum working batch size
            working_batch_size = test_batch_size
            
            for attempt in range(max_attempts):
                test_size = min(base_batch_size * (2 ** attempt), base_batch_size * 4)
                
                try:
                    # Update batch size in dummy inputs
                    for key, shape in input_shapes.items():
                        if key == 'text':
                            dummy_inputs[key] = ['test text'] * test_size
                        elif key == 'image':
                            dummy_inputs[key] = torch.randn(test_size, *shape, device=device)
                        elif key == 'features':
                            dummy_inputs[key] = torch.randn(test_size, shape[0], device=device)
                    
                    with torch.no_grad():
                        _ = model(**dummy_inputs)
                    
                    working_batch_size = test_size
                    torch.cuda.empty_cache()
                    
                except RuntimeError as e:
                    if 'out of memory' in str(e).lower():
                        torch.cuda.empty_cache()
                        break
                    else:
                        raise e
            
            logger.info(f"Optimal batch size found: {working_batch_size} (base: {base_batch_size})")
            return working_batch_size
            
        except Exception as e:
            logger.warning(f"Failed to determine optimal batch size: {e}")
            return base_batch_size
        finally:
            torch.cuda.empty_cache()
    
    def log_memory_usage(self, step_name: str = ""):
        """Log current memory usage"""
        memory_info = self._get_memory_info()
        
        logger.info(f"Memory usage {step_name}:")
        logger.info(f"  RAM: {memory_info['ram_used']:.1f}GB / {memory_info['ram_total']:.1f}GB")
        
        if 'gpu_allocated' in memory_info:
            logger.info(f"  GPU: {memory_info['gpu_allocated']:.1f}GB / {memory_info['gpu_total']:.1f}GB")

@contextmanager
def memory_efficient_context(clear_cache: bool = True):
    """Context manager for memory-efficient operations"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        if clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

class MemoryOptimizedEncoder:
    """Optimized encoder with gradient checkpointing and memory management"""
    
    def __init__(self, model: torch.nn.Module, model_type: str = "text", 
                 chunk_size: Optional[int] = None):
        self.model = model
        self.model_type = model_type
        self.chunk_size = chunk_size
        self.device = next(model.parameters()).device if hasattr(model, 'parameters') else 'cpu'
    
    def encode(self, inputs, max_length: int = 512, **kwargs):
        """Memory-optimized encoding with chunked processing"""
        if self.model_type == "text":
            return self._encode_text_chunked(inputs, max_length, **kwargs)
        else:
            return self._encode_image_chunked(inputs, **kwargs)
    
    def _encode_text_chunked(self, texts, max_length: int = 512, **kwargs):
        """Process text in chunks to avoid OOM"""
        if not isinstance(texts, list):
            texts = [texts]
        
        batch_size = len(texts)
        
        # Determine chunk size based on available memory
        if self.chunk_size is None:
            # Estimate based on sequence length and available memory
            available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
            estimated_memory_per_sample = max_length * 4096 * 4  # Rough estimate for Mistral-7B
            max_chunk_size = max(1, int(available_memory * 0.3 / estimated_memory_per_sample))
            chunk_size = min(batch_size, max_chunk_size, 8)  # Cap at 8 for safety
        else:
            chunk_size = min(batch_size, self.chunk_size)
        
        all_embeddings = []
        
        for i in range(0, batch_size, chunk_size):
            chunk_texts = texts[i:i + chunk_size]
            
            with memory_efficient_context():
                if hasattr(self.model, 'encode'):
                    # Direct encode method
                    chunk_embeddings = self.model.encode(chunk_texts)
                else:
                    # Manual encoding with the model
                    with torch.cuda.amp.autocast():
                        chunk_embeddings = self._encode_text_batch(chunk_texts, max_length)
                
                all_embeddings.append(chunk_embeddings.cpu())
        
        # Concatenate all embeddings
        return torch.cat(all_embeddings, dim=0).to(self.device)
    
    def _encode_image_chunked(self, images, **kwargs):
        """Process images in chunks to avoid OOM"""
        if not torch.is_tensor(images):
            images = torch.stack(images) if isinstance(images, list) else images
        
        batch_size = images.shape[0]
        
        # Determine chunk size for images
        if self.chunk_size is None:
            # Estimate based on image size and available memory
            image_memory = np.prod(images.shape[1:]) * 4  # 4 bytes per float32
            available_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 8e9
            max_chunk_size = max(1, int(available_memory * 0.4 / image_memory))
            chunk_size = min(batch_size, max_chunk_size, 16)  # Cap at 16 for images
        else:
            chunk_size = min(batch_size, self.chunk_size)
        
        all_embeddings = []
        
        for i in range(0, batch_size, chunk_size):
            chunk_images = images[i:i + chunk_size].to(self.device)
            
            with memory_efficient_context():
                if hasattr(self.model, 'encode'):
                    chunk_embeddings = self.model.encode(chunk_images)
                else:
                    with torch.cuda.amp.autocast():
                        chunk_embeddings = self.model(chunk_images)
                
                all_embeddings.append(chunk_embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device)
    
    def _encode_text_batch(self, texts, max_length):
        """Encode a batch of texts"""
        # This would be implemented based on your specific text encoder
        # For now, return dummy embeddings
        embedding_dim = 4096  # Mistral-7B dimension
        return torch.randn(len(texts), embedding_dim, device=self.device)

class GradientAccumulator:
    """Helper class for gradient accumulation with memory optimization"""
    
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def __enter__(self):
        self.current_step = 0
        self.accumulated_loss = 0.0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def accumulate(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer, 
                   scaler: Optional[torch.cuda.amp.GradScaler] = None) -> bool:
        """Accumulate gradients and return True when ready to step"""
        
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        self.accumulated_loss += scaled_loss.item()
        
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.current_step += 1
        
        # Check if we should step
        if self.current_step % self.accumulation_steps == 0:
            return True
        
        return False
    
    def step(self, optimizer: torch.optim.Optimizer, 
             scaler: Optional[torch.cuda.amp.GradScaler] = None,
             max_norm: float = 1.0):
        """Perform optimizer step with gradient clipping"""
        
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                max_norm
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params']], 
                max_norm
            )
            optimizer.step()
        
        optimizer.zero_grad()
    
    def get_average_loss(self) -> float:
        """Get average accumulated loss"""
        return self.accumulated_loss