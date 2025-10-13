import os
import time
import hashlib
import requests
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Tuple, List
from requests.adapters import HTTPAdapter, Retry
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageDownloader:
    """Robust image downloader with retry logic and caching"""
    
    def __init__(self, max_workers: int = 8, timeout: int = 10):
        self.max_workers = max_workers
        self.timeout = timeout
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def download_single(self, url: str, output_path: str) -> bool:
        """Download a single image"""
        try:
            # Check if already exists
            if os.path.exists(output_path):
                return True
            
            # Create directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download with headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://www.amazon.com/'
            }
            
            response = self.session.get(url, headers=headers, timeout=self.timeout)
            
            if response.status_code == 200 and response.content:
                # Verify it's an image
                try:
                    img = Image.open(requests.io.BytesIO(response.content))
                    img.verify()
                    # Save
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    return True
                except:
                    logger.warning(f"Invalid image from {url}")
                    return False
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {str(e)}")
            return False
    
    def download_batch(self, df: pd.DataFrame, image_dir: str, 
                      url_col: str = 'image_link', 
                      id_col: str = 'sample_id') -> dict:
        """Download images in parallel"""
        
        results = {'success': 0, 'failed': 0, 'skipped': 0}
        failed_ids = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for _, row in df.iterrows():
                url = row[url_col]
                sample_id = row[id_col]
                output_path = os.path.join(image_dir, f"{sample_id}.jpg")
                
                if os.path.exists(output_path):
                    results['skipped'] += 1
                    continue
                
                future = executor.submit(self.download_single, url, output_path)
                futures[future] = sample_id
            
            # Process results
            for future in tqdm(as_completed(futures), total=len(futures), 
                             desc="Downloading images"):
                sample_id = futures[future]
                try:
                    success = future.result()
                    if success:
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                        failed_ids.append(sample_id)
                except Exception as e:
                    results['failed'] += 1
                    failed_ids.append(sample_id)
                    logger.error(f"Error downloading {sample_id}: {str(e)}")
        
        # Create placeholder images for failed downloads
        if failed_ids:
            logger.info(f"Creating placeholders for {len(failed_ids)} failed downloads")
            placeholder = Image.new('RGB', (224, 224), color='white')
            for sample_id in failed_ids:
                output_path = os.path.join(image_dir, f"{sample_id}.jpg")
                placeholder.save(output_path)
        
        logger.info(f"Download complete: {results}")
        return results

def download_all_images(config):
    """Download all train and test images"""
    downloader = ImageDownloader(max_workers=8)
    
    # Load dataframes
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    
    logger.info("Downloading training images...")
    train_results = downloader.download_batch(train_df, config.img_train)
    
    logger.info("Downloading test images...")
    test_results = downloader.download_batch(test_df, config.img_test)
    
    return train_results, test_results

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    import random
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)