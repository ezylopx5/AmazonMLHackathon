#!/usr/bin/env python3
"""
Simple image download script for Amazon ML Challenge 2025
Downloads images from URLs in train/test CSV files
"""

import os
import sys
import argparse
import pandas as pd
import requests
import logging
from tqdm import tqdm
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def setup_logging(log_file=None):
    """Setup logging configuration"""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def download_image(url, output_path, timeout=30):
    """Download a single image"""
    try:
        if os.path.exists(output_path):
            return True  # Skip existing
            
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout, stream=True)
        
        if response.status_code == 200:
            # Verify it's an image by checking content type
            content_type = response.headers.get('content-type', '')
            if 'image' not in content_type.lower():
                return False
                
            # Save image
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Verify the saved image
            try:
                with Image.open(output_path) as img:
                    img.verify()
                return True
            except:
                # Remove invalid image
                if os.path.exists(output_path):
                    os.remove(output_path)
                return False
                
    except Exception as e:
        return False
    
    return False

def download_images_from_csv(csv_path, output_dir, num_workers=4, skip_existing=True, logger=None):
    """Download images from CSV file"""
    
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'image_link' not in df.columns:
        logger.error("CSV must contain 'image_link' column")
        return False
    
    if 'sample_id' not in df.columns:
        logger.error("CSV must contain 'sample_id' column")
        return False
    
    # Filter out missing URLs
    original_count = len(df)
    df = df.dropna(subset=['image_link'])
    df = df[df['image_link'].str.strip() != '']
    
    logger.info(f"Found {len(df)} valid URLs out of {original_count} rows")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare download tasks
    download_tasks = []
    for _, row in df.iterrows():
        url = row['image_link']
        sample_id = row['sample_id']
        output_path = os.path.join(output_dir, f"{sample_id}.jpg")
        
        if skip_existing and os.path.exists(output_path):
            continue
            
        download_tasks.append((url, output_path))
    
    logger.info(f"Need to download {len(download_tasks)} images")
    
    if not download_tasks:
        logger.info("No images to download (all exist)")
        return True
    
    # Download images in parallel
    successful = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_url = {
            executor.submit(download_image, url, path): (url, path)
            for url, path in download_tasks
        }
        
        # Process results with progress bar
        for future in tqdm(as_completed(future_to_url), total=len(download_tasks), desc="Downloading"):
            url, path = future_to_url[future]
            try:
                result = future.result()
                if result:
                    successful += 1
                else:
                    failed += 1
                    # Create a placeholder image for failed downloads
                    try:
                        placeholder = Image.new('RGB', (224, 224), color='white')
                        placeholder.save(path)
                    except:
                        pass
            except Exception as e:
                failed += 1
                logger.error(f"Error downloading {url}: {e}")
    
    logger.info(f"Download complete!")
    logger.info(f"  Successful: {successful}")
    logger.info(f"  Failed: {failed}")
    logger.info(f"  Success rate: {successful/(successful+failed)*100:.1f}%")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Download images for Amazon ML Challenge')
    
    parser.add_argument('--train-csv', type=str, required=True,
                       help='Path to training CSV file')
    parser.add_argument('--test-csv', type=str, required=True,
                       help='Path to test CSV file')
    parser.add_argument('--train-output', type=str, required=True,
                       help='Output directory for training images')
    parser.add_argument('--test-output', type=str, required=True,
                       help='Output directory for test images')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of download workers')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip already downloaded images')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    try:
        # Download training images
        logger.info("Starting training image download...")
        success = download_images_from_csv(
            args.train_csv,
            args.train_output,
            args.num_workers,
            args.skip_existing,
            logger
        )
        
        if not success:
            logger.error("Training image download failed")
            return 1
        
        # Download test images
        logger.info("Starting test image download...")
        success = download_images_from_csv(
            args.test_csv,
            args.test_output,
            args.num_workers,
            args.skip_existing,
            logger
        )
        
        if not success:
            logger.error("Test image download failed")
            return 1
        
        logger.info("All downloads completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())