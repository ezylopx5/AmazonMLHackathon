#!/usr/bin/env python3
"""
Image download script for Amazon ML Challenge 2025
Downloads images from URLs in train/test CSV files
"""

import os
import sys
import argparse
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data import ImageDownloader

def setup_logging(log_file: str = None):
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

def download_images_from_csv(csv_path: str, 
                             output_dir: str,
                             num_workers: int = 8,
                             batch_size: int = 1000,
                             skip_existing: bool = True):
    """Download images from CSV file"""
    
    logger = logging.getLogger(__name__)
    
    # Load CSV
    logger.info(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'image_link' not in df.columns:
        logger.error("CSV must contain 'image_link' column")
        return False
    
    # Filter out missing URLs
    df = df.dropna(subset=['image_link'])
    df = df[df['image_link'].str.strip() != '']
    
    logger.info(f"Found {len(df)} images to download")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize downloader
    downloader = ImageDownloader(
        output_dir=output_dir,
        num_workers=num_workers,
        max_retries=3,
        timeout=30
    )
    
    # Process in batches
    total_downloaded = 0
    total_failed = 0
    
    for start_idx in range(0, len(df), batch_size):
        end_idx = min(start_idx + batch_size, len(df))
        batch_df = df.iloc[start_idx:end_idx]
        
        logger.info(f"Processing batch {start_idx//batch_size + 1}/{(len(df)-1)//batch_size + 1}")
        
        # Create URL to filename mapping
        url_to_filename = {}
        for _, row in batch_df.iterrows():
            sample_id = row['sample_id']
            url = row['image_link']
            filename = f"{sample_id}.jpg"
            url_to_filename[url] = filename
        
        # Download batch
        results = downloader.download_images(url_to_filename, skip_existing=skip_existing)
        
        # Count results
        batch_downloaded = sum(1 for r in results.values() if r)
        batch_failed = len(results) - batch_downloaded
        
        total_downloaded += batch_downloaded
        total_failed += batch_failed
        
        logger.info(f"Batch complete: {batch_downloaded} downloaded, {batch_failed} failed")
    
    logger.info(f"Download complete!")
    logger.info(f"  Total downloaded: {total_downloaded}")
    logger.info(f"  Total failed: {total_failed}")
    logger.info(f"  Success rate: {total_downloaded/(total_downloaded+total_failed)*100:.1f}%")
    
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
    parser.add_argument('--num-workers', type=int, default=8,
                       help='Number of download workers')
    parser.add_argument('--batch-size', type=int, default=1000,
                       help='Batch size for downloading')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip already downloaded images')
    parser.add_argument('--force-redownload', action='store_true', default=False,
                       help='Force redownload of existing images')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    # Override skip_existing if force_redownload is set
    skip_existing = args.skip_existing and not args.force_redownload
    
    try:
        # Download training images
        logger.info("Starting training image download...")
        success = download_images_from_csv(
            args.train_csv,
            args.train_output,
            args.num_workers,
            args.batch_size,
            skip_existing
        )
        
        if not success:
            logger.error("Training image download failed")
            sys.exit(1)
        
        # Download test images
        logger.info("Starting test image download...")
        success = download_images_from_csv(
            args.test_csv,
            args.test_output,
            args.num_workers,
            args.batch_size,
            skip_existing
        )
        
        if not success:
            logger.error("Test image download failed")
            sys.exit(1)
        
        logger.info("All downloads completed successfully!")
        
    except Exception as e:
        logger.error(f"Download failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()