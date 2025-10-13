import re
import os
import gc
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
import logging
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

def load_features(feats_dir: str, split: str) -> pd.DataFrame:
    """Load features with fallback from parquet to CSV"""
    parquet_path = os.path.join(feats_dir, f'{split}.parquet')
    csv_path = os.path.join(feats_dir, f'{split}.csv')
    
    if os.path.exists(parquet_path):
        try:
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Failed to load parquet ({e}), trying CSV")
    
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} exists")

class FeatureExtractor:
    """Comprehensive feature extraction for text and images"""
    
    def __init__(self):
        # IPQ patterns in priority order
        self.ipq_patterns = [
            r'ipq[:\s]*(\d+)',
            r'item\s*pack\s*quantity[:\s]*(\d+)',
            r'pack\s*of\s*(\d+)',
            r'\(pack\s*of\s*(\d+)\)',
            r'(\d+)\s*pack',
            r'(\d+)\s*count',
            r'quantity[:\s]*(\d+)'
        ]
        
        # Brand lists
        self.premium_brands = {
            'samsung', 'apple', 'sony', 'lg', 'bose', 'nike', 'adidas', 
            'dyson', 'philips', 'bosch', 'canon', 'nikon', 'dell', 'hp',
            'microsoft', 'intel', 'amd', 'nvidia', 'asus', 'lenovo',
            'gucci', 'prada', 'louis vuitton', 'chanel', 'rolex'
        }
        
        self.budget_brands = {
            'generic', 'basics', 'essentials', 'value', 'economy',
            'amazonbasics', 'great value', 'kirkland', 'member\'s mark'
        }
        
        # Technical specifications
        self.tech_units = [
            'gb', 'tb', 'mb', 'kb', 'ghz', 'mhz', 'hz', 'inch', '"',
            'mah', 'wh', 'watt', 'volt', 'amp', 'rpm', 'hp', 'cc',
            'ml', 'l', 'fl oz', 'oz', 'lb', 'kg', 'g', 'mg',
            'pixel', 'mp', 'fps', 'dpi', 'ppi'
        ]
        
        # Quality indicators
        self.quality_words = [
            'premium', 'pro', 'professional', 'deluxe', 'luxury', 'elite',
            'ultra', 'super', 'max', 'plus', 'advanced', 'supreme',
            'organic', 'natural', 'handmade', 'artisan', 'authentic',
            'genuine', 'original', 'certified', 'guaranteed'
        ]
        
        # Categories with keywords
        self.categories = {
            'electronics': ['phone', 'laptop', 'tablet', 'computer', 'camera', 
                          'tv', 'television', 'monitor', 'speaker', 'headphone',
                          'keyboard', 'mouse', 'printer', 'router', 'modem'],
            'clothing': ['shirt', 'pants', 'dress', 'shoe', 'jacket', 'coat',
                        'jeans', 'sweater', 'hoodie', 'sock', 'underwear',
                        'cotton', 'polyester', 'wool', 'leather', 'silk'],
            'food': ['organic', 'fresh', 'frozen', 'snack', 'beverage', 'drink',
                    'coffee', 'tea', 'chocolate', 'candy', 'protein', 'vitamin',
                    'supplement', 'gluten free', 'sugar free', 'vegan'],
            'home': ['furniture', 'chair', 'table', 'bed', 'sofa', 'couch',
                    'kitchen', 'bathroom', 'bedroom', 'living room', 'decor',
                    'lamp', 'rug', 'curtain', 'pillow', 'blanket'],
            'sports': ['fitness', 'gym', 'yoga', 'running', 'cycling', 'swim',
                      'ball', 'racket', 'weight', 'dumbbell', 'exercise',
                      'workout', 'training', 'athletic', 'sportswear'],
            'beauty': ['makeup', 'cosmetic', 'skincare', 'haircare', 'perfume',
                      'lotion', 'cream', 'shampoo', 'conditioner', 'soap'],
            'toys': ['toy', 'game', 'puzzle', 'doll', 'lego', 'playset',
                    'action figure', 'board game', 'educational', 'kids'],
            'books': ['book', 'novel', 'paperback', 'hardcover', 'kindle',
                     'ebook', 'audiobook', 'textbook', 'magazine', 'comic']
        }
        
        # Unit conversions
        self.unit_conversions = {
            'fl oz': ('ml', 29.5735),
            'fluid ounce': ('ml', 29.5735),
            'ounce': ('g', 28.3495),
            'oz': ('g', 28.3495),
            'pound': ('g', 453.592),
            'lb': ('g', 453.592),
            'liter': ('ml', 1000.0),
            'l': ('ml', 1000.0),
            'gallon': ('ml', 3785.41),
            'quart': ('ml', 946.353),
            'pint': ('ml', 473.176),
            'cup': ('ml', 236.588),
            'tablespoon': ('ml', 14.7868),
            'teaspoon': ('ml', 4.92892),
            'kilogram': ('g', 1000.0),
            'kg': ('g', 1000.0),
            'milligram': ('g', 0.001),
            'mg': ('g', 0.001),
            'inch': ('cm', 2.54),
            '"': ('cm', 2.54),
            'foot': ('cm', 30.48),
            'ft': ('cm', 30.48),
            'yard': ('cm', 91.44),
            'meter': ('cm', 100.0),
            'm': ('cm', 100.0)
        }
    
    def extract_ipq(self, text: str) -> int:
        """Extract Item Pack Quantity with multiple strategies"""
        if pd.isna(text) or not text:
            return 1
        
        text_lower = text.lower()
        
        # Try each pattern in priority order
        for pattern in self.ipq_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    qty = int(match.group(1))
                    # Sanity check
                    if 1 <= qty <= 1000:
                        return qty
                except:
                    continue
        
        return 1
    
    def extract_value_unit(self, text: str) -> Tuple[Optional[float], Optional[str], Optional[str]]:
        """Extract and normalize value and unit from text"""
        if pd.isna(text) or not text:
            return None, None, None
        
        # Look for explicit Value: X Unit: Y pattern
        value_match = re.search(r'Value[:\s]*([\d\.]+)', text, re.IGNORECASE)
        unit_match = re.search(r'Unit[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        
        if value_match and unit_match:
            try:
                value = float(value_match.group(1))
                unit_raw = unit_match.group(1).strip().lower()
                
                # Normalize unit
                unit_normalized = unit_raw.replace(' ', '')
                
                # Convert if possible
                if unit_normalized in self.unit_conversions:
                    new_unit, factor = self.unit_conversions[unit_normalized]
                    return value * factor, new_unit, unit_raw
                
                return value, unit_raw, unit_raw
            except:
                pass
        
        return None, None, None
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive text features"""
        if pd.isna(text) or not text:
            text = ""
        
        text_lower = text.lower()
        
        features = {}
        
        # 1. IPQ features (CRITICAL)
        ipq = self.extract_ipq(text)
        features['ipq'] = ipq
        features['log_ipq'] = np.log1p(ipq)
        features['ipq_squared'] = ipq ** 2
        features['is_multipack'] = int(ipq > 1)
        features['ipq_category'] = min(ipq // 10, 10)  # Bucketed IPQ
        
        # 2. Value and unit
        value, unit_norm, unit_raw = self.extract_value_unit(text)
        features['has_value_unit'] = int(value is not None)
        features['value_normalized'] = value if value is not None else 0.0
        features['log_value'] = np.log1p(value) if value is not None else 0.0
        
        # 3. Brand features
        features['has_premium_brand'] = int(any(brand in text_lower for brand in self.premium_brands))
        features['has_budget_brand'] = int(any(brand in text_lower for brand in self.budget_brands))
        features['has_any_brand'] = int(features['has_premium_brand'] or features['has_budget_brand'])
        
        # Count brand mentions
        brand_count = sum(1 for brand in self.premium_brands if brand in text_lower)
        features['premium_brand_count'] = brand_count
        
        # 4. Technical specifications
        spec_counts = {}
        for unit in self.tech_units:
            pattern = rf'\d+\.?\d*\s*{re.escape(unit)}\b'
            matches = re.findall(pattern, text_lower)
            if matches:
                spec_counts[unit] = len(matches)
        
        features['spec_density'] = len(spec_counts)
        features['total_spec_mentions'] = sum(spec_counts.values())
        features['has_technical_specs'] = int(features['spec_density'] > 0)
        
        # Specific spec indicators
        features['has_memory_spec'] = int(any(u in spec_counts for u in ['gb', 'tb', 'mb']))
        features['has_size_spec'] = int(any(u in spec_counts for u in ['inch', '"', 'cm', 'mm']))
        features['has_weight_spec'] = int(any(u in spec_counts for u in ['kg', 'g', 'lb', 'oz']))
        features['has_volume_spec'] = int(any(u in spec_counts for u in ['ml', 'l', 'fl oz']))
        
        # 5. Quality indicators
        quality_score = sum(1 for word in self.quality_words if word in text_lower)
        features['quality_score'] = quality_score
        features['is_premium_product'] = int(quality_score >= 2)
        features['has_quality_words'] = int(quality_score > 0)
        
        # 6. Category detection
        for cat_name, keywords in self.categories.items():
            cat_score = sum(1 for kw in keywords if kw in text_lower)
            features[f'cat_{cat_name}'] = int(cat_score > 0)
            features[f'cat_{cat_name}_score'] = cat_score
        
        # Dominant category
        cat_scores = {cat: features[f'cat_{cat}_score'] for cat in self.categories.keys()}
        if max(cat_scores.values()) > 0:
            features['dominant_category'] = max(cat_scores, key=cat_scores.get)
        else:
            features['dominant_category'] = 'unknown'
        
        # 7. Numeric analysis
        numbers = re.findall(r'\d+\.?\d*', text)
        numbers = [float(n) for n in numbers[:50]]  # Limit to first 50
        
        if numbers:
            features['max_number'] = max(numbers)
            features['min_number'] = min(numbers)
            features['avg_number'] = np.mean(numbers)
            features['median_number'] = np.median(numbers)
            features['std_number'] = np.std(numbers)
            features['num_count'] = len(numbers)
            features['has_decimal'] = int(any('.' in str(n) for n in numbers))
            features['has_large_number'] = int(max(numbers) > 1000)
        else:
            features['max_number'] = 0
            features['min_number'] = 0
            features['avg_number'] = 0
            features['median_number'] = 0
            features['std_number'] = 0
            features['num_count'] = 0
            features['has_decimal'] = 0
            features['has_large_number'] = 0
        
        # 8. Text statistics
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['unique_words'] = len(set(text_lower.split()))
        features['char_count'] = len(text.replace(' ', ''))
        features['avg_word_length'] = np.mean([len(w) for w in text.split()]) if text else 0
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        
        # Text complexity
        features['lexical_diversity'] = features['unique_words'] / max(features['word_count'], 1)
        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)
        
        # 9. Special patterns
        features['has_model_number'] = int(bool(re.search(r'[A-Z0-9]{4,}', text)))
        features['has_dimensions'] = int('x' in text and any(c.isdigit() for c in text))
        features['has_warranty'] = int('warranty' in text_lower or 'guarantee' in text_lower)
        features['has_discount'] = int(any(w in text_lower for w in ['sale', 'discount', 'save', 'off', '%']))
        features['has_shipping'] = int(any(w in text_lower for w in ['shipping', 'delivery', 'prime']))
        features['has_bullet_points'] = int('bullet point' in text_lower)
        
        # 10. Price hints from text
        price_patterns = [
            r'\$\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)\s*dollars?',
            r'price[:\s]*(\d+\.?\d*)'
        ]
        
        price_hints = []
        for pattern in price_patterns:
            matches = re.findall(pattern, text_lower)
            price_hints.extend([float(m) for m in matches])
        
        if price_hints:
            features['price_hint_max'] = max(price_hints)
            features['price_hint_min'] = min(price_hints)
            features['price_hint_avg'] = np.mean(price_hints)
            features['has_price_hint'] = 1
        else:
            features['price_hint_max'] = 0
            features['price_hint_min'] = 0
            features['price_hint_avg'] = 0
            features['has_price_hint'] = 0
        
        return features
    
    def extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Extract image metadata and quality features - OPTIMIZED VERSION"""
        features = {}
        
        # Check if image path exists
        if not image_path or not os.path.exists(image_path):
            # Return default features for missing images
            return {
                'img_width': 224, 'img_height': 224, 'img_area': 50176,
                'img_aspect_ratio': 1.0, 'img_diagonal': 316.8,
                'is_square': 1, 'is_portrait': 0, 'is_landscape': 0,
                'is_hd': 0, 'is_4k': 0, 'is_small': 0,
                'img_brightness': 128.0, 'img_contrast': 50.0, 'img_saturation': 30.0,
                'img_red_mean': 128.0, 'img_red_std': 50.0,
                'img_green_mean': 128.0, 'img_green_std': 50.0,
                'img_blue_mean': 128.0, 'img_blue_std': 50.0,
                'dominant_red': 0, 'dominant_green': 0, 'dominant_blue': 0,
                'has_white_bg': 0, 'white_corner_count': 0,
                'edge_density': 10.0, 'img_entropy': 5.0, 'color_diversity': 50.0
            }
        
        try:
            # Open and resize image to reasonable size for fast processing
            img = Image.open(image_path).convert('RGB')
            
            # 1. Basic metadata (fast)
            features['img_width'] = img.width
            features['img_height'] = img.height
            features['img_area'] = img.width * img.height
            features['img_aspect_ratio'] = img.width / max(img.height, 1)
            features['img_diagonal'] = np.sqrt(img.width**2 + img.height**2)
            
            # 2. Image quality indicators (fast)
            features['is_square'] = int(abs(img.width - img.height) < 50)
            features['is_portrait'] = int(img.height > img.width)
            features['is_landscape'] = int(img.width > img.height)
            features['is_hd'] = int(min(img.width, img.height) >= 720)
            features['is_4k'] = int(min(img.width, img.height) >= 2160)
            features['is_small'] = int(max(img.width, img.height) < 500)
            
            # OPTIMIZATION: Resize to smaller size for analysis (MASSIVE SPEEDUP)
            # This is the key optimization - analyze a smaller version
            max_analysis_size = 224  # Much smaller than original
            if max(img.width, img.height) > max_analysis_size:
                ratio = max_analysis_size / max(img.width, img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img_analysis = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                img_analysis = img
            
            # 3. Convert to numpy for analysis (now much smaller)
            img_array = np.array(img_analysis)
            
            # 4. Color statistics (fast on small image)
            features['img_brightness'] = float(np.mean(img_array))
            features['img_contrast'] = float(np.std(img_array))
            features['img_saturation'] = float(np.std(img_array.reshape(-1, 3).std(axis=1)))
            
            # Channel statistics
            for i, channel in enumerate(['red', 'green', 'blue']):
                features[f'img_{channel}_mean'] = float(np.mean(img_array[:, :, i]))
                features[f'img_{channel}_std'] = float(np.std(img_array[:, :, i]))
            
            # Dominant channel
            channel_means = [features[f'img_{c}_mean'] for c in ['red', 'green', 'blue']]
            dominant_idx = np.argmax(channel_means)
            features['dominant_red'] = int(dominant_idx == 0)
            features['dominant_green'] = int(dominant_idx == 1)
            features['dominant_blue'] = int(dominant_idx == 2)
            
            # 5. White background detection (on small image)
            h, w = img_array.shape[:2]
            corner_size = max(1, min(10, h//10, w//10))  # Adaptive corner size
            
            corners = [
                img_array[0:corner_size, 0:corner_size].mean(),
                img_array[0:corner_size, -corner_size:].mean(),
                img_array[-corner_size:, 0:corner_size].mean(),
                img_array[-corner_size:, -corner_size:].mean()
            ]
            
            white_threshold = 240
            white_corners = sum(1 for c in corners if c > white_threshold)
            features['has_white_bg'] = int(white_corners >= 3)
            features['white_corner_count'] = white_corners
            
            # Edge detection (simple and fast)
            gray = img_array.mean(axis=2)
            edges_h = np.abs(np.diff(gray, axis=0)).mean()
            edges_v = np.abs(np.diff(gray, axis=1)).mean()
            features['edge_density'] = float(edges_h + edges_v)
            
            # 6. Fast entropy calculation (no scipy needed)
            # Use histogram on smaller image for speed
            hist, _ = np.histogram(img_array.flatten(), bins=64, range=(0, 256))  # Fewer bins
            hist = hist / hist.sum()
            features['img_entropy'] = float(-np.sum(hist * np.log(hist + 1e-10)))
            
            # 7. OPTIMIZED color diversity (much faster)
            # Sample pixels instead of checking all unique colors
            pixels_flat = img_array.reshape(-1, 3)
            sample_pixels = pixels_flat[::max(1, len(pixels_flat)//1000)]  # Sample max 1000 pixels
            unique_colors = len(np.unique(sample_pixels, axis=0))
            features['color_diversity'] = min(unique_colors / 10, 100)  # Normalized differently
            
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {str(e)}")
            # Return default values
            features = {
                'img_width': 224, 'img_height': 224, 'img_area': 50176,
                'img_aspect_ratio': 1.0, 'img_diagonal': 316.8,
                'is_square': 1, 'is_portrait': 0, 'is_landscape': 0,
                'is_hd': 0, 'is_4k': 0, 'is_small': 0,
                'img_brightness': 128.0, 'img_contrast': 50.0, 'img_saturation': 30.0,
                'img_red_mean': 128.0, 'img_red_std': 50.0,
                'img_green_mean': 128.0, 'img_green_std': 50.0,
                'img_blue_mean': 128.0, 'img_blue_std': 50.0,
                'dominant_red': 0, 'dominant_green': 0, 'dominant_blue': 0,
                'has_white_bg': 0, 'white_corner_count': 0,
                'edge_density': 10.0, 'img_entropy': 5.0, 'color_diversity': 50.0
            }
        
        return features
    
    def extract_all_features(self, row: pd.Series, image_dir: str) -> Dict[str, Any]:
        """Extract all features for a single sample"""
        # Text features
        text_features = self.extract_text_features(row.get('catalog_content', ''))
        
        # Image features
        image_path = os.path.join(image_dir, f"{row['sample_id']}.jpg")
        image_features = self.extract_image_features(image_path)
        
        # Combine
        all_features = {**text_features, **image_features}
        all_features['sample_id'] = row['sample_id']
        
        if 'price' in row:
            all_features['price'] = row['price']
        
        return all_features

def build_feature_cache(config):
    """Build and cache all features"""
    logger.info("Building feature cache...")
    
    # Create output directory
    os.makedirs(config.feats_dir, exist_ok=True)
    
    # Initialize extractor
    extractor = FeatureExtractor()
    
    # Load data
    logger.info(f"Loading CSV files...")
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    logger.info(f"Loaded {len(train_df)} train samples, {len(test_df)} test samples")
    
    # Process training data with better memory management
    logger.info("Extracting training features...")
    train_features = []
    batch_size = 1000  # Process in batches to manage memory
    
    for i in tqdm(range(0, len(train_df), batch_size), desc="Train batches"):
        batch_df = train_df.iloc[i:i+batch_size]
        batch_features = []
        
        for _, row in batch_df.iterrows():
            try:
                features = extractor.extract_all_features(row, config.img_train)
                batch_features.append(features)
            except Exception as e:
                logger.warning(f"Error processing sample {row.get('sample_id', 'unknown')}: {e}")
                # Create default features for failed samples
                default_features = extractor.extract_all_features(row, "")  # Will use defaults
                batch_features.append(default_features)
        
        train_features.extend(batch_features)
        
        # Periodic garbage collection
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    train_features_df = pd.DataFrame(train_features)
    
    # Save with fallback from parquet to CSV
    try:
        train_features_df.to_parquet(
            os.path.join(config.feats_dir, 'train.parquet'),
            index=False
        )
        logger.info(f"Saved {len(train_features_df)} training samples with {len(train_features_df.columns)} features to parquet")
    except Exception as e:
        logger.warning(f"Failed to save parquet ({e}), falling back to CSV")
        train_features_df.to_csv(
            os.path.join(config.feats_dir, 'train.csv'),
            index=False
        )
        logger.info(f"Saved {len(train_features_df)} training samples with {len(train_features_df.columns)} features to CSV")
    
    # Process test data with better memory management
    logger.info("Extracting test features...")
    test_features = []
    
    for i in tqdm(range(0, len(test_df), batch_size), desc="Test batches"):
        batch_df = test_df.iloc[i:i+batch_size]
        batch_features = []
        
        for _, row in batch_df.iterrows():
            try:
                features = extractor.extract_all_features(row, config.img_test)
                batch_features.append(features)
            except Exception as e:
                logger.warning(f"Error processing sample {row.get('sample_id', 'unknown')}: {e}")
                # Create default features for failed samples
                default_features = extractor.extract_all_features(row, "")  # Will use defaults
                batch_features.append(default_features)
        
        test_features.extend(batch_features)
        
        # Periodic garbage collection
        if i % (batch_size * 10) == 0:
            gc.collect()
    
    test_features_df = pd.DataFrame(test_features)
    
    # Save with fallback from parquet to CSV
    try:
        test_features_df.to_parquet(
            os.path.join(config.feats_dir, 'test.parquet'),
            index=False
        )
        logger.info(f"Saved {len(test_features_df)} test samples with {len(test_features_df.columns)} features to parquet")
    except Exception as e:
        logger.warning(f"Failed to save parquet ({e}), falling back to CSV")
        test_features_df.to_csv(
            os.path.join(config.feats_dir, 'test.csv'),
            index=False
        )
        logger.info(f"Saved {len(test_features_df)} test samples with {len(test_features_df.columns)} features to CSV")
    
    # Print feature statistics with error handling
    try:
        logger.info("\nFeature Statistics:")
        if 'ipq' in train_features_df.columns:
            logger.info(f"IPQ distribution in train:\n{train_features_df['ipq'].value_counts().head(10)}")
        
        # Category features
        cat_cols = [c for c in train_features_df.columns if c.startswith('cat_') and not c.endswith('_score')]
        if cat_cols:
            logger.info(f"Category distribution:\n{train_features_df[cat_cols].sum()}")
        
        # Image feature stats
        img_cols = [c for c in train_features_df.columns if c.startswith('img_')]
        if img_cols:
            logger.info(f"Image features extracted: {len(img_cols)} features")
            
        logger.info(f"Total features per sample: {len(train_features_df.columns)}")
        
    except Exception as e:
        logger.warning(f"Error computing feature statistics: {e}")
    
    # Final cleanup
    gc.collect()
    
    return train_features_df, test_features_df