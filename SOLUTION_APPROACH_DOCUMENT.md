# Amazon ML Challenge 2025 - Solution Approach Document

## Team Information
- **Challenge**: Amazon ML Challenge 2025 - Product Price Prediction
- **Approach**: Multimodal Machine Learning with Advanced Feature Engineering


## Problem Understanding
The challenge involves predicting product prices using multimodal data including:
- **Text Features**: Product titles, descriptions, categories
- **Numerical Features**: Product specifications, dimensions, ratings
- **Image Features**: Product images (when available)
- **Categorical Features**: Brand, category, subcategory information

## Data Challenges Identified
1. **Missing Images**: 
2. **Feature Sparsity**: Many products lack complete specifications
3. **Price Range Variance**: Products range from $0.13 to $2,796
4. **Categorical Imbalance**: Some categories have very few examples

## Solution Architecture

### 1. Feature Engineering Pipeline
- **Text Processing**: TF-IDF vectorization of product titles and descriptions
- **Numerical Processing**: Statistical transforms (log, sqrt, polynomial features)
- **Image Processing**: ResNet-based feature extraction for available images
- **Missing Value Imputation**: Median/mode imputation with category-based defaults
- **Feature Scaling**: RobustScaler for handling outliers

### 2. Advanced Feature Creation
- **Price Interaction Features**: Ratios and differences between price-related attributes
- **Clustering Features**: K-means clustering (k=5,10,20) for product similarity
- **PCA Components**: Dimensionality reduction for numerical features
- **Target Encoding**: Category-based price encoding with smoothing
- **Statistical Aggregations**: Per-category price statistics

### 3. Model Ensemble Strategy
Our final solution uses a weighted ensemble of three models:

#### Model 1: XGBoost Regressor
- **Configuration**: 1000 estimators, 0.05 learning rate, 8 max depth
- **Features**: All engineered features (250+ selected via mutual information)
- **Validation**: 7-fold cross-validation


#### Model 2: LightGBM Regressor  
- **Configuration**: 1000 estimators, 0.05 learning rate, boosting_type='gbdt'
- **Features**: Same feature set as XGBoost with different handling of categoricals
- **Validation**: 7-fold cross-validation


#### Model 3: CatBoost Regressor
- **Configuration**: 1000 iterations, 0.05 learning rate, 8 depth
- **Features**: Native categorical feature handling
- **Validation**: 7-fold cross-validation


### 4. Ensemble Weighting
Final predictions use inverse SMAPE weighting:
- **Weight Calculation**: w_i = (1/SMAPE_i) / Σ(1/SMAPE_j)
- **Combination**: Weighted average of individual model predictions
- **Post-processing**: Price clipping (min=$0.01, max=$50,000) and rounding

## Technical Implementation

### Feature Selection Strategy
1. **Mutual Information**: Select top 300 features based on target correlation
2. **F-Score Selection**: Further reduce to 250 most informative features
3. **Redundancy Removal**: Remove highly correlated features (>0.95 correlation)

### Cross-Validation Strategy
- **Method**: 7-fold cross-validation with stratified sampling
- **Metric**: SMAPE (Symmetric Mean Absolute Percentage Error)
- **Validation**: Consistent across all models for fair comparison

### Handling Missing Images
Since only 0.2% of samples have images, we implemented:
1. **Default Image Features**: Zero-filled for missing images
2. **Category-Based Imputation**: Use average image features per product category
3. **Text Feature Compensation**: Enhanced text processing to compensate for missing visual information

## Performance Optimization

### Memory Management
- **Batch Processing**: Process features in chunks to handle large dataset
- **Data Types**: Optimized dtypes (float32 instead of float64)
- **Feature Caching**: Save processed features to avoid recomputation

### Computational Efficiency
- **Parallel Processing**: Multi-core feature extraction and model training
- **Early Stopping**: Prevent overfitting and reduce training time
- **Hyperparameter Optimization**: Grid search on validation set



### Leaderboard Expectations
Given the missing image challenge, we expect:
- **Position**
- **Performance Gap**: ~ penalty due to missing images

## Code Structure
```
src/
├── __main__.py          # Training pipeline entry point
├── train_ultra.py       # Advanced model training
├── features.py          # Feature engineering
├── data.py              # Data loading and preprocessing
├── utils.py             # Utility functions
└── config.py            # Configuration management

scripts/
├── create_submission.py # Final submission generation
├── validate.py          # Model validation
└── quick_submission.py  # Rapid submission creation
```

## Conclusion
Our solution addresses the multimodal nature of the problem through comprehensive feature engineering and robust ensemble modeling. Despite the significant challenge of missing images (99.8% missing), we compensate through advanced text processing and statistical feature creation. The 7-fold cross-validated ensemble approach provides robust predictions while the weighted combination leverages the strengths of different algorithms.

The modular code structure allows for easy experimentation and deployment, while the validation pipeline ensures reproducible results. We expect this approach to achieve top 50 performance in the Amazon ML Challenge 2025.