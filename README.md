🧠 Amazon ML Challenge 2025 – Smart Product Pricing (Multimodal Price Prediction)

A comprehensive machine learning pipeline developed for the Amazon ML Challenge 2025, focused on predicting e-commerce product prices using a multimodal dataset (text + numerical + image features).
This repository contains all code, data-handling scripts, and our final ensemble-based approach achieving a 57.77 SMAPE on the leaderboard.

⸻

🚀 Project Overview

The challenge required predicting product prices from multiple data types:
	•	📝 Text features: product titles + descriptions
	•	🔢 Numerical features: specs, dimensions, ratings
	•	🏷️ Categorical features: brand, category, subcategory
	•	🖼️ Image features: extracted using ResNet (when available)

Even with significant data challenges (≈ 99.8 % missing images, high price variance), the solution emphasizes:
	•	Robust preprocessing
	•	Advanced feature engineering
	•	Model ensembling
	•	Efficient memory & computation management

⸻

🧩 Solution Architecture

1️⃣ Feature Engineering Pipeline

Component	Techniques Used
Text Processing	TF-IDF vectorization of titles & descriptions
Numerical Features	Log/sqrt transforms, polynomial features
Image Features	ResNet-based embeddings (zero-filled when missing)
Categorical Handling	Target encoding, one-hot for frequent brands/categories
Scaling	RobustScaler to handle outliers
Imputation	Median/mode + category-based defaults


⸻

2️⃣ Advanced Feature Creation
	•	Price interaction ratios and polynomial combinations
	•	K-Means clustering for similarity-based grouping (k = 5, 10, 20)
	•	PCA components to reduce high-dimensional features
	•	Per-category statistics (mean, std, median price)
	•	Target encoding with smoothing

⸻

3️⃣ Model Ensemble Strategy

A weighted ensemble of three gradient-boosting regressors:

Model	Key Config	Notes
🧠 XGBoost	1000 trees, lr = 0.05, max_depth = 8	Main baseline
⚡ LightGBM	1000 trees, lr = 0.05, boosting_type = ‘gbdt’	Faster convergence
🐈 CatBoost	1000 iters, lr = 0.05, depth = 8	Best single-model performer

Ensemble weighting: inverse SMAPE per model
Final post-processing: clipping (0.01 ≤ price ≤ 50 000) + rounding

⸻

📊 Cross-Validation and Evaluation
	•	7-fold cross-validation (stratified by price bins)
	•	Metric: Symmetric Mean Absolute Percentage Error (SMAPE)
	•	Validation consistency: common folds across all models

Final Leaderboard SMAPE: 57.77 %

⸻

⚙️ Implementation Details

🧮 Feature Selection
	1.	Mutual Information → Top 300 features
	2.	F-Score selection → Refine to 250 most informative
	3.	Correlation filter → drop > 0.95 correlated features

⚡ Performance Optimizations
	•	Memory-optimized datatypes (float32)
	•	Batch processing & feature caching
	•	Parallelized feature extraction
	•	Early stopping + grid search hyperparameter tuning

⸻

📁 Repository Structure

amazon-ml-2025/
├── src/
│   ├── __main__.py          # Main training entry
│   ├── data.py              # Data loading, preprocessing
│   ├── features.py          # Feature engineering utilities
│   ├── train_ultra.py       # Model training pipeline
│   ├── utils.py             # Helper functions
│   └── config.py            # Configuration settings
├── scripts/
│   ├── create_submission.py # Final CSV generation
│   ├── validate.py          # Validation analysis
│   └── quick_submission.py  # Rapid local testing
├── dataset/
│   ├── train.csv / test.csv
│   ├── sample_test.csv
│   └── images/
├── output/
│   ├── checkpoints/ logs/ validation/
└── submissions/


⸻

🧠 Key Learnings

💡 From this project I learned to:
	•	Systematically analyze and decompose a real-world ML problem
	•	Design a complete workflow — from EDA to model validation
	•	Handle multimodal and imbalanced data
	•	Identify and fix bottlenecks during training
	•	Build ensemble systems that generalize better

⸻

🤝 Acknowledgments

Big thanks to my teammates for their support throughout the hackathon,
and to the Amazon ML Challenge 2025 organizers for creating such a learning-rich experience!

⸻

🧰 Tech Stack
	•	Python 3.9 + Pandas + NumPy + Scikit-learn
	•	XGBoost | LightGBM | CatBoost
	•	Matplotlib / Seaborn (for analysis)
	•	PyTorch (ResNet image embeddings)

⸻

📜 License

This project is released under the MIT License.

⸻
