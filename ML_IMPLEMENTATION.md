# Machine Learning Implementation in EngagePredict

The EngagePredict project leverages a highly optimized ensemble Machine Learning pipeline to predict social media post engagement. This document details the step-by-step implementation of the ML models driving the core `ml-service`.

## Overview

Unlike simpler regression models, EngagePredict uses an **Ensemble Learning** approach in our FastAPI ML Service (`ml-service/inference.py`). It analyzes content parameters and platform-specific data to categorize the likelihood of engagement success into `Low`, `Medium`, or `High` tiers, mapping these to a 0-100 optimization scale.

The prediction engine relies on three supervised learning algorithms that vote on the final outcome:
1. **Multiclass Logistic Regression** (30% weight)
2. **Random Forest Classifier** (40% weight)
3. **K-Nearest Neighbors (KNN)** (30% weight)

---

## 1. Feature Extraction Pipeline

Before reaching the ML models, the raw input (caption text, hashtags, timestamps, platform, and media data) is mathematically mapped into a 14-dimensional feature vector.

The 14 features extracted are:
- `caption_length`: Raw character count of the caption.
- `hashtag_count`: Total number of hashtags used.
- `hour`: The hour of posting.
- `is_peak`: A boolean (0 or 1) indicating if the time matches the platform's peak hours.
- `is_best_day`: A boolean indicating if the day of week is optimal for the platform.
- `resolution_score`: A scaled value for media resolution (e.g., SD, 1080p, 4K).
- `orientation_match`: A boolean for whether media orientation matches platform preferences (e.g., Portrait for TikTok).
- `media_quality`: An encoded score reflecting estimated media depth (Low, Medium, High).
- `platform_encoded`: Integer representation of the targeted platform (Instagram, TikTok, YouTube, Twitter, Facebook).
- `has_location`: Boolean representing geolocation tagging.
- `has_cta`: Boolean for the presence of Call-To-Action words (e.g., "like", "share", "comment").
- `has_emoji`: Boolean for emoji presence.
- `hashtag_ratio`: The ratio of hashtags used against the maximum optimal amount.
- `caption_ratio`: The ratio of caption length against the optimal length.

## 2. Advanced Preprocessing

The 14-feature array is mathematically standardized via a **Scikit-Learn StandardScaler**.
Scaling transforms the variables so they have a mean of 0 and unity variance. This is critically important for the **KNN (K-Nearest Neighbors)** and **Logistic Regression** models, which calculate distances and gradients that would otherwise be distorted by larger numerical values (like caption character lengths).

## 3. The ML Ensemble 

Instead of relying on a single algorithm, the system passes the scaled features to three completely different model architectures:

### A. Random Forest Classifier (40% Voting Weight)
- **Role**: Our primary predictor. Uses an ensemble of decision trees.
- **Why**: Excellent at capturing complex, non-linear relationships (e.g., the combination of TikTok + 8 PM + Portrait orientation + short caption). It's naturally robust against overfitting.

### B. Multiclass Logistic Regression (30% Voting Weight)
- **Role**: The baseline linear probabilistic model.
- **Why**: Provides a highly interpretable benchmark and smooths out the variance of the Random Forest. It essentially determines the baseline probability based on linear correlations of features.

### C. K-Nearest Neighbors (KNN) (30% Voting Weight)
- **Role**: Instance-based learning model.
- **Why**: Predicts engagement by finding the most mathematically "similar" historical posts in our dataset. If a new post is almost identical to highly-engaged previous posts, KNN will flag it.

## 4. Final Prediction & Scoring

Each model outputs an array of probabilities for each class (`Low`, `Medium`, `High`).

1. **Weighted Voting**: We combine the probability arrays using the 40/30/30 weights.
2. **Classification**: The class with the highest combined probability becomes the predicted `EngagementLevel`.
3. **Score Mapping**: The resulting probabilities are mathematically mapped back onto a 0-100 scale:
   - Low baseline = `0-49`
   - Medium baseline = `50-74`
   - High baseline = `75-100`

## 5. Deployment Architecture 

- All models are trained via `ml-service/train_models.py` on synthesized historical data.
- The models, scaler, and label encoder are exported via `pickle` into the `ml-service/models/` directory.
- During runtime, the FastAPI server efficiently loads the `.pkl` files into memory once on startup to provide sub-millisecond, low-latency API inference.

*(Note: Earlier theoretical implementations exploring automated PCA and K-Means clustering are preserved in the root `train_models.py` as legacy data analysis scripts.)*
