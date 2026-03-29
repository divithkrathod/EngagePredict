# EngagePredict: Project Details

## Problem Statement
The problem objective is to predict user engagement levels accurately by analyzing user behavior and interaction data in order to enhance content personalization and improve overall user retention.

## Three Brief Objectives
1. **Data Processing:** To effectively gather and process user interaction logs into meaningful data points.
2. **Predictive Modeling:** To build a machine learning model that can forecast user engagement reliably.
3. **Actionable Insights:** To provide clear recommendations based on data to keep users engaged.

## Dataset Used
The project utilizes a custom synthesized dataset named `engage_predict_dataset.csv`, consisting of 1,000 unique user interaction records. It maps directly into a Pandas DataFrame for Machine Learning, capturing the following vital features:
- **Demographics/Technical:** `user_id`, `device_type`
- **Activity Metrics:** `session_duration_minutes`, `pages_visited`, `click_through_rate`
- **Historical Data:** `historical_likes`, `historical_comments`, `days_since_last_login`
- **Target Variables for ML:** `engagement_tier` (Low/Medium/High) and `churn_probability` (0.01 to 0.95).

## How are the ML Models Used and What are they Doing?
The system architecture follows a structured, multi-model ensemble predictive pipeline. The ML models act together to transform raw content parameters and media data into actionable engagement scores:

1. **Feature Extraction & Scaling:**
   - *What it does:* Condenses textual parameters (caption length, hashtags), categorical data (platform, day), and media values (resolution, orientation) into a clean 14-dimensional numerical array.
   - *How it is used:* Provides mathematically normalized values so algorithms can interpret diverse data points equally.
2. **Core Predictive Engines (Ensemble Learning - Supervised):**
   - *What they do:* Three different supervised machine learning architectures independently evaluate the scaled features. They look at historical synthetic data where engagement was known, memorize those multifaceted patterns, and individually calculate the probability of success for new content.
   - *How they are used:* Their independent outputs are combined using a weighted voting system to produce highly robust, consensus-driven predictive metrics (e.g., 'High' Engagement Level and an overarching 0-100 Score).

## ML Supervised Learning Techniques Used
This project heavily utilizes an Ensemble System featuring three primary Supervised Learning techniques for its core predictive engine:
1. **Classification (Random Forest Classifier):** A technique used to categorize data into discrete buckets. Here, Random Forest uses an ensemble of dozens of decision trees to classify the engagement potential, carrying a 40% voting weight natively.
2. **Multiclass Logistic Regression:** A foundational statistical model offering a highly interpretable baseline probability based on linear correlations of features. It carries a 30% voting weight.
3. **K-Nearest Neighbors (KNN):** An instance-based learning algorithm that groups new content with the most mathematically "similar" historical posts to derive its engagement class. It carries the remaining 30% voting weight.

## Methodology
1. **Data Generation:** Creating synthetic training data with realistic engagement patterns.
2. **Feature Extraction:** Processing user inputs to extract 14 relevant core features.
3. **Model Training:** Training individual ML models (Logistic Regression, Random Forest, KNN) on the synthetic dataset.
4. **Ensemble Prediction:** Combining model predictions using weighted voting.
5. **Recommendation Generation:** Providing personal platform-specific tips based on prediction results and best practices.

## How to Run This Project
Follow these simple execution steps to demonstrate the pipeline on your machine:
1. **Prerequisites:** Ensure Python is installed on your Windows system.
2. **Install Dependencies:** Open PowerShell or Command Prompt in the `EngagePredict` project folder and install the required ML libraries:
   ```powershell
   pip install -r requirements.txt
   ```
3. **Generate the Dataset:** If you want to recreate or alter the 1,000-record CSV dataset, run:
   ```powershell
   python generate_mock_data.py
   ```
4. **Train the ML Models:** To load the Pandas DataFrame, scale the variables, train the models, and save them for deployment, run:
   ```powershell
   python train_models.py
   ```
   *Note: This will dynamically execute PCA, K-Means, Random Forest, and XGBoost, output accurate benchmark metrics to the terminal, and export the `.pkl` model files to the `models/` directory.*

## Implementation Progress & Results
**[STATUS: COMPLETE]** The Machine Learning models have been successfully trained on the synthesized data!
- **Data Integration:** The dataset was smoothly implemented into the Pandas pipeline.
- **Dimensionality Reduction:** PCA mapped the metrics correctly, and K-Means accurately mapped distinct overarching clusters.
- **Predictive Core Training:** Both the Random Forest Classifier and the XGBoost Regressor were fully trained.
- **Artifacts Exported:** All deployed models (`pca.pkl`, `kmeans.pkl`, `rf_classifier.pkl`, `xgb_regressor.pkl`, `scaler.pkl`) have been structurally saved inside the `models` directory and are ready for future use!

## SDG Relevance
**Sustainable Development Goal 9: Industry, Innovation, and Infrastructure**
- **Target 9.5:** Enhance scientific research, upgrade the technological capabilities of industrial sectors. 
*Relevance:* By leveraging machine learning algorithms to predict user engagement, this project fosters technological innovation and upgrades digital platform capabilities.
