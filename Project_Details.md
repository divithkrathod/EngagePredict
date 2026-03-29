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
The system architecture follows a structured, sequential predictive pipeline. The ML models act together to transform raw interaction logs into actionable engagement scores:

1. **Dimensionality Reduction (PCA):** 
   - *What it does:* Condenses hundreds of user interaction metrics (clicks, time spent) into the most critical mathematical signals.
   - *How it is used:* Removes noise from the dataset so the predictive models run faster and more accurately.
2. **User Segmentation (K-Means Clustering - Unsupervised):**
   - *What it does:* Analyzes the PCA patterns and groups users into behavioral clusters (e.g., 'Power Users', 'At-Risk Users') without any human labels.
   - *How it is used:* Provides an overarching category for each user before attempting to predict specific outcomes.
3. **Core Predictive Engines (Supervised Learning):**
   - *What they do:* They learn explicitly from historical data. They look at users who previously had high engagement or churned, memorize their behavioral patterns, and apply those patterns to guess behavior in new users.
   - *How they are used:* They generate the final predictive metrics (Categorical Engagement Tier and Numerical Churn Probability) which platform administrators can use to intervene.

## ML Supervised Learning Techniques Used
This project heavily utilizes two primary Supervised Learning techniques for its core predictive engine:
1. **Classification (Random Forest Classifier):** A technique used to categorize data into discrete buckets. Here, Random Forest uses an ensemble of dozens of decision trees to classify users into distinct tiers: *Low, Medium, or High* engagement.
2. **Regression (XGBoost Regressor):** A technique used to forecast exact numerical values. Here, Extreme Gradient Boosting (XGBoost) evaluates behavior to predict continuous variables, specifically the numerical *Churn Probability* percentage (e.g., 75% likely to churn).

## Methodology
1. **Data Collection:** Gathering behavioral data from user sessions.
2. **Data Preprocessing & PCA:** Cleaning the data, handling missing information, and reducing dimensionality.
3. **User Segmentation:** Using K-Means to find natural behavioral groupings.
4. **Model Training:** Training Random Forest and XGBoost models on the segmented data.
5. **Evaluation:** Testing the models to ensure their predictions are accurate.
6. **Output Prediction:** Generating the final engagement score.

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
