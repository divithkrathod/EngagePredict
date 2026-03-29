# EngagePredict Project Documentation

## Techniques
The project employs machine learning techniques including:
- Ensemble Learning: Combination of Logistic Regression, Random Forest Classifier, and K-Nearest Neighbors (KNN) models
- Feature Engineering: Extraction of content features such as caption length, hashtag count, posting time, media quality, and platform-specific parameters
- Multiclass Classification: Predicting engagement levels (Low, Medium, High) using weighted voting from ensemble models

## Dataset Used
The project utilizes synthetic data generated programmatically to simulate real-world social media engagement patterns. The dataset includes features like:
- Caption characteristics (length, emojis, call-to-action presence)
- Hashtag usage and ratios
- Posting timing (hour, day of week, peak hours)
- Media properties (resolution, orientation, quality)
- Platform-specific configurations for Instagram, TikTok, YouTube, Twitter, and Facebook

## Methodology
The methodology involves:
1. Data Generation: Creating synthetic training data with realistic engagement patterns
2. Feature Extraction: Processing user inputs to extract relevant features
3. Model Training: Training individual ML models (Logistic Regression, Random Forest, KNN) on the synthetic dataset
4. Ensemble Prediction: Combining model predictions using weighted voting (30% Logistic, 40% Random Forest, 30% KNN)
5. Recommendation Generation: Providing platform-specific tips based on prediction results and best practices

## Implementation
The system is implemented as a full-stack web application with three main components:
- Frontend: React 18 with Vite, Tailwind CSS for responsive UI
- Backend: Node.js with Express.js, Firebase for authentication and data storage
- ML Service: Python FastAPI service handling model inference, media analysis, and recommendations

## Problem Statement
The problem objective is to predict social media post engagement levels by analyzing content features and platform characteristics in order to provide actionable recommendations for optimizing posting strategies and improving content reach.

## SDG Relevance
This project aligns with Sustainable Development Goal 9 (SDG 9): Industry, Innovation and Infrastructure, specifically:
- Target 9.5: Enhance scientific research, upgrade the technological capabilities of industrial sectors in all countries, in particular developing countries, including, by 2030, encouraging innovation and substantially increasing the number of research and development workers per 1 million people and public and private research and development spending
- Target 9.B: Support domestic technology development, research and innovation in developing countries, including by ensuring a conducive policy environment for, inter alia, industrial diversification and value addition to commodities

The project demonstrates technological innovation in social media analytics and provides tools that can enhance digital communication strategies for businesses and individuals.

## Three Brief Objectives Used in This Project
1. Predict engagement potential of social media content using machine learning models
2. Analyze media files for quality and optimization recommendations
3. Provide personalized, platform-specific suggestions to improve content performance