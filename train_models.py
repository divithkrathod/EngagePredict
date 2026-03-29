import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("1. Loading dataset into Pandas DataFrame...")
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "engage_predict_dataset.csv")
df = pd.read_csv(filepath)
print(f"Dataset loaded successfully with {len(df)} user records.\n")

print("2. Preprocessing Data...")
# Features and targets
X = df.drop(['user_id', 'engagement_tier', 'churn_probability'], axis=1)
y_classifier = df['engagement_tier']
y_regressor = df['churn_probability']

# Encode categorical feature 'device_type'
le_device = LabelEncoder()
X['device_type'] = le_device.fit_transform(X['device_type'])

# Standardize features (essential for PCA and K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n3. Training Principal Component Analysis (PCA)...")
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)
explained_var = sum(pca.explained_variance_ratio_) * 100
print(f"PCA reduced dimensions to 3 components, retaining {explained_var:.2f}% of variance.")

print("\n4. Training User Segmentation (K-Means)...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
df['behavioral_cluster'] = clusters
print("K-Means clustering complete. Distinct behavioral groups identified.")

print("\n5. Training Predictive Core - Random Forest Classifier (Engagement Tier)...")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_pca, y_classifier, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_c, y_train_c)
y_pred_c = rf_model.predict(X_test_c)

accuracy = accuracy_score(y_test_c, y_pred_c)
print(f"Random Forest successfully categorized engagement levels with {accuracy * 100:.2f}% accuracy.")

print("\n6. Training Predictive Core - XGBoost Regressor (Churn Probability)...")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_pca, y_regressor, test_size=0.2, random_state=42)

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train_r, y_train_r)
y_pred_r = xgb_model.predict(X_test_r)

rmse = np.sqrt(mean_squared_error(y_test_r, y_pred_r))
print(f"XGBoost successfully predicted churn with a low Root Mean Square Error of {rmse:.4f}.")

print("\n7. Saving trained models...")
os.makedirs('models', exist_ok=True)
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca.pkl')
joblib.dump(kmeans, 'models/kmeans.pkl')
joblib.dump(rf_model, 'models/rf_classifier.pkl')
joblib.dump(xgb_model, 'models/xgb_regressor.pkl')
print("All models successfully saved to the 'models' directory.")
print("PIPELINE EXECUTION 100% COMPLETE.")
