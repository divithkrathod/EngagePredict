import csv
import random
import os

filename = "engage_predict_dataset.csv"
filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

# Dataset configuration
num_records = 1000

# Generating data
data = []
# Header
data.append([
    "user_id", "session_duration_minutes", "pages_visited", "click_through_rate", 
    "historical_likes", "historical_comments", "days_since_last_login", 
    "device_type", "engagement_tier", "churn_probability" # Target variables
])

for i in range(1, num_records + 1):
    user_id = f"USER_{i:04d}"
    
    # Simulate somewhat realistic correlations based on engagement tiers
    engagement_level = random.choice(["Low", "Medium", "High"])
    
    if engagement_level == "High":
        session_duration = round(random.uniform(20.0, 120.0), 1)
        pages_visited = random.randint(10, 50)
        ctr = round(random.uniform(0.15, 0.40), 3) # 15% to 40% CTR
        likes = random.randint(20, 150)
        comments = random.randint(5, 50)
        days_since_last = random.randint(0, 3)
        churn_prob = round(random.uniform(0.01, 0.15), 3) # Very low churn risk
    elif engagement_level == "Medium":
        session_duration = round(random.uniform(5.0, 30.0), 1)
        pages_visited = random.randint(3, 15)
        ctr = round(random.uniform(0.05, 0.20), 3)
        likes = random.randint(5, 25)
        comments = random.randint(0, 10)
        days_since_last = random.randint(2, 10)
        churn_prob = round(random.uniform(0.10, 0.40), 3) # Moderate churn risk
    else: # Low engagement
        session_duration = round(random.uniform(0.5, 10.0), 1)
        pages_visited = random.randint(1, 4)
        ctr = round(random.uniform(0.01, 0.08), 3)
        likes = random.randint(0, 5)
        comments = random.randint(0, 2)
        days_since_last = random.randint(7, 45) # Hasn't logged in for a while
        churn_prob = round(random.uniform(0.55, 0.95), 3) # High churn risk
        
    device = random.choices(["Mobile", "Desktop", "Tablet"], weights=[60, 30, 10])[0]
    
    data.append([
        user_id, session_duration, pages_visited, ctr, 
        likes, comments, days_since_last, device, 
        engagement_level, churn_prob
    ])

with open(filepath, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f"Successfully generated {num_records} synthesized user records in '{filename}'.")
print(f"Location: {filepath}")
