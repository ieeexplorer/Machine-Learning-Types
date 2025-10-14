import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Sample data: House features -> Price category (Cheap, Medium, Expensive)
data = {
    'size_sqft': [1200, 1500, 1800, 2000, 2400, 800, 3000, 1600, 1400, 2200],
    'bedrooms': [2, 3, 3, 4, 4, 1, 5, 3, 2, 4],
    'bathrooms': [1, 2, 2, 2, 3, 1, 4, 2, 1, 3],
    'location_score': [3, 5, 6, 8, 9, 2, 10, 4, 3, 7],
    'price_category': ['Cheap', 'Medium', 'Medium', 'Expensive', 'Expensive', 
                      'Cheap', 'Expensive', 'Medium', 'Cheap', 'Expensive']
}

df = pd.DataFrame(data)
print("LABELED TRAINING DATA:")
print(df.head())

# Prepare features (X) and labels (y)
X = df[['size_sqft', 'bedrooms', 'bathrooms', 'location_score']]
y = df['price_category']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  # LEARNING FROM LABELED DATA

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"\nClassification Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict on new unseen data
new_house = [[1700, 3, 2, 6]]  # Features of a new house
predicted_category = model.predict(new_house)
print(f"Predicted price category for new house: {predicted_category[0]}")
