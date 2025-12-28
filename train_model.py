import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import json

def train_and_save():
    print("Loading data...")
    # Load dataset
    cd = pd.read_csv("Credit.csv")
    
    # Preprocessing
    # Map Class to 1 (Good) and 0 (Bad)
    if 'Class' in cd.columns:
        cd['Class'] = cd['Class'].map({'Good': 1, 'Bad': 0})
    
    # Define X and Y
    X = cd.drop('Class', axis=1)
    y = cd['Class']
    
    # Save column names to ensure consistency in App
    columns = list(X.columns)
    with open('model_columns.json', 'w') as f:
        json.dump(columns, f)
    print(f"Saved {len(columns)} feature names to model_columns.json")
    
    # Train Model - Switching to Random Forest
    print("Training Random Forest Model...")
    # Using specific parameters often good for credit data, or default if unspecified
    model = RandomForestClassifier(n_estimators=100, random_state=42) 
    model.fit(X, y)
    
    # Save Model
    joblib.dump(model, 'model.pkl')
    print("Model saved to model.pkl")
    
    # Verify accuracy
    acc = model.score(X, y)
    print(f"Training Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_and_save()
