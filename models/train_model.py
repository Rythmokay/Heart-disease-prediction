import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train():
    """
    Trains a heart disease prediction model using the UCI Heart Disease dataset
    and saves it to disk.
    """
    # Download the heart disease dataset (you can replace with your own dataset)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    # Column names for the dataset
    column_names = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load the dataset
    print("Loading dataset...")
    df = pd.read_csv(url, header=None, names=column_names, na_values='?')
    
    # Handle missing values
    df = df.dropna()
    
    # Convert target to binary (0 = no disease, 1 = disease)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training model...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Classification report:\n{report}")
    
    # Save both the model and scaler
    print("Saving model...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    joblib.dump(model, os.path.join(model_dir, 'heart_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    
    print("Model trained and saved successfully!")
    return model, scaler

if __name__ == "__main__":
    train()