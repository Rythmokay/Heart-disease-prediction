from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
from models.train_model import train

app = Flask(__name__)

# Model file paths
MODEL_PATH = os.path.join('models', 'heart_model.pkl')
SCALER_PATH = os.path.join('models', 'scaler.pkl')

# Load model and scaler (or train if not available)
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model not found. Training new model...")
        model, scaler = train()
    return model, scaler

# Feature names in the correct order
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    try:
        # Get data from request
        data = request.json
        
        # Create a dataframe with the input data
        input_data = pd.DataFrame([data], columns=FEATURE_NAMES)
        
        # Load model and scaler
        model, scaler = load_model()
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        # Return the result
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'message': 'Heart disease detected' if prediction == 1 else 'No heart disease detected',
            'success': True
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    # Make sure model is loaded or trained before starting the app
    load_model()
    app.run(debug=True)