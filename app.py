from flask import Flask, render_template, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import traceback
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
        print(f"Received data: {data}")
        
        # Validate input data
        if not data:
            raise ValueError("No input data provided")
            
        # Check if all required features are present
        for feature in FEATURE_NAMES:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
        
        # Create a dataframe with the input data
        input_data = pd.DataFrame([data], columns=FEATURE_NAMES)
        print(f"Input dataframe: {input_data.head()}")
        
        # Convert all values to appropriate numeric types
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='raise')
        
        # Load model and scaler
        model, scaler = load_model()
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]
        
        print(f"Prediction: {prediction}, Probability: {prediction_proba}")
        
        # Return the result
        return jsonify({
            'prediction': int(prediction),
            'probability': float(prediction_proba),
            'message': 'Heart disease detected' if prediction == 1 else 'No heart disease detected',
            'success': True
        })
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
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
    # Disable threading to avoid issues with Python 3.13
    app.run(debug=True, use_reloader=False, threaded=False)