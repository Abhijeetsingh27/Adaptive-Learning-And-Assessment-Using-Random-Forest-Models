from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import json
from train_model import train

app = Flask(__name__)
MODEL_DIR = 'models'

# Global variables for models
clf = None
reg = None
scaler = None

def load_models():
    global clf, reg, scaler
    try:
        clf = joblib.load(os.path.join(MODEL_DIR, 'rf_classifier.pkl'))
        reg = joblib.load(os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        print("Models loaded successfully.")
    except FileNotFoundError:
        print("Models not found. Training initial models...")
        train()
        clf = joblib.load(os.path.join(MODEL_DIR, 'rf_classifier.pkl'))
        reg = joblib.load(os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

load_models()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Load feature names if available to support flexible schemas
        feature_names_path = os.path.join(MODEL_DIR, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
        else:
            # Fallback for old model
            feature_names = ['topic_accuracy', 'avg_time_per_question', 'retry_rate', 'skip_rate', 'attempt_count', 'engagement_score']
            
        print(f"Model expects features: {feature_names}")

        # Prepare features in correct order
        features = []
        for feat in feature_names:
            val = data.get(feat)
            if val is None:
                # Default values if missing
                features.append(0)
            else:
                features.append(float(val))
                
        features_array = np.array([features])
        features_scaled = scaler.transform(features_array)
        
        class_pred = clf.predict(features_scaled)[0]
        reg_pred = reg.predict(features_scaled)[0]
        
        # Heuristic for recommended level
        # If state is 'confusion' or 'guessing', recommend easier level or revisit topic
        # If 'boredom' (implies too easy) or 'focused', recommend harder level if score prediction is good
        
        rec_level = 'medium' # Default
        if class_pred in ['confusion', 'guessing', 'frustration']:
            rec_level = 'easy'
        elif class_pred in ['boredom', 'focused']:
            if float(reg_pred) > 75:
                rec_level = 'hard'
                
        return jsonify({
            'learning_state': class_pred,
            'next_score_prediction': float(reg_pred),
            'recommended_level': rec_level
        })
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain_endpoint():
    train()
    load_models()
    return jsonify({'status': 'Retraining complete', 'metrics': get_metrics_data()})

@app.route('/metrics', methods=['GET'])
def metrics():
    return jsonify(get_metrics_data())

def get_metrics_data():
    try:
        with open(os.path.join(MODEL_DIR, 'metrics.json'), 'r') as f:
            return json.load(f)
    except:
        return {}

if __name__ == '__main__':
    app.run(port=5000, debug=True)
