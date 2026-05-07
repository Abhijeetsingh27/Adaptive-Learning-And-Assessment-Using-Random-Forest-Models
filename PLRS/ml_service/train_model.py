import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import joblib
import os
import json

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'topic_accuracy': np.random.uniform(0, 1, n_samples),
        'avg_time_per_question': np.random.uniform(2, 60, n_samples),
        'retry_rate': np.random.uniform(0, 1, n_samples),
        'skip_rate': np.random.uniform(0, 0.5, n_samples),
        'attempt_count': np.random.randint(1, 10, n_samples),
        'engagement_score': np.zeros(n_samples), # Derived
        'difficulty_level': np.random.choice(['easy', 'medium', 'hard'], n_samples),
        # Targets
        'learning_state': np.random.choice(['focused', 'confusion', 'boredom', 'guessing', 'disengagement'], n_samples),
        'recommended_topic_id': np.random.randint(1, 10, n_samples), # simplified
        'next_score_prediction': np.zeros(n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some meaningful correlation
    df['engagement_score'] = (df['topic_accuracy'] * 0.4) + (1 - df['retry_rate']) * 0.3 + (1 - df['skip_rate']) * 0.3
    df['next_score_prediction'] = df['topic_accuracy'] * 100 + np.random.normal(0, 5, n_samples)
    
    return df

DATA_DIR = 'data'
DATA_FILE = 'simulated_student_learning_data.csv'

def load_data():
    file_path = os.path.join(DATA_DIR, DATA_FILE)
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        return df
    else:
        print("Dataset not found. Generating synthetic data (fallback)...")
        return generate_synthetic_data()

def train():
    df = load_data()
    
    # Check if we are using the specific provided dataset structure
    if 'HRV' in df.columns and 'Emotion' in df.columns:
        print("Detected physiological dataset structure.")
        # Features from the new dataset
        feature_cols = ['HRV', 'Skin_Temperature', 'Expression_Joy', 'Expression_Confusion', 'Steps', 'Session_Duration']
        
        # Ensure all columns exist (fill missing if valid file but partial data - unlikely for provided csv but good practice)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[feature_cols]
        
        # Target A: Classifier (Emotion -> Learning State)
        # Map Emotion labels to our system's Learning States if they differ, or use directly
        # Dataset Emotions: Interest, Happiness, Boredom, Confusion
        # System States: focused, confusion, boredom, guessing, disengagement
        # Mapping: Interest->focused, Happiness->focused, Boredom->boredom, Confusion->confusion
        
        y_class = df['Emotion'].replace({
            'Interest': 'focused',
            'Happiness': 'focused', # or 'flow'
            'Boredom': 'boredom',
            'Confusion': 'confusion'
        })
        
        # Target B: Regressor (Engagement_Level -> Score Prediction Proxy)
        # We will predict Engagement_Level (1-5) as a proxy for "Next Score" or performance
        y_reg = df['Engagement_Level']
        
    else:
        print("Using generic/synthetic feature set.")
        X = df[['topic_accuracy', 'avg_time_per_question', 'retry_rate', 'skip_rate', 'attempt_count', 'engagement_score']]
        y_class = df['learning_state']
        y_reg = df['next_score_prediction']
    
    # Split
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Classifier
    print("Training Classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_class_train)
    
    # Train Regressor
    print("Training Regressor...")
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_scaled, y_reg_train)
    
    # Evaluate
    class_preds = clf.predict(X_test_scaled)
    reg_preds = reg.predict(X_test_scaled)
    
    metrics = {
        'accuracy': accuracy_score(y_class_test, class_preds),
        'precision': precision_score(y_class_test, class_preds, average='weighted', zero_division=0),
        'recall': recall_score(y_class_test, class_preds, average='weighted', zero_division=0),
        'rmse': np.sqrt(mean_squared_error(y_reg_test, reg_preds))
    }
    
    print("Metrics:", metrics)
    
    # Save Models and Scaler
    joblib.dump(clf, os.path.join(MODEL_DIR, 'rf_classifier.pkl'))
    joblib.dump(reg, os.path.join(MODEL_DIR, 'rf_regressor.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    
    # Save feature names to ensure app.py knows what to expect
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("Models saved.")

if __name__ == "__main__":
    train()
