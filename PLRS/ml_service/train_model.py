import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.feature_selection import SelectKBest, f_classif
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

def add_simple_features(df):
    """Add simple engineered features (no date/time)"""
    
    print("Adding simple engineered features...")
    
    # Only add features if we have the physiological dataset
    if 'HRV' in df.columns and 'Skin_Temperature' in df.columns:
        # 1. Simple ratios
        df['hrv_temp_ratio'] = df['HRV'] / (df['Skin_Temperature'].abs() + 0.001)
        df['joy_confusion_ratio'] = df['Expression_Joy'] / (df['Expression_Confusion'] + 0.001)
        
        # 2. Simple combinations
        df['expression_balance'] = df['Expression_Joy'] - df['Expression_Confusion']
        df['expression_total'] = df['Expression_Joy'] + df['Expression_Confusion']
        
        # 3. Activity features
        df['steps_per_minute'] = df['Steps'] / (df['Session_Duration'] + 1)
        df['activity_intensity'] = df['Steps'] * df['Session_Duration']
        
        # 4. Physiological stress index
        df['physiological_stress'] = np.sqrt(df['HRV']**2 + df['Skin_Temperature']**2)
        
        # 5. Emotional arousal (simple but effective)
        df['emotional_arousal'] = df['Expression_Joy'] + df['Expression_Confusion']
        
        # 6. Polynomial features for non-linear relationships
        df['hrv_squared'] = df['HRV'] ** 2
        df['temp_squared'] = df['Skin_Temperature'] ** 2
        df['hrv_temp_product'] = df['HRV'] * df['Skin_Temperature']
        
        # 8. Interaction features
        df['physio_emotion_interaction'] = df['physiological_stress'] * df['emotional_arousal']
        
        print(f"Added 11 engineered features")
    
    return df

def load_data():
    file_path = os.path.join(DATA_DIR, DATA_FILE)
    if os.path.exists(file_path):
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        df = add_simple_features(df)
        return df
    else:
        print("Dataset not found. Generating synthetic data (fallback)...")
        return generate_synthetic_data()

def train():
    df = load_data()
    
    # Check if we are using the specific provided dataset structure
    if 'HRV' in df.columns and 'Emotion' in df.columns:
        print("Detected physiological dataset structure.")
        # Original features + engineered features
        base_features = ['HRV', 'Skin_Temperature', 'Expression_Joy', 'Expression_Confusion', 'Steps', 'Session_Duration']
        engineered_features = ['hrv_temp_ratio', 'joy_confusion_ratio', 'expression_balance', 'expression_total', 
                              'steps_per_minute', 'activity_intensity', 'physiological_stress', 'emotional_arousal',
                              'hrv_squared', 'temp_squared', 'hrv_temp_product',
                              'physio_emotion_interaction']
        
        feature_cols = base_features + engineered_features
        
        # Ensure all columns exist (fill missing if valid file but partial data)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[feature_cols]
        print(f"Using {len(feature_cols)} features ({len(base_features)} original + {len([f for f in engineered_features if f in df.columns])} engineered)")
        
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
    
    # Feature selection to keep only the best features
    print("Selecting best features...")
    selector = SelectKBest(score_func=f_classif, k=15)  # Keep top 15 features
    X_train_selected = selector.fit_transform(X_train_scaled, y_class_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} best features: {selected_features}")
    
    # Train Ensemble Classifier for better accuracy
    print("Training Ensemble Classifier...")
    
    # Create multiple diverse classifiers
    clf1 = RandomForestClassifier(
        n_estimators=200, max_depth=15, max_features='sqrt', 
        class_weight='balanced', random_state=42
    )
    clf2 = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features='log2', 
        class_weight='balanced', random_state=123
    )
    clf3 = RandomForestClassifier(
        n_estimators=250, max_depth=18, max_features=None, 
        class_weight='balanced', random_state=456
    )
    
    # Create voting ensemble
    clf = VotingClassifier(
        estimators=[('rf1', clf1), ('rf2', clf2), ('rf3', clf3)],
        voting='soft'  # Use probability averaging
    )
    
    # Fit ensemble on selected features
    clf.fit(X_train_selected, y_class_train)
    
    # Cross-validation score for confidence
    cv_scores = cross_val_score(clf, X_train_selected, y_class_train, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train Regressor (optimized parameters)
    print("Training Regressor...")
    reg = RandomForestRegressor(
        n_estimators=200,           # More trees
        max_depth=12,               # Reasonable depth
        min_samples_split=4,        # Prevent overfitting
        min_samples_leaf=2,         # Prevent overfitting
        max_features='sqrt',        # Better feature selection
        random_state=42
    )
    reg.fit(X_train_selected, y_reg_train)  # Use selected features too
    
    # Evaluate
    class_preds = clf.predict(X_test_selected)
    reg_preds = reg.predict(X_test_selected)
    
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
    joblib.dump(selector, os.path.join(MODEL_DIR, 'selector.pkl'))
    
    # Save feature names to ensure app.py knows what to expect
    feature_names = X.columns.tolist()
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))
    
    with open(os.path.join(MODEL_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)
        
    print("Models saved.")

if __name__ == "__main__":
    train()
