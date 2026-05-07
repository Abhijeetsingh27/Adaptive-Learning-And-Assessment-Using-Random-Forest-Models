import requests
import json
import random

def test_ml_service():
    print("Testing ML Service Integration...")
    url = "http://127.0.0.1:5000/predict"
    
    # Simulate data exactly as views.py does
    accuracy = 0.85 
    time_spent = 300
    
    # Heuristic simulation (copied from views.py)
    base_joy = accuracy
    base_confusion = 1.0 - accuracy
    simulated_hrv = 60 + (accuracy * 20) + random.uniform(-5, 5)
    simulated_skin_temp = 36.5 + random.uniform(-0.5, 0.5)
    simulated_joy = max(0, min(1, base_joy + random.uniform(-0.1, 0.1)))
    simulated_confusion = max(0, min(1, base_confusion + random.uniform(-0.1, 0.1)))
    simulated_steps = 50 + 85 # score assumption
    
    payload = {
        'HRV': simulated_hrv,
        'Skin_Temperature': simulated_skin_temp,
        'Expression_Joy': simulated_joy,
        'Expression_Confusion': simulated_confusion,
        'Steps': simulated_steps,
        'Session_Duration': time_spent,
        'topic_accuracy': accuracy,
        'avg_time_per_question': time_spent,
        'retry_rate': 0, 
        'skip_rate': 0, 
        'attempt_count': 1, 
        'engagement_score': (accuracy * 0.5) + 0.5
    }
    
    print(f"Sending payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("\n Success! Received Response:")
            print(json.dumps(data, indent=2))
            
            # Validation
            required_keys = ['learning_state', 'next_score_prediction', 'recommended_level']
            missing = [k for k in required_keys if k not in data]
            
            if missing:
                print(f" FAILED: Missing keys in response: {missing}")
            else:
                print(" Response structure is correct.")
                print(f"   - Learning State: {data['learning_state']}")
                print(f"   - Engagement (Next Score): {data['next_score_prediction']}")
                print(f"   - Recommended Level: {data['recommended_level']}")
        else:
            print(f" FAILED: Status Code {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f" FAILED: Connection Check. Is Flask running? Error: {e}")

if __name__ == "__main__":
    test_ml_service()
