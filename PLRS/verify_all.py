import requests
import sys

def check_flask():
    print("Checking Flask ML Service...")
    try:
        # Simple health check or just connectivity
        response = requests.get("http://127.0.0.1:5000", timeout=5) 
        # Note: The ML service might only accept POST on /predict, so a 404 or 405 on / is fine as long as it connects.
        # But let's use the predict endpoint to be sure, or just catch connection error.
        print(f"Flask Service responded with code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(" Flask Service NOT reachable on port 5000")
        return False
    except Exception as e:
        print(f" Flask Service error: {e}")
        return False

def check_django():
    print("Checking Django Backend...")
    try:
        response = requests.get("http://127.0.0.1:8000", timeout=5)
        print(f"Django Backend responded with code: {response.status_code}")
        return True
    except requests.exceptions.ConnectionError:
        print(" Django Backend NOT reachable on port 8000")
        return False
    except Exception as e:
        print(f" Django Backend error: {e}")
        return False

if __name__ == "__main__":
    flask_ok = check_flask()
    django_ok = check_django()
    
    if flask_ok and django_ok:
        print("\n BOTH SERVICES ARE RUNNING!")
        sys.exit(0)
    else:
        print("\n ONE OR MORE SERVICES FAILED.")
        sys.exit(1)
