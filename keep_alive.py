# Keep Render API warm - Run this on a cron job (e.g., cron-job.org)
import requests
import time

API_URL = "https://your-app.onrender.com/health"  # Replace with your Render URL

def ping_api():
    try:
        response = requests.get(API_URL, timeout=10)
        print(f"✅ Ping successful: {response.status_code}")
    except Exception as e:
        print(f"❌ Ping failed: {e}")

if __name__ == "__main__":
    while True:
        ping_api()
        time.sleep(300)  # Ping every 5 minutes
