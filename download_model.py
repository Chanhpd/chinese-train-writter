# Download Model Script
# Chạy script này trong Build Command của Render

import os
import urllib.request

MODEL_URL = "https://your-cloud-storage.com/siamese_model_full.h5"  # Thay bằng link thực
MODEL_PATH = "trained/siamese_model_full.h5"

print("📥 Downloading model...")
os.makedirs("trained", exist_ok=True)
urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
print(f"✅ Model downloaded to {MODEL_PATH}")
