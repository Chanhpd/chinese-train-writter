# Convert Keras model to TensorFlow.js format
import tensorflowjs as tfjs
import os

MODEL_PATH = "trained/siamese_model_full.h5"
OUTPUT_DIR = "web_demo/tfjs_model"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Convert model
print(f"Converting {MODEL_PATH} to TensorFlow.js format...")
tfjs.converters.save_keras_model(
    MODEL_PATH,
    OUTPUT_DIR
)

print(f"✅ Model converted successfully!")
print(f"📁 Output directory: {OUTPUT_DIR}")
print(f"Files created:")
for file in os.listdir(OUTPUT_DIR):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, file)) / 1024
    print(f"  - {file} ({size:.1f} KB)")
