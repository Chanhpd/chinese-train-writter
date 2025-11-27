# Flask API for Siamese Network - Chấm điểm chữ Hán
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Define custom functions (must be defined before loading model)
def euclidean_distance(vects):
    """Compute Euclidean distance between two vectors"""
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    """Contrastive loss function"""
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Load model with custom objects
MODEL_PATH = "trained/siamese_model_full.h5"
print(f"Loading model from {MODEL_PATH}...")
model = keras.models.load_model(
    MODEL_PATH, 
    custom_objects={
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss
    },
    compile=False
)
print("✅ Model loaded successfully!")

IMG_SIZE = 128

def preprocess_image(image_data):
    """
    Preprocess image from base64 or file
    Returns: numpy array (128, 128, 1) normalized to [0, 1]
    """
    # Decode base64 if needed
    if isinstance(image_data, str):
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
    else:
        img = Image.open(image_data)
    
    # Convert to grayscale
    img = img.convert('L')
    
    # Resize
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert to numpy and normalize
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add channel dimension
    img_array = np.expand_dims(img_array, axis=-1)
    
    return img_array

@app.route('/')
def home():
    return jsonify({
        'message': 'Siamese Network API - Chấm điểm chữ Hán',
        'version': '1.0',
        'endpoints': {
            '/score': 'POST - Compare two images and return score',
            '/health': 'GET - Check API health'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/score', methods=['POST'])
def score_images():
    """
    Compare two images and return similarity score (0-100)
    
    Request body (JSON):
    {
        "image_reference": "base64_string or file",
        "image_user": "base64_string or file"
    }
    
    Response:
    {
        "distance": 0.234,
        "score": 76.6,
        "interpretation": "Good"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'image_reference' not in data or 'image_user' not in data:
            return jsonify({
                'error': 'Missing required fields: image_reference, image_user'
            }), 400
        
        # Preprocess images
        img_ref = preprocess_image(data['image_reference'])
        img_user = preprocess_image(data['image_user'])
        
        # Check if user actually drew something (basic sanity check)
        user_pixel_ratio = np.mean(img_user < 0.9)
        ref_pixel_ratio = np.mean(img_ref < 0.9)
        
        print(f"[DEBUG] Reference pixel ratio: {ref_pixel_ratio:.4f} ({ref_pixel_ratio*100:.2f}%)")
        print(f"[DEBUG] User pixel ratio: {user_pixel_ratio:.4f} ({user_pixel_ratio*100:.2f}%)")
        
        # Only reject if user drew almost nothing (less than 0.5% of image)
        if user_pixel_ratio < 0.005:
            print(f"[DEBUG] REJECTED - Canvas is empty")
            return jsonify({
                'success': True,
                'distance': 1.0,
                'score': 0.0,
                'interpretation': "Empty canvas - Please draw the character"
            })
        
        print(f"[DEBUG] PASSED - Canvas has content")
        
        # Add batch dimension
        img_ref = np.expand_dims(img_ref, axis=0)
        img_user = np.expand_dims(img_user, axis=0)
        
        # Predict distance
        distance = model.predict([img_ref, img_user], verbose=0)[0][0]
        distance = float(distance)
        
        print(f"[DEBUG] Raw distance from model: {distance:.4f}")
        
        # Convert to score (0-100)
        # Apply exponential scaling to amplify small differences
        # Based on observed range: 0.15 (excellent) to 0.60 (poor)
        
        # Use power function to amplify differences
        normalized_distance = min(distance / 0.6, 1.0)  # Normalize to [0, 1]
        penalty = (normalized_distance ** 2)  # Square to amplify differences
        score = 100 - (penalty * 100)
        
        score = max(0, min(100, score))
        
        print(f"[DEBUG] Final score: {score:.2f}")
        
        # Interpretation
        if score >= 90:
            interpretation = "Excellent"
        elif score >= 75:
            interpretation = "Very Good"
        elif score >= 60:
            interpretation = "Good"
        elif score >= 40:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        return jsonify({
            'success': True,
            'distance': round(distance, 4),
            'score': round(score, 2),
            'interpretation': interpretation
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 Siamese Network API Server")
    print("="*50)
    print(f"Model: {MODEL_PATH}")
    print(f"Server: http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET  /         : API info")
    print("  - GET  /health   : Health check")
    print("  - POST /score    : Score handwriting")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
