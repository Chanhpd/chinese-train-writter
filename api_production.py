# Flask API Production - Handwriting Scoring System
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
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow Flutter app to call API

# Define custom functions for model loading
def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Load model
MODEL_PATH = "trained/siamese_model_full.h5"
logger.info(f"Loading model from {MODEL_PATH}...")
try:
    model = keras.models.load_model(
        MODEL_PATH, 
        custom_objects={
            'euclidean_distance': euclidean_distance,
            'contrastive_loss': contrastive_loss
        },
        compile=False
    )
    logger.info("✅ Model loaded successfully!")
except Exception as e:
    logger.error(f"❌ Failed to load model: {str(e)}")
    model = None

IMG_SIZE = 128

def preprocess_image(image_data):
    """
    Preprocess image from base64 string
    Args:
        image_data: base64 string (with or without data:image prefix)
    Returns:
        numpy array (128, 128, 1) normalized to [0, 1]
    """
    try:
        # Remove data URL prefix if present
        if isinstance(image_data, str) and image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64
        image_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        img = img.convert('L')
        
        # Resize
        img = img.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0
        
        # Add channel dimension
        img_array = np.expand_dims(img_array, axis=-1)
        
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/')
def home():
    """API documentation"""
    return jsonify({
        'name': 'Handwriting Scoring API',
        'version': '1.0.0',
        'status': 'online',
        'model_loaded': model is not None,
        'endpoints': {
            'GET /': 'API documentation',
            'GET /health': 'Health check',
            'POST /score': 'Score handwriting comparison',
            'POST /batch_score': 'Score multiple characters at once'
        },
        'usage': {
            '/score': {
                'method': 'POST',
                'content_type': 'application/json',
                'body': {
                    'character': 'Chinese character (optional, for logging)',
                    'image_reference': 'base64 string of reference image',
                    'image_user': 'base64 string of user drawing'
                },
                'response': {
                    'success': True,
                    'distance': 0.234,
                    'score': 85.5,
                    'interpretation': 'Very Good'
                }
            }
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/score', methods=['POST'])
def score_handwriting():
    """
    Score a single handwriting comparison
    
    Request body (JSON):
    {
        "character": "好",  // Optional - for logging
        "image_reference": "base64_string",
        "image_user": "base64_string"
    }
    
    Response:
    {
        "success": true,
        "character": "好",
        "distance": 0.234,
        "score": 85.5,
        "interpretation": "Very Good",
        "timestamp": "2025-11-27T21:45:00"
    }
    """
    try:
        # Check model
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        if 'image_reference' not in data or 'image_user' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields: image_reference, image_user'
            }), 400
        
        character = data.get('character', '')
        
        # Preprocess images
        img_ref = preprocess_image(data['image_reference'])
        img_user = preprocess_image(data['image_user'])
        
        # Check if user drew something
        user_pixel_ratio = np.mean(img_user < 0.9)
        
        if user_pixel_ratio < 0.005:
            logger.warning(f"Empty canvas detected for character: {character}")
            return jsonify({
                'success': True,
                'character': character,
                'distance': 1.0,
                'score': 0.0,
                'interpretation': 'Empty',
                'message': 'Canvas is empty - Please draw the character',
                'timestamp': datetime.now().isoformat()
            })
        
        # Add batch dimension
        img_ref = np.expand_dims(img_ref, axis=0)
        img_user = np.expand_dims(img_user, axis=0)
        
        # Predict distance
        distance = model.predict([img_ref, img_user], verbose=0)[0][0]
        distance = float(distance)
        
        # Convert to score (0-100)
        normalized_distance = min(distance / 0.6, 1.0)
        penalty = (normalized_distance ** 2)
        score = 100 - (penalty * 100)
        score = max(0, min(100, score))
        
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
        
        logger.info(f"Scored '{character}': distance={distance:.4f}, score={score:.2f}")
        
        return jsonify({
            'success': True,
            'character': character,
            'distance': round(distance, 4),
            'score': round(score, 2),
            'interpretation': interpretation,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in /score: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch_score', methods=['POST'])
def batch_score():
    """
    Score multiple characters at once
    
    Request body (JSON):
    {
        "items": [
            {
                "character": "好",
                "image_reference": "base64_string",
                "image_user": "base64_string"
            },
            {
                "character": "学",
                "image_reference": "base64_string",
                "image_user": "base64_string"
            }
        ]
    }
    
    Response:
    {
        "success": true,
        "results": [
            {
                "character": "好",
                "distance": 0.234,
                "score": 85.5,
                "interpretation": "Very Good"
            },
            ...
        ],
        "summary": {
            "total": 2,
            "average_score": 82.3
        }
    }
    """
    try:
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        data = request.get_json()
        
        if not data or 'items' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required field: items'
            }), 400
        
        items = data['items']
        results = []
        total_score = 0
        
        for item in items:
            try:
                character = item.get('character', '')
                img_ref = preprocess_image(item['image_reference'])
                img_user = preprocess_image(item['image_user'])
                
                # Check canvas
                user_pixel_ratio = np.mean(img_user < 0.9)
                if user_pixel_ratio < 0.005:
                    results.append({
                        'character': character,
                        'distance': 1.0,
                        'score': 0.0,
                        'interpretation': 'Empty'
                    })
                    continue
                
                # Predict
                img_ref_batch = np.expand_dims(img_ref, axis=0)
                img_user_batch = np.expand_dims(img_user, axis=0)
                distance = float(model.predict([img_ref_batch, img_user_batch], verbose=0)[0][0])
                
                # Score
                normalized_distance = min(distance / 0.6, 1.0)
                score = 100 - ((normalized_distance ** 2) * 100)
                score = max(0, min(100, score))
                
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
                
                results.append({
                    'character': character,
                    'distance': round(distance, 4),
                    'score': round(score, 2),
                    'interpretation': interpretation
                })
                
                total_score += score
                
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
                results.append({
                    'character': item.get('character', ''),
                    'error': str(e)
                })
        
        average_score = total_score / len(results) if results else 0
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': {
                'total': len(results),
                'average_score': round(average_score, 2)
            },
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error in /batch_score: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Handwriting Scoring API - Production")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Status: {'✅ Ready' if model else '❌ Model not loaded'}")
    print(f"Server: http://0.0.0.0:5000")
    print("\nEndpoints:")
    print("  - GET  /          : API documentation")
    print("  - GET  /health    : Health check")
    print("  - POST /score     : Score single character")
    print("  - POST /batch_score : Score multiple characters")
    print("="*60 + "\n")
    
    # Production settings
    app.run(
        debug=False,  # Disable debug mode for production
        host='0.0.0.0',  # Listen on all interfaces
        port=5000,
        threaded=True  # Handle multiple requests
    )
