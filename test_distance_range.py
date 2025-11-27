# Test script to check distance range from model
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# Define custom functions
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
print(f"Loading model from {MODEL_PATH}...")
model = keras.models.load_model(
    MODEL_PATH, 
    custom_objects={
        'euclidean_distance': euclidean_distance,
        'contrastive_loss': contrastive_loss
    },
    compile=False
)
print("✅ Model loaded successfully!\n")

IMG_SIZE = 128

def create_test_image(quality='good'):
    """Create test images with different quality levels"""
    img = Image.new('L', (IMG_SIZE, IMG_SIZE), color=255)
    draw = ImageDraw.Draw(img)
    
    if quality == 'perfect':
        # Perfect match - same image
        draw.line([(30, 30), (100, 100)], fill=0, width=10)
        draw.line([(100, 30), (30, 100)], fill=0, width=10)
    elif quality == 'good':
        # Good - slight variation
        draw.line([(32, 28), (98, 102)], fill=0, width=10)
        draw.line([(102, 28), (28, 102)], fill=0, width=10)
    elif quality == 'fair':
        # Fair - more variation
        draw.line([(35, 25), (95, 105)], fill=0, width=12)
        draw.line([(105, 25), (25, 105)], fill=0, width=8)
    elif quality == 'poor':
        # Poor - very different
        draw.line([(20, 40), (80, 120)], fill=0, width=15)
        draw.rectangle([(50, 20), (100, 70)], outline=0, width=5)
    elif quality == 'random':
        # Completely different
        draw.ellipse([(30, 30), (100, 100)], outline=0, width=10)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)
    return img_array

# Test different scenarios
print("="*60)
print("TESTING DISTANCE RANGES")
print("="*60)

scenarios = [
    ('Perfect Match (Same Image)', 'perfect', 'perfect'),
    ('Excellent (Very Similar)', 'good', 'good'),
    ('Good (Similar with variation)', 'good', 'fair'),
    ('Fair (Some differences)', 'good', 'poor'),
    ('Poor (Very Different)', 'good', 'random'),
]

results = []

for name, ref_quality, user_quality in scenarios:
    img_ref = create_test_image(ref_quality)
    img_user = create_test_image(user_quality)
    
    # Add batch dimension
    img_ref = np.expand_dims(img_ref, axis=0)
    img_user = np.expand_dims(img_user, axis=0)
    
    # Predict
    distance = model.predict([img_ref, img_user], verbose=0)[0][0]
    distance = float(distance)
    
    # Current formula
    old_score = max(0, min(100, (1 - distance) * 100))
    
    # New formula
    if distance <= 0.1:
        new_score = 100
    elif distance <= 0.3:
        new_score = 100 - (distance - 0.1) * 250
    elif distance <= 0.6:
        new_score = 50 - (distance - 0.3) * 100
    else:
        new_score = max(0, 20 - (distance - 0.6) * 20)
    new_score = max(0, min(100, new_score))
    
    results.append((name, distance, old_score, new_score))
    print(f"\n{name}:")
    print(f"  Distance: {distance:.4f}")
    print(f"  Old Score: {old_score:.1f}")
    print(f"  New Score: {new_score:.1f}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"{'Scenario':<35} {'Distance':<12} {'Old Score':<12} {'New Score'}")
print("-"*60)
for name, dist, old, new in results:
    print(f"{name:<35} {dist:<12.4f} {old:<12.1f} {new:.1f}")

print("\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)
print("Based on these results, you should adjust the scoring formula")
print("in api_server.py to match your expectations.")
print("\nDistance ranges observed:")
print(f"  - Perfect match: ~{results[0][1]:.4f}")
print(f"  - Good quality: ~{results[1][1]:.4f} - {results[2][1]:.4f}")
print(f"  - Poor quality: ~{results[3][1]:.4f} - {results[4][1]:.4f}")
