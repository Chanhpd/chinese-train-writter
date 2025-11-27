# ========================================
# SIAMESE NETWORK - TRAIN ON COLAB
# Copy toàn bộ file này vào Google Colab
# ========================================

# BƯỚC 1: Upload dataset_pairs lên Colab (nén thành zip trước)
# Hoặc mount Google Drive nếu đã upload lên Drive

# BƯỚC 2: Install dependencies (chỉ cần chạy 1 lần)
# !pip install tensorflow pillow numpy matplotlib

# BƯỚC 3: Import libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
import tensorflow.keras.backend as K
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ========================================
# CONFIG
# ========================================
DATASET_DIR = "dataset_pairs"  # Thay đổi nếu bạn đặt tên khác
IMG_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "pairs.csv")
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001

# ========================================
# 1. LOAD DATASET
# ========================================
print("\n=== LOADING DATASET ===")
df = pd.read_csv(CSV_FILE)
print(f"Total pairs: {len(df)}")
print(f"Positive pairs: {len(df[df['label']==1])}")
print(f"Negative pairs: {len(df[df['label']==0])}")

# Split train/validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
print(f"Train pairs: {len(train_df)}, Val pairs: {len(val_df)}")

# ========================================
# 2. DATA LOADER
# ========================================
def load_image(img_path):
    """Load and preprocess image"""
    img = Image.open(img_path).convert('L')  # Grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img

def create_dataset(dataframe, img_dir, batch_size=32, shuffle=True):
    """Create tf.data.Dataset for efficient loading"""
    def load_pair(row):
        img_a = load_image(os.path.join(img_dir, row['img_a']))
        img_b = load_image(os.path.join(img_dir, row['img_b']))
        label = np.float32(row['label'])
        return (img_a, img_b), label
    
    # Pre-load all data into memory (dataset is small ~400MB)
    data_a = []
    data_b = []
    labels = []
    
    for _, row in dataframe.iterrows():
        img_a = load_image(os.path.join(img_dir, row['img_a']))
        img_b = load_image(os.path.join(img_dir, row['img_b']))
        data_a.append(img_a)
        data_b.append(img_b)
        labels.append(row['label'])
    
    data_a = np.array(data_a, dtype=np.float32)
    data_b = np.array(data_b, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(((data_a, data_b), labels))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

# ========================================
# 3. BUILD SIAMESE NETWORK
# ========================================
print("\n=== BUILDING MODEL ===")

def euclidean_distance(vects):
    """Compute Euclidean distance between two vectors"""
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def contrastive_loss(y_true, y_pred, margin=1.0):
    """
    Contrastive loss function
    y_true: 1 if similar (positive pair), 0 if dissimilar (negative pair)
    y_pred: distance between embeddings
    """
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_base_network(input_shape):
    """Base CNN for feature extraction"""
    input_layer = Input(shape=input_shape)
    
    # Convolutional layers
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation=None)(x)  # Embedding vector (128-dim)
    
    return Model(input_layer, x, name='base_network')

# Create Siamese model
input_shape = (IMG_SIZE, IMG_SIZE, 1)
base_network = create_base_network(input_shape)

input_a = Input(shape=input_shape, name='input_a')
input_b = Input(shape=input_shape, name='input_b')

# Both inputs share the same base network (same weights)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute distance
distance = layers.Lambda(euclidean_distance, name='distance')([processed_a, processed_b])

# Create final model
siamese_model = Model([input_a, input_b], distance, name='siamese_network')

# Compile model
siamese_model.compile(
    loss=contrastive_loss,
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

siamese_model.summary()

# ========================================
# 4. TRAIN MODEL
# ========================================
print("\n=== TRAINING MODEL ===")

# Create datasets
print("Loading training data...")
train_dataset = create_dataset(train_df, IMG_DIR, BATCH_SIZE, shuffle=True)
print("Loading validation data...")
val_dataset = create_dataset(val_df, IMG_DIR, BATCH_SIZE, shuffle=False)

# Callbacks
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    'siamese_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

early_stop_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# Train
history = siamese_model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    verbose=1
)

# ========================================
# 5. PLOT TRAINING HISTORY
# ========================================
print("\n=== PLOTTING RESULTS ===")

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

# ========================================
# 6. TEST MODEL
# ========================================
print("\n=== TESTING MODEL ===")

# Load best model
siamese_model.load_weights('siamese_best.h5')

# Test on some validation pairs
test_samples = val_df.sample(10)

for idx, row in test_samples.iterrows():
    img_a = load_image(os.path.join(IMG_DIR, row['img_a']))
    img_b = load_image(os.path.join(IMG_DIR, row['img_b']))
    
    # Predict
    distance = siamese_model.predict([
        np.expand_dims(img_a, 0),
        np.expand_dims(img_b, 0)
    ], verbose=0)[0][0]
    
    # Convert distance to score (0-100)
    score = max(0, (1 - distance) * 100)
    
    label_text = "SIMILAR" if row['label'] == 1 else "DIFFERENT"
    print(f"{row['img_a']} vs {row['img_b']}: Distance={distance:.4f}, Score={score:.1f}/100, Label={label_text}")

# ========================================
# 7. SAVE FINAL MODEL
# ========================================
print("\n=== SAVING MODEL ===")

# Save full model
siamese_model.save('siamese_model_full.h5')
print("✅ Saved: siamese_model_full.h5")

# Save base network only (for inference)
base_network.save('siamese_base_network.h5')
print("✅ Saved: siamese_base_network.h5")

# Save to TensorFlow.js format (for web deployment)
# !pip install tensorflowjs
# !tensorflowjs_converter --input_format=keras siamese_model_full.h5 tfjs_model

print("\n=== ✅ TRAINING COMPLETE ===")
print("Download these files:")
print("  1. siamese_model_full.h5 (full model)")
print("  2. siamese_base_network.h5 (base network only)")
print("  3. training_history.png (training plots)")
