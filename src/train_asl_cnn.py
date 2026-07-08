"""
ULTRA-OPTIMIZED ASL ALPHABET RECOGNITION
For: Intel i3-7020U, 8GB RAM, no GPU
Training time: ~6-10 mins per epoch (total ~1-1.5 hours)
"""

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ========== 1. HARDWARE OPTIMIZATIONS ==========
# Limit CPU threads to prevent overheating (adjust if still hot -> set to 1)
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ========== 2. CONFIGURATION ==========
IMG_HEIGHT, IMG_WIDTH = 64, 64       # Small size = 4x faster than 128x128
BATCH_SIZE = 16                       # Low batch to save RAM
EPOCHS = 20                           # Early stopping will cut earlier
NUM_CLASSES = 26

# IMPORTANT: adjust these paths to match YOUR folder structure
# Typical structure:
#   asl_alphabet_train/
#       asl_alphabet_train/   <-- inner folder
#           A/, B/, C/, ...
TRAIN_DIR = 'asl_alphabet_train/asl_alphabet_train'
TEST_DIR = 'asl_alphabet_test/asl_alphabet_test'   # optional

# ========== 3. DATA LOADING ==========
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,           # Slight rotation (cheap augmentation)
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"\n✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {validation_generator.samples}")
print(f"✅ Class mapping: {train_generator.class_indices}\n")

# ========== 4. EXTREMELY LIGHTWEIGHT CNN ==========
model = models.Sequential([
    # Block 1: 16 filters
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2,2),
    
    # Block 2: 32 filters
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Block 3: 64 filters
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    # Classifier head
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ========== 5. CALLBACKS ==========
# Save the best model after every epoch (no progress loss)
checkpoint = ModelCheckpoint(
    'asl_best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Stop training if validation loss doesn't improve for 4 epochs
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True,
    verbose=1
)

# Reduce learning rate when plateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

# ========== 6. TRAINING ==========
print("\n🚀 Starting training... (this may take 1-2 hours total)\n")

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ========== 7. FINAL SAVE ==========
model.save('asl_final_model.keras')
print("\n✅ Final model saved as 'asl_final_model.keras'")

# ========== 8. QUICK TEST ON TEST FOLDER (optional) ==========
if os.path.exists(TEST_DIR):
    from sklearn.metrics import classification_report
    
    def load_test_images(test_dir):
        images, labels = [], []
        class_names = [chr(ord('A')+i) for i in range(26)]
        for file in sorted(os.listdir(test_dir)):
            if file.endswith(('.jpg', '.png')):
                # Extract letter from first character (e.g., "A_test.jpg" -> 'A')
                letter = file[0].upper()
                if letter in class_names:
                    img = cv2.imread(os.path.join(test_dir, file))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH)) / 255.0
                    images.append(img)
                    labels.append(class_names.index(letter))
        return np.array(images), np.array(labels)
    
    test_imgs, test_labels = load_test_images(TEST_DIR)
    if len(test_imgs) > 0:
        preds = np.argmax(model.predict(test_imgs, verbose=0), axis=1)
        print("\n📊 Test Set Performance (26 letters):")
        target_names = [chr(ord('A')+i) for i in range(26)]
        print(classification_report(test_labels, preds, target_names=target_names, zero_division=0))
    else:
        print("\n⚠️ No test images found in", TEST_DIR)
else:
    print(f"\n⚠️ Test directory '{TEST_DIR}' not found. Skipping test evaluation.")

print("\n🏁 Training complete! You can now run webcam detection with 'asl_final_model.keras'")