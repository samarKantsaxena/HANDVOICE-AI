import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# ---------- HARDWARE OPTIMIZATION ----------
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ---------- CONFIG ----------
ORIGINAL_MODEL = 'asl_final_model.keras'   # your existing model
TRAIN_CSV = 'sign_mnist_train.csv'
TEST_CSV = 'sign_mnist_test.csv'
NEW_MODEL = 'asl_final_model_finetuned.keras'
IMG_SIZE = (64, 64)          # <-- FIXED: match original model
BATCH_SIZE = 32
EPOCHS = 10

# ---------- 1. LOAD ORIGINAL MODEL ----------
print("Loading original model...")
model = models.load_model(ORIGINAL_MODEL)
print("Original model loaded.")
print("Expected input shape:", model.input_shape)   # should be (None, 64, 64, 3)

# Freeze all layers except the last 4 (top layers)
for layer in model.layers[:-4]:
    layer.trainable = False
for layer in model.layers[-4:]:
    layer.trainable = True
print("Model layers frozen (except last 4 layers).")

# ---------- 2. LOAD SIGN LANGUAGE MNIST DATA ----------
print("\nLoading Sign Language MNIST CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

y_train = train_df['label'].values
X_train = train_df.drop('label', axis=1).values
y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Reshape to 28x28 grayscale
X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')
X_train /= 255.0
X_test /= 255.0

# ---------- 3. PREPROCESS TO 64x64 RGB ----------
def preprocess_for_model(images, target_size=(64,64)):
    processed = []
    for img in images:
        img = img.squeeze()                     # (28,28)
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img_rgb = np.stack([img_resized]*3, axis=-1)  # (64,64,3)
        processed.append(img_rgb)
    return np.array(processed)

print("\nPreprocessing training images (resize to 64x64 & convert to RGB)...")
X_train_processed = preprocess_for_model(X_train, IMG_SIZE)
X_test_processed = preprocess_for_model(X_test, IMG_SIZE)
print(f"Processed shape: {X_train_processed.shape}")

# ---------- 4. MAP LABELS (MNIST 24 -> original 26) ----------
def map_mnist_to_26(mnist_label):
    if mnist_label < 9:
        return mnist_label          # A-I
    else:
        return mnist_label + 1      # K-Y (skip J at index 9)

y_train_mapped = np.array([map_mnist_to_26(l) for l in y_train])
y_test_mapped = np.array([map_mnist_to_26(l) for l in y_test])

y_train_cat = tf.keras.utils.to_categorical(y_train_mapped, num_classes=26)
y_test_cat = tf.keras.utils.to_categorical(y_test_mapped, num_classes=26)

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_processed, y_train_cat, test_size=0.2, random_state=42
)

print(f"Fine‑tuning data: {len(X_train_split)} train, {len(X_val_split)} val, {len(X_test_processed)} test")

# ---------- 5. COMPILE ----------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- 6. CALLBACKS ----------
checkpoint = ModelCheckpoint('finetuned_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

# ---------- 7. FINE-TUNE ----------
print("\n🚀 Fine‑tuning on Sign Language MNIST data...\n")
history = model.fit(
    X_train_split, y_train_split,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val_split, y_val_split),
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ---------- 8. EVALUATE ----------
test_loss, test_acc = model.evaluate(X_test_processed, y_test_cat, verbose=0)
print(f"\n✅ Test Accuracy on Sign Language MNIST (fine‑tuned): {test_acc:.4f}")

# ---------- 9. SAVE ----------
model.save(NEW_MODEL)
print(f"\n✅ Fine‑tuned model saved as '{NEW_MODEL}'")