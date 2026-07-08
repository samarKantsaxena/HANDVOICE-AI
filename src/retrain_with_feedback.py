import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ---------- CONFIG ----------
ORIGINAL_MODEL = 'asl_final_model.keras'
FEEDBACK_DIR = 'collected_feedback'
NEW_MODEL = 'asl_final_model_finetuned.keras'
IMG_SIZE = (64, 64)
BATCH_SIZE = 16
EPOCHS = 10

# ---------- LOAD ORIGINAL MODEL ----------
model = load_model(ORIGINAL_MODEL)
print("Loaded original model.")

# ---------- DATA GENERATOR FOR FEEDBACK DATA ----------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    FEEDBACK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    FEEDBACK_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

if train_gen.samples == 0:
    print("❌ No feedback data found. Please run webcam interactive first.")
    exit()

print(f"✅ Loaded {train_gen.samples} training samples from feedback.")

# ---------- FREEZE BASE LAYERS (optional) ----------
# If you want to only train the top layers, uncomment:
# for layer in model.layers[:-4]:
#     layer.trainable = False

# ---------- COMPILE & FINE-TUNE ----------
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.5, patience=2)
]

print("\n🔄 Fine-tuning on new data...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ---------- SAVE NEW MODEL ----------
model.save(NEW_MODEL)
print(f"\n✅ Fine-tuned model saved as '{NEW_MODEL}'")
print("You can now use this model in webcam_asl_interactive.py (change MODEL_PATH).")