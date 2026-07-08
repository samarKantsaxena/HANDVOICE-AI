import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ---------- HARDWARE OPTIMIZATION ----------
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ---------- CONFIG ----------
IMG_SIZE = (128, 128)          # MobileNet expects at least 128x128
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 26

# Path to your original dataset (the one with A/, B/, ... folders)
TRAIN_DIR = 'asl_alphabet_train/asl_alphabet_train'
TEST_DIR = 'asl_alphabet_test/asl_alphabet_test'

# ---------- DATA GENERATORS (WITH REAL-WORLD AUGMENTATION) ----------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,          # slight rotation (hands can be tilted)
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# ---------- BUILD MODEL USING MOBILENETV2 (FROZEN) ----------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze all convolutional layers

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------- CALLBACKS ----------
checkpoint = ModelCheckpoint('mobilenet_asl_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ---------- TRAIN ----------
print("\n🚀 Training MobileNetV2-based model on your ASL dataset...\n")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ---------- SAVE FINAL MODEL ----------
model.save('mobilenet_asl_final.keras')
print("\n✅ Model saved as 'mobilenet_asl_final.keras'")

# ---------- OPTIONAL: TEST ON YOUR TEST FOLDER ----------
if os.path.exists(TEST_DIR):
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )
    test_loss, test_acc = model.evaluate(test_generator, verbose=0)
    print(f"\n📊 Test Accuracy on your test set: {test_acc:.4f}")