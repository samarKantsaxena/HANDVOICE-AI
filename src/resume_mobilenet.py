import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ---------- SAME CONFIGURATION AS BEFORE ----------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 15          # total epochs you want
INITIAL_EPOCH = 5    # you stopped after 5 epochs (adjust if you stopped later)

TRAIN_DIR = 'asl_alphabet_train/asl_alphabet_train'
TEST_DIR = 'asl_alphabet_test/asl_alphabet_test'

# ---------- LOAD THE SAVED MODEL ----------
model = load_model('mobilenet_asl_best.keras')
print("Loaded model from checkpoint.")

# ---------- REBUILD DATA GENERATORS (same as before) ----------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
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

# ---------- COMPILE (if needed) ----------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ---------- CALLBACKS ----------
checkpoint = ModelCheckpoint('mobilenet_asl_best.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# ---------- RESUME TRAINING ----------
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE,
    initial_epoch=INITIAL_EPOCH,   # start from here
    epochs=EPOCHS,                 # train until this total epoch number
    callbacks=[checkpoint, early_stop, reduce_lr],
    verbose=1
)

# ---------- SAVE FINAL ----------
model.save('mobilenet_asl_final.keras')
print("Training resumed and completed.")