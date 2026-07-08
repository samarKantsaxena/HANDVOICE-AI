import cv2
import numpy as np
import os
import time
from tensorflow.keras.models import load_model

# ---------- CONFIGURATION ----------
MODEL_PATH = 'asl_final_model_finetuned.keras'
COLLECT_DIR = 'collected_feedback'
IMG_SIZE = (64, 64)

# ---------- LOAD MODEL ----------
model = load_model(MODEL_PATH)
class_names = [chr(ord('A') + i) for i in range(26)]

# ---------- CREATE FEEDBACK FOLDERS ----------
if not os.path.exists(COLLECT_DIR):
    os.makedirs(COLLECT_DIR)
for letter in class_names:
    os.makedirs(os.path.join(COLLECT_DIR, letter), exist_ok=True)

# ---------- HELPER: DRAW TOP 3 PREDICTIONS ----------
def draw_top3_predictions(frame, predictions):
    h, w = frame.shape[:2]
    top3_idx = np.argsort(predictions)[-3:][::-1]
    top3_conf = predictions[top3_idx]
    top3_labels = [class_names[i] for i in top3_idx]

    # Semi-transparent panel on the right side
    panel_w = 250
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    y_start = 60
    for i, (label, conf) in enumerate(zip(top3_labels, top3_conf)):
        text = f"{i+1}. {label}: {conf:.2%}"
        color = (0, 255, 0) if i == 0 else (200, 200, 200)
        cv2.putText(frame, text, (w - panel_w + 10, y_start + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame, top3_labels

# ---------- MANUAL INPUT FUNCTION ----------
def manual_letter_input():
    print("\n✏️  Manual entry mode (type letter A-Z then press Enter):")
    while True:
        user_input = input(">>> ").strip().upper()
        if len(user_input) == 1 and user_input in class_names:
            return user_input
        else:
            print("Invalid input. Please enter a single letter A-Z.")

# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🎥 ASL Interactive Learning v2 - with manual entry")
print("--------------------------------------------------")
print("Top 3 predictions on the right panel.")
print("Keys:")
print("  1,2,3  → confirm that prediction")
print("  4      → manually enter correct letter (opens terminal input)")
print("  s      → skip this frame (no save)")
print("  q      → quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    input_batch = np.expand_dims(img_norm, axis=0)

    # Predict
    preds = model.predict(input_batch, verbose=0)[0]
    display_frame, top3_labels = draw_top3_predictions(frame.copy(), preds)

    # Instructions at bottom
    cv2.putText(display_frame, "1,2,3: confirm | 4: manual | s: skip | q: quit",
                (10, display_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1)

    cv2.imshow('ASL Interactive Learning v2', display_frame)
    key = cv2.waitKey(1) & 0xFF

    # --- Handle confirmation keys 1,2,3 ---
    if key in [ord('1'), ord('2'), ord('3')]:
        idx = int(chr(key)) - 1
        chosen_letter = top3_labels[idx]
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(COLLECT_DIR, chosen_letter, f"manual_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Saved as '{chosen_letter}' -> {save_path}")
        # Flash feedback
        cv2.putText(display_frame, f"Saved: {chosen_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        cv2.imshow('ASL Interactive Learning v2', display_frame)
        cv2.waitKey(500)

    # --- Manual entry key 4 ---
    elif key == ord('4'):
        # Pause the webcam display briefly (still show frame)
        cv2.putText(display_frame, "Manual mode - check terminal", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.imshow('ASL Interactive Learning v2', display_frame)
        cv2.waitKey(100)
        # Get correct letter from terminal
        correct_letter = manual_letter_input()
        timestamp = int(time.time() * 1000)
        save_path = os.path.join(COLLECT_DIR, correct_letter, f"manual_{timestamp}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"✅ Manually saved as '{correct_letter}' -> {save_path}")
        cv2.putText(display_frame, f"Manually saved: {correct_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        cv2.imshow('ASL Interactive Learning v2', display_frame)
        cv2.waitKey(500)

    elif key == ord('s'):
        print("⏭️ Skipped")
        cv2.putText(display_frame, "Skipped", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
        cv2.imshow('ASL Interactive Learning v2', display_frame)
        cv2.waitKey(300)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📁 All collected data saved in '{COLLECT_DIR}'")