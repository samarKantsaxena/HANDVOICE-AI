import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ---------- CONFIG ----------
MODEL_PATH = 'mobilenet_asl_best.keras'  # or 'mobilenet_asl_final.keras'
IMG_SIZE = (128, 128)
CLASS_NAMES = [chr(ord('A') + i) for i in range(26)]

# ---------- LOAD MODEL ----------
model = load_model(MODEL_PATH)
print("✅ Model loaded.")

# ---------- WEB CAM ----------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ---------- STATS ----------
total = 0
correct = 0
accuracy = 0.0

print("\n🎥 ASL Webcam Test – Press keys:")
print("  1  → correct prediction")
print("  2  → wrong prediction")
print("  s  → skip this frame")
print("  q  → quit and show final accuracy")
print("------------------------------------------------")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    img = cv2.resize(frame, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    input_batch = np.expand_dims(img_norm, axis=0)

    # Predict
    preds = model.predict(input_batch, verbose=0)[0]
    top3_idx = np.argsort(preds)[-3:][::-1]
    top3_conf = preds[top3_idx]
    top3_labels = [CLASS_NAMES[i] for i in top3_idx]

    best_label = top3_labels[0]
    best_conf = top3_conf[0]

    # --- Display on frame ---
    # Show top prediction large
    color = (0, 255, 0) if best_conf > 0.7 else (0, 255, 255)
    cv2.putText(frame, f"{best_label} ({best_conf:.2%})", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)

    # Show top 3 on right side
    h, w = frame.shape[:2]
    panel_w = 250
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, (label, conf) in enumerate(zip(top3_labels, top3_conf)):
        text = f"{i+1}. {label}: {conf:.2%}"
        cv2.putText(frame, text, (w - panel_w + 15, 50 + i*40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # Show running accuracy
    if total > 0:
        acc_text = f"Accuracy: {accuracy:.1%} ({correct}/{total})"
        cv2.putText(frame, acc_text, (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('ASL Webcam Test – Press 1 (correct) or 2 (wrong)', frame)

    # --- Key handling ---
    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):
        # Correct
        total += 1
        correct += 1
        accuracy = correct / total
        print(f"✅ Correct: {best_label} ({best_conf:.2%})   Acc: {accuracy:.1%}")

    elif key == ord('2'):
        # Wrong
        total += 1
        accuracy = correct / total
        print(f"❌ Wrong:  {best_label} (should be something else)   Acc: {accuracy:.1%}")

    elif key == ord('s'):
        print("⏭️ Skipped")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ---------- FINAL REPORT ----------
print("\n🏁 Final Results:")
if total > 0:
    print(f"Total samples: {total}")
    print(f"Correct:       {correct}")
    print(f"Accuracy:      {accuracy:.1%}")
else:
    print("No feedback given.")