import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('asl_final_model_finetuned.keras')
class_names = [chr(ord('A') + i) for i in range(26)]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Testing fine‑tuned model. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess: resize to 64x64, normalize
    img = cv2.resize(frame, (64, 64))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb / 255.0
    input_batch = np.expand_dims(img_norm, axis=0)
    
    # Predict
    preds = model.predict(input_batch, verbose=0)[0]
    top_idx = np.argmax(preds)
    letter = class_names[top_idx]
    confidence = preds[top_idx]
    
    # Show result
    color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
    cv2.putText(frame, f"{letter} ({confidence:.2%})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    
    cv2.imshow('Fine‑Tuned ASL Model', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()