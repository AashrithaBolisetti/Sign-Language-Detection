import cv2
import mediapipe as mp
import numpy as np
import os
import string
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
DATA_DIR = "data/landmarks"
os.makedirs(DATA_DIR, exist_ok=True)

LABELS = list(string.ascii_uppercase)  # A-Z
SAMPLES_PER_LETTER = 200
CAPTURE_INTERVAL = 0.05  # seconds

# -----------------------------
# MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Ask which letter
# -----------------------------
letter = input("Enter letter to collect data for (A-Z): ").strip().upper()

if letter not in LABELS:
    print("❌ Invalid letter")
    exit()

print(f"\n📌 Collecting {SAMPLES_PER_LETTER} samples for letter: {letter}")
print("✋ Hold gesture steady")
print("⏳ Starting in 3 seconds...")
time.sleep(3)

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
data = []
last_capture_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_time = time.time()
        if current_time - last_capture_time >= CAPTURE_INTERVAL:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                data.append(landmarks)
                last_capture_time = current_time
                print(f"✅ {len(data)}/{SAMPLES_PER_LETTER}")

    cv2.putText(
        frame,
        f"Letter: {letter} | Samples: {len(data)}/{SAMPLES_PER_LETTER}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Collect Sign Language Data", frame)

    if len(data) >= SAMPLES_PER_LETTER:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -----------------------------
# Save landmarks
# -----------------------------
data = np.array(data, dtype=np.float32)
np.save(os.path.join(DATA_DIR, f"{letter}.npy"), data)

print(f"\n🎉 Saved {len(data)} samples to data/landmarks/{letter}.npy")