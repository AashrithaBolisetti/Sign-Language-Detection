import cv2
import mediapipe as mp
import numpy as np
import joblib
import time
import pyttsx3
import string

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model/landmark_svm_model.pkl"
CONF_THRESHOLD = 0.75
LETTER_HOLD_TIME = 1.0     # seconds
SPACE_GAP_TIME = 1.5       # seconds (no hand = space)

# -----------------------------
# Load model
# -----------------------------
model = joblib.load(MODEL_PATH)
LABELS = list(string.ascii_uppercase)

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
# Text-to-Speech
# -----------------------------
engine = pyttsx3.init()

# -----------------------------
# State variables
# -----------------------------
current_letter = ""
last_letter = ""
letter_start_time = 0
last_hand_time = time.time()

word = ""
sentence = ""

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)
print("📷 Webcam started")
print("Controls: q=quit | c=clear | s=speak")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hand_detected = False

    if result.multi_hand_landmarks:
        hand_detected = True
        last_hand_time = time.time()

        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Extract landmarks
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)

        # Predict
        probs = model.predict_proba(landmarks)[0]
        idx = np.argmax(probs)
        confidence = probs[idx]

        if confidence > CONF_THRESHOLD:
            current_letter = LABELS[idx]

            if current_letter != last_letter:
                last_letter = current_letter
                letter_start_time = time.time()

            elif time.time() - letter_start_time > LETTER_HOLD_TIME:
                word += current_letter
                last_letter = ""
                letter_start_time = time.time()
                print("Added:", current_letter)

            cv2.putText(
                frame,
                f"{current_letter} ({confidence:.2f})",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3
            )

    # Space detection
    if not hand_detected and time.time() - last_hand_time > SPACE_GAP_TIME:
        if word != "":
            sentence += word + " "
            print("Word added:", word)
            word = ""
            last_hand_time = time.time()

    # Display sentence
    cv2.putText(
        frame,
        "Sentence: " + sentence + word,
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 0),
        2
    )

    cv2.imshow("Sign Language Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""
        word = ""
        print("Cleared")
    elif key == ord('s'):
        if sentence.strip():
            print("Speaking:", sentence)
            engine.say(sentence)
            engine.runAndWait()

cap.release()
cv2.destroyAllWindows()