import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import string

model = load_model("../models/isl_model.h5")
labels = list(string.ascii_uppercase)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks).reshape(1, -1)
        prediction = model.predict(landmarks, verbose=0)
        letter = labels[np.argmax(prediction)]

        cv2.putText(
            frame,
            f"Prediction: {letter}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 0),
            3
        )

    cv2.imshow("ISL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
