import cv2
import os
import string
import numpy as np
import mediapipe as mp
import time

DATASET_PATH = "../dataset"    # Folder to save landmarks
SAMPLES_PER_LABEL = 20         # Number of samples per gesture
CAMERA_INDEX = 0               # Default camera

# Create dataset folder if it doesn't exist
os.makedirs(DATASET_PATH, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Initialize camera
cap = cv2.VideoCapture(CAMERA_INDEX)

for label in string.ascii_uppercase:  # You can change to ['A'] to capture only A
    label_path = os.path.join(DATASET_PATH, label)
    os.makedirs(label_path, exist_ok=True)

    print(f"\n✋ Prepare to record gesture for: {label}")
    print("Starting in 20 seconds...")
    for i in range(30, 0, -1):
        print(i, end=' ', flush=True)
        time.sleep(1)
    print("\nRecording now!")

    count = 0
    while count < SAMPLES_PER_LABEL:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Camera frame not received. Exiting...")
            break

        # Flip and convert frame
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]

            landmarks = []
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)
            np.save(os.path.join(label_path, f"{count}.npy"), landmarks)
            count += 1

        # Show progress on screen
        cv2.putText(
            frame,
            f"{label}: {count}/{SAMPLES_PER_LABEL}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Landmark Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC to exit
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"✅ Done {label}")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("\n🎉 Dataset collection complete!")
