import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from utils import normalize_landmarks

class HandSignPredictor:
    def __init__(self, model_path=None, label_map_path=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, 'model', 'isl_model.pkl')
        if label_map_path is None:
            label_map_path = os.path.join(base_dir, 'model', 'label_map.pkl')

        # Load Model
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(label_map_path, 'rb') as f:
                self.labels = pickle.load(f)
            self.model_loaded = True
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            self.labels = []
            
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def predict(self, frame):
        """
        Processes frame, detects hand, predicts sign.
        Returns: predicted_label, annotated_frame
        """
        if not self.model_loaded:
            return "Model Error", frame

        # Convert to RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        predicted_char = ""
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks
                lm_list = []
                for lm in hand_landmarks.landmark:
                    lm_list.append([lm.x, lm.y, lm.z]) 
                
                lm_array = np.array(lm_list) # Shape (21, 3)
                
                # Check for 20 vs 21 landmarks compatibility
                n_features = self.model.n_features_in_
                
                # Logic to slice if needed (assuming dataset has 20 landmarks)
                # If model expects 60 features (20*3), we take first 20 landmarks.
                if n_features == 60: # 20*3
                    lm_subset = lm_array[:20, :]
                elif n_features == 40: # 20*2 (x,y only)
                    lm_subset = lm_array[:20, :2]
                elif n_features == 63: # 21*3
                    lm_subset = lm_array
                elif n_features == 42: # 21*2
                    lm_subset = lm_array[:, :2]
                else:
                    lm_subset = lm_array # Hope for best

                # NORMALIZE (Critical step)
                norm_features = normalize_landmarks(lm_subset)
                
                # Flatten
                features = norm_features.flatten().reshape(1, -1)

                try:
                    prediction = self.model.predict(features)
                    class_idx = int(prediction[0])
                    if 0 <= class_idx < len(self.labels):
                        predicted_char = self.labels[class_idx]
                    else:
                        predicted_char = str(class_idx)
                except Exception as e:
                    print(f"Prediction error: {e}")

                # Draw
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
        return predicted_char, frame
