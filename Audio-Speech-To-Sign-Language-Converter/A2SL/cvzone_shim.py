import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import os

class HandDetector:
    def __init__(self, staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5):
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for simplicity and compatibility
            num_hands=maxHands,
            min_hand_detection_confidence=detectionCon,
            min_hand_presence_confidence=minTrackCon
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def findHands(self, img, draw=True, flipType=True):
        if img is None or img.size == 0:
            return [], img
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        try:
            results = self.detector.detect(mp_image)
        except Exception as e:
            print(f"Shim detection error: {e}")
            return [], img

        
        allHands = []
        if results.hand_landmarks:
            for hand_landmarks, handedness in zip(results.hand_landmarks, results.handedness):
                myHand = {}
                mylmList = []
                xList = []
                yList = []
                h, w, c = img.shape
                for lm in hand_landmarks:
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)
                
                # Bounding Box
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), bbox[1] + (bbox[3] // 2)
                
                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)
                
                # Handedness
                label = handedness[0].category_name
                if flipType:
                    # In Mediapipe Tasks, handedness is relative to the camera
                    # cvzone's flipType logic for legacy MP was: 
                    # if handType.classification[0].label == "Right": myHand["type"] = "Left" else: myHand["type"] = "Right"
                    myHand["type"] = "Left" if label == "Right" else "Right"
                else:
                    myHand["type"] = label
                
                allHands.append(myHand)
                
                if draw:
                    # We can implement basic drawing if needed, but final_pred.py doesn't seem to rely on it (draw=False)
                    pass
        
        return allHands, img
    def close(self):
        if hasattr(self, 'detector'):
            self.detector.close()
