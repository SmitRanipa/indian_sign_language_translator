import numpy as np
import os
import pickle

def normalize_landmarks(landmarks):
    """
    Normalizes landmarks to be translation and scale invariant.
    Expected shape: (N, 2) or (N, 3)
    Output shape: Scaled (N, 2) or (N, 3) flattened or not? 
    Let's keep it structured here.
    """
    # Convert to numpy array if list
    landmarks = np.array(landmarks)
    
    # Center around the first landmark (usually wrist) or mean
    # Using wrist (index 0) as origin is common in hand tracking
    base = landmarks[0]
    centered = landmarks - base
    
    # Scale by maximum distance from origin to keep within [-1, 1] approx
    # This makes it scale invariant (hand distance from camera doesn't matter)
    max_dist = np.max(np.abs(centered))
    if max_dist > 0:
        normalized = centered / max_dist
    else:
        normalized = centered
        
    return normalized

def save_pickle(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
