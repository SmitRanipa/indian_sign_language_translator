import os
import numpy as np
import pickle
from tqdm import tqdm
from utils import normalize_landmarks

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data")
X_PATH = os.path.join(PROCESSED_DATA_DIR, "X.npy")
Y_PATH = os.path.join(PROCESSED_DATA_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "model", "label_map.pkl")

def load_and_process_data():
    """
    Loads .npy files from dataset/{label}/*.npy
    Normalizes them.
    Flattens them.
    Saves X.npy and y.npy
    """
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset directory '{DATASET_DIR}' not found relative to {os.getcwd()}")
        return

    X = []
    y = []
    labels = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    label_map = {label: idx for idx, label in enumerate(labels)}
    
    print(f"Found {len(labels)} classes: {labels}")

    if not labels:
        print("No classes found in dataset directory.")
        return

    # Inspect first file to determine expected shape
    first_label = labels[0]
    label_path = os.path.join(DATASET_DIR, first_label)
    files = os.listdir(label_path)
    if not files:
        print(f"No files in {label_path}")
        return  
        
    first_file = files[0]
    sample_path = os.path.join(label_path, first_file)
    sample_data = np.load(sample_path)
    print(f"Sample data shape: {sample_data.shape}")
    
    # Handle already flattened data
    if len(sample_data.shape) == 1:
        is_flattened = True
        n_features = sample_data.shape[0]
        # Guess dimensions (assume 3D if divisible by 3, else 2D)
        if n_features % 3 == 0:
            dims = 3
        else:
            dims = 2
        n_landmarks = n_features // dims
        print(f"Detected flattened data. Reshaping to ({n_landmarks}, {dims}) for normalization.")
    else:
        is_flattened = False
        n_landmarks = sample_data.shape[0]
        dims = sample_data.shape[1]

    for label in tqdm(labels, desc="Loading Data"):
        label_dir = os.path.join(DATASET_DIR, label)
        class_idx = label_map[label]
        
        for file_name in os.listdir(label_dir):
            if file_name.endswith('.npy'):
                file_path = os.path.join(label_dir, file_name)
                try:
                    data = np.load(file_path)
                    
                    if is_flattened:
                        if data.shape != (n_features,):
                             continue
                        data = data.reshape(-1, dims)
                    else:
                        if data.shape != (n_landmarks, dims):
                            continue
                    
                    # Normalize landmarks (Critical step)
                    norm_data = normalize_landmarks(data)
                        
                    # Flatten
                    X.append(norm_data.flatten())
                    y.append(class_idx)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

    if not X:
        print("No valid data loaded.")
        return

    X = np.array(X)
    y = np.array(y)
    
    print(f"Processed Type: X shape: {X.shape}, y shape: {y.shape}")

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LABEL_MAP_PATH), exist_ok=True)
    
    np.save(X_PATH, X)
    np.save(Y_PATH, y)
    
    with open(LABEL_MAP_PATH, 'wb') as f:
        pickle.dump(labels, f) # Save list of labels for decoding
        
    print(f"Saved processed data to {PROCESSED_DATA_DIR}")
    print(f"Saved label map to {LABEL_MAP_PATH}")

if __name__ == "__main__":
    load_and_process_data()
