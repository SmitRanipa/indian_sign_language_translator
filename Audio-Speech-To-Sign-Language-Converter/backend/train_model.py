import numpy as np
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "isl_model.pkl")

def train():
    print("Loading data...")
    try:
        X = np.load(os.path.join(DATA_DIR, "X.npy"))
        y = np.load(os.path.join(DATA_DIR, "y.npy"))
    except FileNotFoundError:
        print("Data files not found. Run load_data.py first.")
        return

    print(f"Data Loaded: X={X.shape}, y={y.shape}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train
    print("Training RandomForest Classifier...")
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
