# ISL Translator Backend

## Setup

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Prepare Data**:
    Ensure your dataset is in `dataset/{label}/*.npy`.
    Run the data loading script:
    ```bash
    python load_data.py
    ```
    This will create `data/X.npy` and `data/y.npy`.

3.  **Train Model**:
    ```bash
    python train_model.py
    ```
    This will save the model to `model/isl_model.pkl`.

4.  **Run Application**:
    ```bash
    python app.py
    ```
    Server will start at `http://localhost:5000`.

## API Endpoints

-   `GET /video_feed`: Real-time video stream with prediction overlay.
-   `GET /predict`: Single frame prediction JSON.
-   `POST /text-to-gesture`: Convert text to gesture image paths.

## Folder Structure

-   `dataset/`: Your raw .npy files.
-   `data/`: Processed NumPy arrays (generated).
-   `model/`: Trained model and label map (generated).
-   `static/signs/`: Place ISL gesture images here for text-to-speech feature.
