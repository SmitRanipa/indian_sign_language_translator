# Indian Sign Language (ISL) Assistant (Final Merged Project)

This project is a comprehensive **Bi-Directional Communication System** designed to bridge the gap between hearing individuals and the deaf/hard-of-hearing community. It combines two powerful modules into a single web application:
1.  **Audio/Speech-to-Sign Language Converter:** Translates spoken or typed English into ISL animations.
2.  **Sign Language-to-Text/Speech Converter:** Recognizes hand gestures via webcam and translates them into spoken text.

---

## 🚀 Features

### Mode 1: Speech to ISL Animation (For Hearing Users)
-   **Voice Input:** Uses Web Speech API to capture spoken sentences.
-   **Text Processing (NLP):**
    -   Tokenizes sentences into keywords.
    -   Removes stop words (is, am, are) for ISL grammar compliance.
    -   Lemmatizes words to their base forms (e.g., "Running" -> "Run").
-   **Animation Playback:**
    -   Matches keywords to a database of pre-recorded ISL videos.
    -   Result is played as a seamless sequence of sign videos.
    -   **Finger Spelling Fallback:** Used for names or words not in the database.

### Mode 2: Gesture Recognition (For Deaf Users)
-   **Real-Time Detection:** Uses webcam input to track hand movements.
-   **Skeletal Tracking:** Utilizes **MediaPipe** to extract 21 hand landmarks, ensuring robustness against background clutter and lighting conditions.
-   **AI Classification:**
    -   A **CNN (Convolutional Neural Network)** classifies general hand shapes.
    -   **Geometric Logic:** Custom algorithms measure distances between fingers to distinguish similar signs (e.g., 'U' vs 'V').
-   **Text-to-Speech:** Converts recognized gestured sentences into audible speech.

---

## 🛠️ Technology Stack

-   **Backend:** Django (Python)
-   **Frontend:** HTML5, CSS3, JavaScript (Vanilla)
-   **Machine Learning & CV:**
    -   **MediaPipe:** For Hand Tracking & Landmark Extraction.
    -   **TensorFlow / Keras:** For gesture classification (CNN Model).
    -   **OpenCV (`cv2`):** For image processing.
-   **Natural Language Processing:**
    -   **NLTK:** For tokenization, lemmatization, and POS tagging.
-   **Browser APIs:**
    -   Web Speech API (Speech Recognition & Synthesis).

---

## ⚙️ Installation & Setup

### Prerequisites
-   Python 3.8 or higher installed.
-   A webcam (for gesture recognition).
-   Basic understanding of command line / terminal.

### Step-by-Step Guide

1.  **Navigate to the Project Directory**
    The main working project is located in the `Audio-Speech-To-Sign-Language-Converter` directory.
    ```bash
    cd Audio-Speech-To-Sign-Language-Converter/Audio-Speech-To-Sign-Language-Converter
    ```

2.  **Create a Virtual Environment (Optional but Recommended)**
    ```bash
    python -m venv venv
    
    # Windows
    venv\Scripts\activate
    
    # Mac/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Install all required Python libraries from `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

    *If you face issues with `dlib` or `opencv`, ensure you have CMake and C++ build tools installed.*

4.  **Download NLTK Data**
    The project uses NLTK for text processing. You might need to download specific data packages. Run python shell:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('omw-1.4')
    exit()
    ```

5.  **Run Migrations**
    Initialize the database.
    ```bash
    python manage.py migrate
    ```

6.  **Start the Server**
    ```bash
    python manage.py runserver
    ```

7.  **Access the Application**
    Open your browser (Chrome recommended for Web Speech API support) and go to:
    `http://127.0.0.1:8000/`

---

## 📖 How to Use

### 1. Translating Speech to Sign (Home Page)
1.  Click on the **Microphone Icon** and speak a sentence (e.g., "Hello world").
2.  Alternatively, type text in the input box.
3.  Click **"Translate"**.
4.  Watch the avatar perform the sign language translation.

### 2. Translating Sign to Text (Gesture Page)
1.  Navigate to the **"Gesture"** tab/link in the navigation bar.
2.  Allow camera permissions when prompted.
3.  Show your hand clearly to the camera with a neutral background.
4.  Performing a sign (A-Z) will display the letter on screen.
5.  Form words and click **"Speak"** to hear them read aloud.

---

## 🤝 Contribution
This project was developed for the **Final Year Hackathon**. It integrates computer vision and natural language processing to solve real-world accessibility challenges.

**Team Members:**
- Smit Ranipa
- Parth Zalariya
- Rutvik Bhagiya
- Hadiyal Devang
