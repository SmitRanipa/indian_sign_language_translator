from flask import Flask, jsonify, request, Response
from flask_cors import CORS
import cv2
import os
import json
from landmark_predictor import HandSignPredictor

app = Flask(__name__)
CORS(app) # Enable CORS

predictor = HandSignPredictor()
cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return """
    <html>
        <head>
            <title>ISL Backend Debug</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; background: #f0f0f0; margin: 0; padding: 20px; }
                .container { max-width: 800px; margin: auto; background: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
                img { border: 2px solid #333; border-radius: 5px; margin-top: 10px; }
                h1 { color: #333; }
                .status { color: green; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ISL Backend Status: <span class="status">RUNNING</span></h1>
                <p>If you see the video below, the backend is working!</p>
                <img src="/video_feed" width="640" height="480" />
                <p>Endpoint: <code>/video_feed</code></p>
                
                <h3>Test Text-to-Gesture</h3>
                <input type="text" id="textInput" placeholder="Enter output text" style="padding: 10px; width: 200px;">
                <button onclick="testGesture()" style="padding: 10px; background: #007bff; color: white; border: none; cursor: pointer;">Test</button>
                <div id="gestureResult" style="margin-top: 20px; text-align: left; background: #eee; padding: 10px; border-radius: 5px;"></div>
            </div>
            <script>
                async function testGesture() {
                    const text = document.getElementById('textInput').value;
                    const res = await fetch('/text-to-gesture', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    const data = await res.json();
                    document.getElementById('gestureResult').innerText = JSON.stringify(data, null, 2);
                }
            </script>
        </body>
    </html>
    """

# Check if webcam opened
if not cap.isOpened():
    print("Warning: Could not open webcam.")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Predict
        label, annotated_frame = predictor.predict(frame)
        
        # Overlay label
        if label:
            cv2.putText(annotated_frame, f"Prediction: {label}", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
        # Encode
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['GET'])
def predict_snapshot():
    """Returns a single JSON prediction (useful for polling if stream not used)"""
    success, frame = cap.read()
    if success:
        label, _ = predictor.predict(frame)
        return jsonify({'prediction': label})
    return jsonify({'error': 'Camera not available'}), 503

@app.route('/text-to-gesture', methods=['POST'])
def text_to_gesture():
    """
    Input: {"text": "A"}
    Output: {"images": ["/assets/A.jpg"]}
    """
    data = request.json
    text = data.get('text', '').upper()
    
    # Simple mapping logic
    # Assuming images are stored in a static folder or frontend assets
    # Since this is backend, we return paths (frontend should serve them or backend static)
    # Let's assume backend serves them from static/
    
    result_images = []
    for char in text:
        if 'A' <= char <= 'Z':
            result_images.append(f"/static/signs/{char}.jpg")
        elif char == ' ':
             result_images.append("/static/signs/space.jpg")
             
    return jsonify({'images': result_images})

if __name__ == "__main__":
    # Ensure static folder exists for text-to-gesture
    os.makedirs('static/signs', exist_ok=True)
    app.run(debug=True, port=5000, host='0.0.0.0')
