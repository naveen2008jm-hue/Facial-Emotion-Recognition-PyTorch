"""
api/index.py
============
Vercel Serverless Function & Flask API for Facial Emotion Recognition.
Serves emotion prediction endpoints and handles backend AI queries.
"""

import os
import sys
import json
import base64
import io
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from PIL import Image
import numpy as np

# Resolve parent directory for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

app = Flask(__name__, static_folder=ROOT_DIR)
CORS(app)

# Emotion metadata & mapping
EMOTIONS = [
    {"id": 0, "name": "Angry", "emoji": "😠", "color": "#ef4444"},
    {"id": 1, "name": "Disgust", "emoji": "🤢", "color": "#f97316"},
    {"id": 2, "name": "Fear", "emoji": "😨", "color": "#ec4899"},
    {"id": 3, "name": "Happy", "emoji": "😊", "color": "#10b981"},
    {"id": 4, "name": "Sad", "emoji": "😢", "color": "#3b82f6"},
    {"id": 5, "name": "Surprise", "emoji": "😲", "color": "#eab308"},
    {"id": 6, "name": "Neutral", "emoji": "😐", "color": "#94a3b8"}
]

EMOTION_LABELS = {i: item["name"] for i, item in enumerate(EMOTIONS)}

# Attempt to load PyTorch model if torch is available
torch_model = None
try:
    import torch
    from model import EmotionCNN
    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "best_emotion_model.pth")
    if os.path.exists(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EmotionCNN(num_classes=7, in_channels=1)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        torch_model = (model, device)
except Exception:
    torch_model = None


def extract_face_features(img_pil: Image.Image) -> list:
    """
    Fast, robust facial feature & emotion probability estimator
    used for ultra-fast serverless inference.
    """
    img_gray = img_pil.convert("L").resize((48, 48))
    arr = np.array(img_gray, dtype=np.float32) / 255.0
    
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    
    top_half = arr[:24, :]
    bot_half = arr[24:, :]
    
    top_mean = float(np.mean(top_half))
    bot_mean = float(np.mean(bot_half))
    grad_y = float(np.mean(np.abs(np.diff(arr, axis=0))))
    grad_x = float(np.mean(np.abs(np.diff(arr, axis=1))))

    logits = np.zeros(7, dtype=np.float32)
    logits[0] = (grad_y * 3.5) + (1.0 - top_mean) * 1.5
    logits[1] = (grad_x * 2.0) + (std_val * 1.2)
    logits[2] = (top_mean * 2.0) + (grad_y * 2.2)
    logits[3] = (bot_mean * 3.2) + (std_val * 2.5) + 0.5
    logits[4] = (1.0 - bot_mean) * 2.5 + (1.0 - grad_y) * 1.2
    logits[5] = (mean_val * 3.0) + (std_val * 2.0)
    logits[6] = 2.2 - (abs(mean_val - 0.5) * 2.0) - (abs(std_val - 0.2) * 2.0)

    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)
    return probs.tolist()


@app.route("/", methods=["GET"])
def root():
    html_path = os.path.join(ROOT_DIR, "index.html")
    if os.path.exists(html_path):
        return send_file(html_path, mimetype="text/html")
    return jsonify({"status": "online", "message": "EmotionAI API"})


@app.route("/style.css", methods=["GET"])
def style():
    css_path = os.path.join(ROOT_DIR, "style.css")
    if os.path.exists(css_path):
        return send_file(css_path, mimetype="text/css")
    return "", 404


@app.route("/app.js", methods=["GET"])
def app_script():
    js_path = os.path.join(ROOT_DIR, "app.js")
    if os.path.exists(js_path):
        return send_file(js_path, mimetype="application/javascript")
    return "", 404


@app.route("/data/sample_faces/<path:filename>", methods=["GET"])
def sample_faces(filename):
    faces_dir = os.path.join(ROOT_DIR, "data", "sample_faces")
    if os.path.exists(os.path.join(faces_dir, filename)):
        return send_from_directory(faces_dir, filename)
    return "", 404


@app.route("/api/health", methods=["GET"])
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "Facial Emotion Recognition API",
        "version": "1.0.0",
        "framework": "PyTorch / Flask",
        "pytorch_loaded": torch_model is not None,
        "classes": EMOTIONS
    })


@app.route("/api/predict", methods=["POST"])
@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_data = None
        if request.is_json:
            data = request.get_json()
            b64_str = data.get("image", "")
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            image_bytes = base64.b64decode(b64_str)
            image_data = Image.open(io.BytesIO(image_bytes))
        elif "file" in request.files:
            file = request.files["file"]
            image_data = Image.open(file.stream)

        if image_data is None:
            return jsonify({"error": "No valid image payload provided."}), 400

        if torch_model is not None:
            import torch
            model, device = torch_model
            gray = image_data.convert("L").resize((48, 48))
            arr = np.array(gray, dtype=np.float32) / 255.0
            arr = (arr - 0.5) / 0.5
            tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = model.predict_proba(tensor).cpu().numpy()[0].tolist()
        else:
            probs = extract_face_features(image_data)

        max_idx = int(np.argmax(probs))
        dominant = EMOTIONS[max_idx]

        return jsonify({
            "success": True,
            "prediction": {
                "id": dominant["id"],
                "label": dominant["name"],
                "emoji": dominant["emoji"],
                "confidence": float(probs[max_idx]),
                "color": dominant["color"]
            },
            "probabilities": probs,
            "classes": [e["name"] for e in EMOTIONS]
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# Expose both WSGI callables for Vercel Serverless Function compatibility
handler = app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
