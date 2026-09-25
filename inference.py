"""
inference.py
============
Real-time webcam demo and single-image inference for Facial Emotion Recognition.

Capabilities:
1. Single Image Mode: Predicts emotion on an input image and displays probabilities.
2. Real-Time Webcam Mode: Uses OpenCV Haar Cascade for face detection and overlays
   bounding box, predicted emotion, and probability bars in real-time.
"""

import argparse
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from data_loader import EMOTION_LABELS
from model import EmotionCNN

# Emotion color palette for OpenCV overlay (BGR format)
EMOTION_COLORS = {
    0: (0, 0, 255),       # Angry: Red
    1: (0, 140, 255),     # Disgust: Orange
    2: (255, 0, 180),     # Fear: Magenta
    3: (0, 255, 0),       # Happy: Green
    4: (255, 100, 0),     # Sad: Deep Sky Blue
    5: (0, 255, 255),     # Surprise: Yellow
    6: (200, 200, 200),   # Neutral: Light Gray
}


def load_model(checkpoint_path: str, device: torch.device) -> EmotionCNN:
    """Loads trained EmotionCNN model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Please train the model first.")

    model = EmotionCNN(num_classes=len(EMOTION_LABELS), in_channels=1)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"[*] Loaded model checkpoint from: {checkpoint_path}")
    return model


def preprocess_face(face_gray: np.ndarray) -> torch.Tensor:
    """
    Preprocesses a 48x48 cropped grayscale face image for the PyTorch model.
    """
    # Resize to exact 48x48 if needed
    if face_gray.shape != (48, 48):
        face_gray = cv2.resize(face_gray, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize to [-1, 1] matching training transforms
    tensor = torch.from_numpy(face_gray).float().unsqueeze(0).unsqueeze(0)  # (1, 1, 48, 48)
    tensor = tensor / 255.0
    tensor = (tensor - 0.5) / 0.5
    return tensor


def predict_face_emotion(model: EmotionCNN, face_gray: np.ndarray, device: torch.device) -> Tuple[str, float, np.ndarray]:
    """
    Predicts the emotion of a cropped face image.
    Returns: (predicted_emotion_name, confidence_score, all_probabilities)
    """
    tensor = preprocess_face(face_gray).to(device)
    with torch.no_grad():
        probs = model.predict_proba(tensor).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_label = EMOTION_LABELS[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_label, confidence, probs


def draw_hud(frame: np.ndarray, x: int, y: int, w: int, h: int, label: str, conf: float, probs: np.ndarray) -> np.ndarray:
    """
    Draws modern bounding box and emotion probability bars on the frame.
    """
    color = EMOTION_COLORS.get(list(EMOTION_LABELS.values()).index(label), (0, 255, 0))

    # Draw rounded-corner style bounding box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    # Header label background
    header_text = f"{label} ({conf * 100:.1f}%)"
    (text_w, text_h), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y - text_h - 12), (x + text_w + 10, y), color, -1)
    cv2.putText(frame, header_text, (x + 5, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw emotion probabilities bar HUD on the side
    hud_x = x + w + 10
    hud_y = y
    hud_w = 140
    hud_h = 16

    # If HUD goes off frame to the right, place it on the left
    if hud_x + hud_w > frame.shape[1]:
        hud_x = max(10, x - hud_w - 10)

    for i, (cls_idx, cls_name) in enumerate(EMOTION_LABELS.items()):
        bar_y = hud_y + i * (hud_h + 6)
        if bar_y + hud_h > frame.shape[0]:
            break

        prob = probs[cls_idx]
        bar_fill_w = int(hud_w * prob)

        # Background bar
        cv2.rectangle(frame, (hud_x, bar_y), (hud_x + hud_w, bar_y + hud_h), (50, 50, 50), -1)
        # Filled bar
        bar_color = EMOTION_COLORS.get(cls_idx, (0, 255, 0))
        cv2.rectangle(frame, (hud_x, bar_y), (hud_x + bar_fill_w, bar_y + hud_h), bar_color, -1)

        # Text
        txt = f"{cls_name[:4]}: {prob * 100:.0f}%"
        cv2.putText(frame, txt, (hud_x + 4, bar_y + hud_h - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def run_image_inference(image_path: str, model: EmotionCNN, device: torch.device, output_path: str = None):
    """Performs emotion recognition on a single static image."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at '{image_path}'")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Face detector
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("[!] No face detected by OpenCV Haar Cascade. Using whole image as face.")
        faces = [(0, 0, img.shape[1], img.shape[0])]

    print(f"[*] Detected {len(faces)} face(s) in image.")
    for idx, (x, y, w, h) in enumerate(faces):
        face_roi = gray[y : y + h, x : x + w]
        label, conf, probs = predict_face_emotion(model, face_roi, device)
        print(f"\nFace #{idx + 1}:")
        print(f"  -> Predicted Emotion: {label} ({conf * 100:.2f}%)")
        print("  -> Probability Distribution:")
        for c_idx, c_name in EMOTION_LABELS.items():
            print(f"     * {c_name:<9}: {probs[c_idx] * 100:5.2f}%")

        draw_hud(img, x, y, w, h, label, conf, probs)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"\n[*] Annotated output image saved to: {output_path}")


def run_webcam_demo(model: EmotionCNN, device: torch.device, cam_id: int = 0):
    """Runs real-time webcam facial emotion recognition."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        print(f"[!] Error: Could not open webcam with ID {cam_id}.")
        return

    print("\n[*] Starting Webcam Emotion Recognition...")
    print("[*] Press 'q' or 'ESC' in the video window to exit.")

    fps_history = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Failed to grab frame from webcam.")
            break

        start_time = cv2.getTickCount()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            face_roi = gray[y : y + h, x : x + w]
            label, conf, probs = predict_face_emotion(model, face_roi, device)
            draw_hud(frame, x, y, w, h, label, conf, probs)

        # FPS calculation
        end_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (end_time - start_time)
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)

        # Status header banner
        cv2.putText(
            frame,
            f"Facial Emotion Recognition (PyTorch) | FPS: {avg_fps:.1f}",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Facial Emotion Recognition - Press Q to Exit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # ESC or q
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[*] Webcam stream stopped.")


def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Recognition Inference & Real-Time Demo")
    parser.add_argument("--mode", type=str, choices=["webcam", "image"], default="webcam", help="Inference mode: 'webcam' or 'image'")
    parser.add_argument("--image", type=str, default=None, help="Path to input image (required if mode is 'image')")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_emotion_model.pth", help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="outputs/prediction_result.png", help="Path to save annotated image")
    parser.add_argument("--cam_id", type=int, default=0, help="Webcam device ID (default: 0)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(args.checkpoint, device)

    if args.mode == "image":
        if not args.image:
            raise ValueError("Please provide --image <path_to_image> when mode is 'image'.")
        run_image_inference(args.image, model, device, args.output)
    else:
        run_webcam_demo(model, device, args.cam_id)


if __name__ == "__main__":
    main()
