"""
sample_data_gen.py
==================
Generates a mock FER-2013 CSV dataset and sample test images so the entire project
(DataLoader, Training, Evaluation, and Inference) can be tested immediately.
Works with zero external dependencies (pure standard library) with enhanced features if cv2/numpy/pandas are installed.
"""

import csv
import os
import random

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pandas as pd
except ImportError:
    pd = None

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def create_synthetic_pixels(emotion_id: int, size: int = 48) -> list:
    """
    Creates a 48x48 list of pixel values (0-255) representing a face with emotion patterns.
    """
    if np is not None and cv2 is not None:
        img = np.ones((size, size), dtype=np.uint8) * 130
        center = (size // 2, size // 2)
        radius = size // 2 - 4
        cv2.circle(img, center, radius, 220, -1)
        cv2.circle(img, center, radius, 40, 1)

        left_eye = (int(size * 0.33), int(size * 0.38))
        right_eye = (int(size * 0.67), int(size * 0.38))
        cv2.circle(img, left_eye, max(2, size // 16), 20, -1)
        cv2.circle(img, right_eye, max(2, size // 16), 20, -1)

        if emotion_id == 0:  # Angry
            cv2.line(img, (int(size * 0.27), int(size * 0.27)), (int(size * 0.40), int(size * 0.33)), 30, 2)
            cv2.line(img, (int(size * 0.73), int(size * 0.27)), (int(size * 0.60), int(size * 0.33)), 30, 2)
            cv2.ellipse(img, (size // 2, int(size * 0.79)), (int(size * 0.21), int(size * 0.12)), 0, 180, 360, 30, 2)
        elif emotion_id == 1:  # Disgust
            cv2.line(img, (int(size * 0.29), int(size * 0.31)), (int(size * 0.42), int(size * 0.31)), 30, 2)
            cv2.line(img, (int(size * 0.58), int(size * 0.31)), (int(size * 0.71), int(size * 0.31)), 30, 2)
            cv2.line(img, (int(size * 0.37), int(size * 0.73)), (int(size * 0.62), int(size * 0.67)), 30, 2)
        elif emotion_id == 2:  # Fear
            cv2.circle(img, left_eye, max(3, size // 10), 20, 1)
            cv2.circle(img, right_eye, max(3, size // 10), 20, 1)
            cv2.ellipse(img, (size // 2, int(size * 0.71)), (int(size * 0.12), int(size * 0.17)), 0, 0, 360, 30, -1)
        elif emotion_id == 3:  # Happy
            cv2.ellipse(img, (size // 2, int(size * 0.58)), (int(size * 0.25), int(size * 0.21)), 0, 0, 180, 30, 2)
        elif emotion_id == 4:  # Sad
            cv2.ellipse(img, (size // 2, int(size * 0.79)), (int(size * 0.21), int(size * 0.17)), 0, 180, 360, 30, 2)
        elif emotion_id == 5:  # Surprise
            cv2.line(img, (int(size * 0.27), int(size * 0.23)), (int(size * 0.40), int(size * 0.23)), 30, 2)
            cv2.line(img, (int(size * 0.60), int(size * 0.23)), (int(size * 0.73), int(size * 0.23)), 30, 2)
            cv2.circle(img, (size // 2, int(size * 0.71)), max(3, size // 8), 30, -1)
        else:  # Neutral
            cv2.line(img, (int(size * 0.33), int(size * 0.71)), (int(size * 0.67), int(size * 0.71)), 30, 2)

        noise = np.random.normal(0, 8, (size, size)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img.flatten().tolist()
    else:
        # Pure Python fallback
        pixels = []
        for r in range(size):
            for c in range(size):
                val = 130
                # Eyes
                if (16 <= r <= 20 and 12 <= c <= 18) or (16 <= r <= 20 and 30 <= c <= 36):
                    val = 20
                # Mouth
                elif 32 <= r <= 36 and 16 <= c <= 32:
                    val = 20 + emotion_id * 20
                # Add random noise
                val = max(0, min(255, val + random.randint(-10, 10)))
                pixels.append(val)
        return pixels


def generate_mock_fer2013(
    output_path: str = "data/fer2013.csv",
    samples_per_class_train: int = 40,
    samples_per_class_val: int = 10,
    samples_per_class_test: int = 10,
):
    """Generates synthetic FER-2013 CSV format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    rows = []

    splits = [
        ("Training", samples_per_class_train),
        ("PublicTest", samples_per_class_val),
        ("PrivateTest", samples_per_class_test),
    ]

    for usage, count in splits:
        for emotion_id in range(len(EMOTION_LABELS)):
            for _ in range(count):
                pixels = create_synthetic_pixels(emotion_id, size=48)
                pixel_str = " ".join(str(p) for p in pixels)
                rows.append({
                    "emotion": emotion_id,
                    "pixels": pixel_str,
                    "Usage": usage,
                })

    random.seed(42)
    random.shuffle(rows)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["emotion", "pixels", "Usage"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[*] Generated synthetic dataset with {len(rows)} samples at: {output_path}")


def generate_sample_images(output_dir: str = "data/sample_faces"):
    """Generates standalone sample PNG images if cv2/numpy or PIL are available."""
    os.makedirs(output_dir, exist_ok=True)
    if cv2 is not None and np is not None:
        for emotion_id, emotion_name in EMOTION_LABELS.items():
            pixels = create_synthetic_pixels(emotion_id, size=160)
            img = np.array(pixels, dtype=np.uint8).reshape(160, 160)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            file_path = os.path.join(output_dir, f"{emotion_name.lower()}_sample.png")
            cv2.imwrite(file_path, img_bgr)
        print(f"[*] Generated standalone sample face images in: {output_dir}")
    else:
        print("[*] Skipped standalone image PNG generation (install opencv-python or PIL to generate PNGs).")


if __name__ == "__main__":
    generate_mock_fer2013("data/fer2013.csv")
    generate_sample_images("data/sample_faces")
