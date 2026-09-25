# 🎭 Facial Emotion Recognition System (FER) in PyTorch

A deep learning mini-project for classifying human facial expressions into **7 basic emotions** using a custom Convolutional Neural Network (CNN) built with **PyTorch** and trained on the **FER-2013** dataset.

---

## 📌 Project Overview

| Property | Details |
|---|---|
| **Coursework Domain** | Artificial Intelligence / Deep Learning / Computer Vision |
| **Framework** | PyTorch 2.x |
| **Dataset** | FER-2013 (Facial Expression Recognition 2013) |
| **Input Shape** | 48 × 48 grayscale facial images (1 channel) |
| **Target Classes (7)** | `0: Angry`, `1: Disgust`, `2: Fear`, `3: Happy`, `4: Sad`, `5: Surprise`, `6: Neutral` |
| **Core Architecture** | Custom 4-Stage Deep CNN (`EmotionCNN`) with Batch Normalization & Dropout |
| **Optimization** | AdamW Optimizer + ReduceLROnPlateau Scheduler + Weighted CrossEntropyLoss |

---

## 📂 Project Structure

```
AI project/
├── data/                         # Dataset directory (fer2013.csv or mock data)
│   └── sample_faces/             # Generated test face images
├── checkpoints/                  # Saved PyTorch model checkpoints (.pth)
│   ├── best_emotion_model.pth
│   └── latest_emotion_model.pth
├── outputs/                      # Output graphs, evaluation reports, and predictions
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── training_history.json
├── data_loader.py                # CSV parser, Dataset class, Augmentation, & DataLoader factory
├── model.py                      # Custom CNN architecture (EmotionCNN) & parameter counter
├── train.py                      # Training loop with validation, weighting, and early stopping
├── evaluate.py                   # Test set evaluation, metrics calculation, and confusion matrix
├── inference.py                  # Live webcam demo & single image inference with OpenCV
├── sample_data_gen.py            # Quick synthetic data generator for instant pipeline testing
├── requirements.txt              # Required Python libraries
└── README.md                     # Project documentation & coursework viva notes
```

---

## 🚀 Quick Start Guide

### 1. Set Up Environment & Install Dependencies

Open PowerShell / Terminal inside this project folder:

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
# source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

### 2. Getting the FER-2013 Dataset

#### Option A: Kaggle Official FER-2013 (Full Dataset ~300MB)
1. Download `fer2013.csv` from [Kaggle FER-2013 Competition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) or [Kaggle Datasets](https://www.kaggle.com/datasets/msambare/fer2013).
2. Place the extracted `fer2013.csv` inside the `data/` folder:
   ```
   AI project/data/fer2013.csv
   ```

#### Option B: Instant Mock Dataset (For Immediate Testing & Verification)
If you do not have the full Kaggle dataset downloaded yet, generate a synthetic mini-dataset in seconds:
```bash
python sample_data_gen.py
```
This generates `data/fer2013.csv` with 420 synthetic faces and sample test images in `data/sample_faces/`.

---

### 3. Training the Model

Run the training pipeline:

```bash
# Standard training (default parameters: 35 epochs, batch size 64)
python train.py --csv_path data/fer2013.csv --epochs 35 --batch_size 64

# Training on GPU with custom learning rate
python train.py --csv_path data/fer2013.csv --epochs 40 --lr 0.001 --device cuda
```

**What happens during training:**
- Computes **inverse class frequency weights** to prevent the model from ignoring underrepresented emotions like *Disgust*.
- Applies **online data augmentation** (horizontal flips, random rotation, slight translation).
- Dynamically reduces learning rate when validation accuracy plateaus (`ReduceLROnPlateau`).
- Saves the best checkpoint to `checkpoints/best_emotion_model.pth`.
- Generates loss and accuracy curves at `outputs/training_curves.png`.

---

### 4. Evaluating the Model

Assess test performance, precision/recall/F1-score per emotion, and generate confusion matrices:

```bash
python evaluate.py --csv_path data/fer2013.csv --checkpoint checkpoints/best_emotion_model.pth
```

**Generated artifacts:**
- `outputs/confusion_matrix.png` (Raw counts + Normalized Recall heatmap)
- `outputs/classification_report.txt` (Per-class precision, recall, and F1-score)

---

### 5. Running Inference & Real-Time Demo

#### A. Real-Time Webcam Demo
Uses your webcam, detects faces in real-time with OpenCV Haar Cascades, classifies expressions, and renders confidence bars:

```bash
python inference.py --mode webcam --checkpoint checkpoints/best_emotion_model.pth
```
*(Press `q` or `ESC` in the video window to exit).*

#### B. Single Image Prediction
Predict emotion on any photo or test face:

```bash
python inference.py --mode image --image data/sample_faces/happy_sample.png --output outputs/prediction_result.png
```

---

## 🧠 Model Architecture Details

The `EmotionCNN` model is structured into 4 convolutional blocks followed by a dense classification head:

```
Input: (1, 48, 48) Grayscale
  │
  ├── Block 1: [Conv 3x3 (32) -> BN -> ELU] x 2 -> MaxPool 2x2 -> Dropout(0.20)  [Output: 32 x 24 x 24]
  │
  ├── Block 2: [Conv 3x3 (64) -> BN -> ELU] x 2 -> MaxPool 2x2 -> Dropout(0.25)  [Output: 64 x 12 x 12]
  │
  ├── Block 3: [Conv 3x3 (128) -> BN -> ELU] x 2 -> MaxPool 2x2 -> Dropout(0.25) [Output: 128 x 6 x 6]
  │
  ├── Block 4: [Conv 3x3 (256) -> BN -> ELU] x 2 -> MaxPool 2x2 -> Dropout(0.30) [Output: 256 x 3 x 3]
  │
  ├── Flatten -> 256 * 3 * 3 = 2,304 features
  │
  ├── Dense Head:
  │     ├── Linear(2304 -> 256) -> BatchNorm1d -> ELU -> Dropout(0.5)
  │     ├── Linear(256 -> 128)  -> BatchNorm1d -> ELU -> Dropout(0.4)
  │     └── Linear(128 -> 7)    -> Class Logits
  ▼
Output: 7 Emotion Probabilities (Softmax)
```

---

## 🎓 Coursework Viva / Presentation Q&A

1. **Why use ELU instead of standard ReLU?**
   - Exponential Linear Units (ELU) have smooth negative saturation values, pushing mean activations closer to zero and speeding up convergence while preventing "dead neurons".
2. **Why is Class Weighting necessary in FER-2013?**
   - In FER-2013, the *Happy* class has ~9,000 samples, whereas *Disgust* has only ~500 samples (~18x difference). Standard Cross-Entropy would bias predictions heavily towards dominant classes. Class weighting penalizes mistakes on minority classes proportionally.
3. **What is the role of Batch Normalization?**
   - Normalizes intermediate layer activations, stabilizing learning dynamics, reducing internal covariate shift, and acting as a mild regularizer.
4. **How is Overfitting mitigated?**
   - Dropout layers (20% to 50%), Weight Decay ($L_2$ regularization in AdamW), and Real-time Data Augmentation (random horizontal flips, rotations).
