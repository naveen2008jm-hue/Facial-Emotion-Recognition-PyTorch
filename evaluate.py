"""
evaluate.py
===========
Evaluation script to assess the trained Facial Emotion Recognition model on the test dataset.

Outputs:
- Overall test accuracy and top-2 accuracy
- Per-class classification metrics (Precision, Recall, F1-Score)
- Confusion Matrix visualization saved to `outputs/confusion_matrix.png`
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import EMOTION_LABELS, get_data_loaders
from model import EmotionCNN


def evaluate_test_set(model: nn.Module, test_loader, device: torch.device):
    """
    Runs model on test dataset and collects true labels, predicted labels, and probabilities.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating Test Set"):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_targets.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Top-1 accuracy
    top1_acc = np.mean(all_preds == all_targets)

    # Top-2 accuracy
    top2_preds = np.argsort(all_probs, axis=1)[:, -2:]
    top2_correct = [target in top2_preds[i] for i, target in enumerate(all_targets)]
    top2_acc = np.mean(top2_correct)

    return all_targets, all_preds, all_probs, top1_acc, top2_acc


def plot_confusion_matrix(targets: np.ndarray, preds: np.ndarray, class_names: list, output_path: str):
    """
    Plots both raw counts and normalized confusion matrices side-by-side.
    """
    cm = confusion_matrix(targets, preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)  # handle any zero division

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar=True,
    )
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Predicted Emotion", fontsize=11)
    axes[0].set_ylabel("True Emotion", fontsize=11)

    # Normalized heatmap
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("Normalized Confusion Matrix (Recall per Class)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Predicted Emotion", fontsize=11)
    axes[1].set_ylabel("True Emotion", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Confusion matrix plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Facial Emotion Recognition model")
    parser.add_argument("--csv_path", type=str, default="data/fer2013.csv", help="Path to fer2013.csv dataset")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_emotion_model.pth", help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save evaluation artifacts")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Running evaluation on device: {device}")

    # Check checkpoint existence
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint file '{args.checkpoint}' not found. Please train the model first with 'python train.py'."
        )

    # Load DataLoaders
    _, _, test_loader, _ = get_data_loaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
    )

    # Initialize model & load weights
    model = EmotionCNN(num_classes=len(EMOTION_LABELS), in_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[*] Loaded checkpoint from '{args.checkpoint}' (saved from Epoch {checkpoint.get('epoch', 'N/A')})")

    # Evaluate on test set
    class_names = [EMOTION_LABELS[i] for i in range(len(EMOTION_LABELS))]
    targets, preds, probs, top1_acc, top2_acc = evaluate_test_set(model, test_loader, device)

    # Print Summary Metrics
    print("\n" + "=" * 60)
    print("                    EVALUATION RESULTS")
    print("=" * 60)
    print(f" Top-1 Test Accuracy : {top1_acc * 100:.2f}%")
    print(f" Top-2 Test Accuracy : {top2_acc * 100:.2f}%")
    print("=" * 60)

    # Classification Report
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    print("\nDetailed Per-Class Classification Report:\n")
    print(report)

    # Save report to text file
    report_path = os.path.join(args.output_dir, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(f"Top-1 Accuracy: {top1_acc * 100:.2f}%\n")
        f.write(f"Top-2 Accuracy: {top2_acc * 100:.2f}%\n\n")
        f.write(report)
    print(f"[*] Classification report written to: {report_path}")

    # Plot & save Confusion Matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(targets, preds, class_names, cm_path)


if __name__ == "__main__":
    main()
