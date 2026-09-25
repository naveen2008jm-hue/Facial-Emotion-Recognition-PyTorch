"""
train.py
========
Training script for Facial Emotion Recognition (FER-2013) in PyTorch.

Features:
- AdamW optimizer with weight decay regularization
- Weighted CrossEntropyLoss to address class imbalance
- Learning rate scheduler (ReduceLROnPlateau)
- Model checkpointing for the best validation accuracy
- Early stopping mechanism
- Training & validation loss / accuracy curve plotting
"""

import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loader import EMOTION_LABELS, get_data_loaders
from model import EmotionCNN, count_parameters


def set_seed(seed: int = 42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Executes one training epoch.
    Returns: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluates model on validation or test DataLoader.
    Returns: (average_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="  Val  ", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_metrics(history: Dict[str, List[float]], output_path: str):
    """
    Plots training & validation loss and accuracy curves.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", linewidth=2, linestyle="--")
    plt.title("Cross-Entropy Loss vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Loss", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, [acc * 100 for acc in history["train_acc"]], label="Train Acc", color="#2ca02c", linewidth=2)
    plt.plot(epochs, [acc * 100 for acc in history["val_acc"]], label="Val Acc", color="#d62728", linewidth=2, linestyle="--")
    plt.title("Classification Accuracy (%) vs. Epochs", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Accuracy (%)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[*] Training curves saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition CNN on FER-2013")
    parser.add_argument("--csv_path", type=str, default="data/fer2013.csv", help="Path to fer2013.csv dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--use_class_weights", action="store_true", default=True, help="Use inverse class frequency weights in loss")
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false", help="Disable class weighting")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save model weights")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save training plots and logs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'cuda', 'cpu', or 'auto'")
    args = parser.parse_args()

    set_seed(args.seed)

    # Device selection
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[*] Running on device: {device}")

    # Ensure output directories exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    train_loader, val_loader, test_loader, class_weights = get_data_loaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Initialize model
    model = EmotionCNN(num_classes=len(EMOTION_LABELS), in_channels=1).to(device)
    total_p, train_p = count_parameters(model)
    print(f"[*] EmotionCNN initialized with {total_p:,} total parameters ({train_p:,} trainable).")

    # Loss function with class weights
    if args.use_class_weights:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("[*] Using Weighted Cross-Entropy Loss.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("[*] Using Standard Cross-Entropy Loss.")

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    print("\n" + "=" * 70)
    print(f"{'Epoch':^7} | {'Train Loss':^11} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Acc':^10} | {'Time':^7}")
    print("=" * 70)

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - epoch_start
        print(
            f"{epoch:^7d} | {train_loss:^11.4f} | {train_acc * 100:^9.2f}% | {val_loss:^10.4f} | {val_acc * 100:^9.2f}% | {elapsed:^6.1f}s",
            end="",
        )

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0

            best_model_path = os.path.join(args.checkpoint_dir, "best_emotion_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "emotion_labels": EMOTION_LABELS,
                },
                best_model_path,
            )
            print(f" -> [BEST SAVED ({val_acc * 100:.2f}%)]")
        else:
            patience_counter += 1
            print("")

        # Save latest checkpoint
        latest_model_path = os.path.join(args.checkpoint_dir, "latest_emotion_model.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            },
            latest_model_path,
        )

        # Early Stopping check
        if patience_counter >= args.patience:
            print(f"\n[!] Early stopping triggered at epoch {epoch} (no validation improvement for {args.patience} epochs).")
            break

    total_time = time.time() - start_time
    print("=" * 70)
    print(f"[*] Training finished in {total_time / 60:.2f} minutes.")
    print(f"[*] Best Validation Accuracy: {best_val_acc * 100:.2f}% at Epoch {best_epoch}")

    # Save metrics JSON & plot
    history_json_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_json_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"[*] Training history saved to: {history_json_path}")

    plot_metrics(history, os.path.join(args.output_dir, "training_curves.png"))


if __name__ == "__main__":
    main()
