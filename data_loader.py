"""
data_loader.py
==============
Data loading, preprocessing, and augmentation pipeline for the FER-2013 dataset.

FER-2013 details:
- 48x48 pixel grayscale face images
- 7 emotion classes:
    0: Angry
    1: Disgust
    2: Fear
    3: Happy
    4: Sad
    5: Surprise
    6: Neutral
- CSV format columns: 'emotion', 'pixels' (space-separated 2304 integers), 'Usage' ('Training', 'PublicTest', 'PrivateTest')
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Class ID to Emotion name mapping
EMOTION_LABELS: Dict[int, str] = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def get_transforms(is_train: bool = True) -> transforms.Compose:
    """
    Returns image transformation pipelines.

    Training transforms include data augmentation (horizontal flip, slight rotation,
    translation) to prevent overfitting on the relatively small 48x48 faces.
    """
    if is_train:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])


class FER2013Dataset(Dataset):
    """
    PyTorch Dataset wrapper for FER-2013 images parsed from CSV.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, transform: Optional[transforms.Compose] = None):
        """
        Args:
            images: np.ndarray of shape (N, 48, 48), uint8.
            labels: np.ndarray of shape (N,), int64.
            transform: torchvision transforms to apply.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)
        else:
            # Fallback if no transform provided
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.0
            img = (img - 0.5) / 0.5

        return img, torch.tensor(label, dtype=torch.long)


def parse_pixels(pixels_series: pd.Series) -> np.ndarray:
    """
    Parses a pandas Series of space-separated pixel strings into a numpy array (N, 48, 48).
    """
    num_samples = len(pixels_series)
    images = np.empty((num_samples, 48, 48), dtype=np.uint8)
    for i, pixel_str in enumerate(pixels_series):
        arr = np.fromstring(pixel_str, dtype=np.uint8, sep=" ")
        if arr.shape[0] == 48 * 48:
            images[i] = arr.reshape(48, 48)
        else:
            # Handle possible corrupted entries
            images[i] = np.zeros((48, 48), dtype=np.uint8)
    return images


def compute_class_weights(labels: np.ndarray, num_classes: int = 7) -> torch.Tensor:
    """
    Computes inverse class frequencies to weight the CrossEntropyLoss,
    addressing severe class imbalances in FER-2013 (e.g., Disgust ~500 samples vs Happy ~9000).

    weight_c = total_samples / (num_classes * count_c)
    """
    counts = np.bincount(labels, minlength=num_classes)
    total_samples = len(labels)
    # Avoid division by zero with clip
    counts = np.clip(counts, a_min=1, a_max=None)
    weights = total_samples / (num_classes * counts.astype(np.float32))
    # Normalize weights so mean is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def get_data_loaders(
    csv_path: str,
    batch_size: int = 64,
    num_workers: int = 0,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.Tensor]:
    """
    Loads FER-2013 from CSV and returns train, val, and test DataLoaders along with class weights.

    Args:
        csv_path: Path to fer2013.csv
        batch_size: Mini-batch size for DataLoader
        num_workers: Number of DataLoader subprocesses (0 for Windows compatibility)
        val_ratio: Fallback validation split ratio if 'Usage' column missing
        test_ratio: Fallback test split ratio if 'Usage' column missing
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader, test_loader, class_weights_tensor)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset CSV not found at '{csv_path}'. "
            "Please download FER-2013 or run 'python sample_data_gen.py' to generate a mock dataset."
        )

    print(f"[*] Reading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Check required columns
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("CSV must contain 'emotion' and 'pixels' columns.")

    # Split dataset based on 'Usage' column if present
    if "Usage" in df.columns:
        print("[*] Splitting using 'Usage' column (Training / PublicTest / PrivateTest)...")
        train_df = df[df["Usage"] == "Training"]
        val_df = df[df["Usage"] == "PublicTest"]
        test_df = df[df["Usage"] == "PrivateTest"]

        # If PrivateTest is empty, split PublicTest between val and test
        if len(test_df) == 0 and len(val_df) > 0:
            val_df_split = val_df.sample(frac=0.5, random_state=seed)
            test_df = val_df.drop(val_df_split.index)
            val_df = val_df_split
    else:
        print(f"[*] 'Usage' column missing. Splitting randomly (train: {1.0 - val_ratio - test_ratio:.0%}, val: {val_ratio:.0%}, test: {test_ratio:.0%})...")
        df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n_total = len(df_shuffled)
        n_val = int(n_total * val_ratio)
        n_test = int(n_total * test_ratio)
        n_train = n_total - n_val - n_test

        train_df = df_shuffled.iloc[:n_train]
        val_df = df_shuffled.iloc[n_train : n_train + n_val]
        test_df = df_shuffled.iloc[n_train + n_val :]

    print(f"[*] Dataset partition sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Parse pixels and extract labels
    train_images = parse_pixels(train_df["pixels"])
    train_labels = train_df["emotion"].to_numpy(dtype=np.int64)

    val_images = parse_pixels(val_df["pixels"])
    val_labels = val_df["emotion"].to_numpy(dtype=np.int64)

    test_images = parse_pixels(test_df["pixels"])
    test_labels = test_df["emotion"].to_numpy(dtype=np.int64)

    # Compute class weights from training set to handle class imbalance
    class_weights = compute_class_weights(train_labels, num_classes=len(EMOTION_LABELS))
    print("[*] Computed class weights for CrossEntropyLoss:")
    for idx, name in EMOTION_LABELS.items():
        count = np.sum(train_labels == idx)
        print(f"    - Class {idx} ({name:<9}): {count:>5} samples (weight: {class_weights[idx]:.3f})")

    # Construct Datasets
    train_dataset = FER2013Dataset(train_images, train_labels, transform=get_transforms(is_train=True))
    val_dataset = FER2013Dataset(val_images, val_labels, transform=get_transforms(is_train=False))
    test_dataset = FER2013Dataset(test_images, test_labels, transform=get_transforms(is_train=False))

    # Construct DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, class_weights


if __name__ == "__main__":
    # Self-test snippet
    print("EMOTION_LABELS:", EMOTION_LABELS)
