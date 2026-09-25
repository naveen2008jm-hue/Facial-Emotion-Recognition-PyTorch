"""
model.py
========
Custom Convolutional Neural Network (EmotionCNN / FERNet) for Facial Emotion Recognition.

Architecture Overview:
----------------------
- Input: (B, 1, 48, 48) Grayscale Face
- 4 Feature Extraction Blocks (Conv -> BN -> ELU/ReLU -> Conv -> BN -> ELU/ReLU -> MaxPool -> Dropout)
- Fully Connected Classification Head (Dense -> BN -> Dropout -> Dense -> Dropout -> Logits)
- Output: (B, 7) Logits corresponding to the 7 emotion classes
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Standard Conv Block: [Conv2d -> BatchNorm2d -> ELU] x 2 -> MaxPool2d -> Dropout2d
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.25):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ELU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EmotionCNN(nn.Module):
    """
    Custom 4-block CNN designed for 48x48 Facial Emotion Recognition.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 1):
        super(EmotionCNN, self).__init__()
        self.num_classes = num_classes

        # Feature Extractor Blocks:
        # Input: (1, 48, 48)
        # Block 1 -> (32, 24, 24)
        self.block1 = ConvBlock(in_channels, 32, dropout_p=0.20)
        # Block 2 -> (64, 12, 12)
        self.block2 = ConvBlock(32, 64, dropout_p=0.25)
        # Block 3 -> (128, 6, 6)
        self.block3 = ConvBlock(64, 128, dropout_p=0.25)
        # Block 4 -> (256, 3, 3)
        self.block4 = ConvBlock(128, 256, dropout_p=0.30)

        # Classification Head:
        # Flatten: 256 * 3 * 3 = 2304 features
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ELU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, 1, 48, 48)
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        x = torch.flatten(x, 1)  # (B, 2304)
        logits = self.classifier(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes softmax emotion probabilities.
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    Returns (total_params, trainable_params).
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    # Test model instantiation and dummy forward pass
    net = EmotionCNN(num_classes=7)
    dummy_input = torch.randn(4, 1, 48, 48)
    out = net(dummy_input)
    proba = net.predict_proba(dummy_input)
    total_p, train_p = count_parameters(net)

    print(f"Model initialized successfully!")
    print(f"Total Parameters: {total_p:,} (Trainable: {train_p:,})")
    print(f"Input Shape:  {dummy_input.shape}")
    print(f"Output Logits Shape: {out.shape}")
    print(f"Output Proba Shape:  {proba.shape}")
