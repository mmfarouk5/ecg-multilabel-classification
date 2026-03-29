"""
1D Convolutional Neural Network for ECG classification.

A multi-block 1D CNN with batch normalization, ReLU activations,
and adaptive pooling for sequence classification.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    1D CNN for multi-label ECG classification.

    Architecture:
        Conv1d blocks (conv → BN → ReLU → MaxPool) × n_blocks
        → AdaptiveAvgPool1d → FC → output logits

    Args:
        input_channels: Number of input channels (leads), default 12.
        num_classes: Number of output classes.
        base_filters: Number of filters in the first conv layer.
        n_blocks: Number of convolutional blocks.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate before the final FC layer.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        blocks = []
        in_ch = input_channels
        for i in range(n_blocks):
            out_ch = base_filters * (2 ** i)
            blocks.append(self._make_block(in_ch, out_ch, kernel_size))
            in_ch = out_ch

        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Sequential:
        """Create a single Conv1d → BN → ReLU → MaxPool block."""
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape ``(batch, channels, seq_len)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
