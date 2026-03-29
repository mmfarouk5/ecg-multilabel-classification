"""
1D ResNet for ECG classification.

Implements a residual network with 1D convolutions adapted for
time-series ECG signals, following the ResNet design pattern.
"""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """
    Basic residual block for 1D convolutions.

    Architecture: Conv → BN → ReLU → Conv → BN + skip → ReLU

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        kernel_size: Convolution kernel size.
        stride: Stride for downsampling.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection with optional projection
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(nn.Module):
    """
    1D ResNet for multi-label ECG classification.

    Architecture:
        Initial conv → 4 residual stages → AdaptiveAvgPool → FC

    Args:
        input_channels: Number of input channels (leads).
        num_classes: Number of output classes.
        base_filters: Base filter count.
        n_blocks_per_stage: Blocks per residual stage.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate before the FC layer.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        n_blocks_per_stage: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        # Initial convolution
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, base_filters, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # Residual stages
        self.stage1 = self._make_stage(base_filters, base_filters, n_blocks_per_stage, kernel_size, stride=1)
        self.stage2 = self._make_stage(base_filters, base_filters * 2, n_blocks_per_stage, kernel_size, stride=2)
        self.stage3 = self._make_stage(base_filters * 2, base_filters * 4, n_blocks_per_stage, kernel_size, stride=2)
        self.stage4 = self._make_stage(base_filters * 4, base_filters * 8, n_blocks_per_stage, kernel_size, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 8, num_classes)

    @staticmethod
    def _make_stage(
        in_ch: int, out_ch: int, n_blocks: int, kernel_size: int, stride: int
    ) -> nn.Sequential:
        """Build a residual stage with multiple blocks."""
        layers = [ResidualBlock1D(in_ch, out_ch, kernel_size, stride)]
        for _ in range(1, n_blocks):
            layers.append(ResidualBlock1D(out_ch, out_ch, kernel_size, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input ``(batch, channels, seq_len)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
