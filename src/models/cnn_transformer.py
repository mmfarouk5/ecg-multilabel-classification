"""
CNN-Transformer hybrid model for ECG classification.

Uses 1D CNN blocks for local feature extraction and downsampling,
followed by a Transformer encoder for global temporal modeling.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class CNNTransformer(nn.Module):
    """
    CNN-Transformer hybrid for multi-label ECG classification.

    Architecture:
        Conv1d blocks → Positional encoding → Transformer encoder
        → CLS token pooling → FC → logits

    Args:
        input_channels: Number of input channels (leads).
        num_classes: Number of output classes.
        cnn_filters: CNN output filters.
        n_cnn_blocks: Number of CNN blocks.
        kernel_size: CNN kernel size.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_transformer_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        cnn_filters: int = 64,
        n_cnn_blocks: int = 3,
        kernel_size: int = 7,
        d_model: int = 128,
        nhead: int = 8,
        num_transformer_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model

        # CNN feature extractor
        cnn_layers = []
        in_ch = input_channels
        for i in range(n_cnn_blocks):
            out_ch = cnn_filters * (2 ** i)
            cnn_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, 2),
            ))
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # Project CNN output to d_model
        self.proj = nn.Linear(in_ch, d_model)

        # Positional encoding
        self.pos_encoder = nn.ModuleDict()
        max_len = 5000
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.pe_dropout = nn.Dropout(dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input ``(batch, channels, seq_len)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        batch_size = x.size(0)

        # CNN features: (batch, cnn_out_ch, reduced_len)
        x = self.cnn(x)

        # (batch, reduced_len, cnn_out_ch) → (batch, reduced_len, d_model)
        x = x.permute(0, 2, 1)
        x = self.proj(x)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Add positional encoding
        x = x + self.pe[:, :x.size(1)]
        x = self.pe_dropout(x)

        # Transformer
        x = self.transformer(x)

        # CLS token → classifier
        cls_repr = x[:, 0]
        return self.fc(cls_repr)
