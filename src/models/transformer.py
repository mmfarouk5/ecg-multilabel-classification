"""
Transformer encoder for ECG classification.

Uses positional encoding and multi-head self-attention to classify
ECG signals. Returns attention weights for interpretability.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for sequence models.

    Args:
        d_model: Embedding dimension.
        max_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding; x shape (batch, seq_len, d_model)."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    Transformer encoder for multi-label ECG classification.

    Architecture:
        Input projection → Positional encoding → Transformer encoder layers
        → CLS token pooling → FC → logits

    Returns attention weights from the last encoder layer for
    interpretability.

    Args:
        input_channels: Number of input channels (leads).
        num_classes: Number of output classes.
        d_model: Transformer embedding dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer encoder layers.
        dim_feedforward: Feed-forward network dimension.
        dropout: Dropout rate.
        max_seq_len: Maximum input sequence length.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        **kwargs,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input channels to d_model
        self.input_proj = nn.Linear(input_channels, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)

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
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Classifier
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

        # Hook for capturing attention weights
        self._attention_weights: Optional[torch.Tensor] = None

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input ``(batch, channels, seq_len)``.
            return_attention: If True, also return attention weights.

        Returns:
            Logits ``(batch, num_classes)``, and optionally attention
            weights ``(batch, nhead, seq_len+1, seq_len+1)``.
        """
        # (batch, channels, seq_len) → (batch, seq_len, channels)
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_proj(x)  # (batch, seq_len, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, seq_len+1, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Capture attention from last layer
        if return_attention:
            # Register hook on the last encoder layer
            attn_weights = []

            def _hook(module, input, output):
                # Access self-attention inside the encoder layer
                pass

            # Feed through all layers except last
            for layer in self.transformer_encoder.layers[:-1]:
                x = layer(x)

            # Last layer: manually compute attention
            last_layer = self.transformer_encoder.layers[-1]
            # Use the self_attn module directly
            attn_out, weights = last_layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            # Continue through the rest of the layer
            x = last_layer.norm1(x + last_layer.dropout1(attn_out))
            ff_out = last_layer.linear2(
                last_layer.dropout(last_layer.activation(last_layer.linear1(x)))
            )
            x = last_layer.norm2(x + last_layer.dropout2(ff_out))

            # CLS token representation
            cls_repr = x[:, 0]
            logits = self.fc(cls_repr)
            return logits, weights
        else:
            x = self.transformer_encoder(x)
            cls_repr = x[:, 0]
            logits = self.fc(cls_repr)
            return logits
