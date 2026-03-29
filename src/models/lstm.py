"""
Bidirectional LSTM for ECG classification.

Uses a multi-layer bidirectional LSTM followed by an attention-weighted
aggregation and fully connected classifier.
"""

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for multi-label ECG classification.

    Architecture:
        Input projection → BiLSTM layers → Temporal attention → FC → logits

    Args:
        input_channels: Number of input channels (leads).
        num_classes: Number of output classes.
        hidden_size: LSTM hidden dimension (per direction).
        num_layers: Number of LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_dim = hidden_size * 2  # bidirectional

        # Temporal attention
        self.attention = nn.Sequential(
            nn.Linear(lstm_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_out_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor ``(batch, channels, seq_len)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        # Reshape: (batch, channels, seq_len) → (batch, seq_len, channels)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)

        # Attention-weighted aggregation
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = (lstm_out * attn_weights).sum(dim=1)  # (batch, hidden*2)

        context = self.dropout(context)
        return self.fc(context)
