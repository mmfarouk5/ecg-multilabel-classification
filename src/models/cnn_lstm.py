"""
CNN-LSTM hybrid model for ECG classification.

Uses 1D CNN blocks for local feature extraction followed by a 
bidirectional LSTM for temporal modeling.
"""

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid for multi-label ECG classification.

    Architecture:
        Conv1d blocks → BiLSTM → Last hidden state → FC → logits

    Args:
        input_channels: Number of input channels (leads).
        num_classes: Number of output classes.
        cnn_filters: Number of filters per CNN block.
        n_cnn_blocks: Number of CNN blocks.
        kernel_size: CNN kernel size.
        lstm_hidden: LSTM hidden dimension per direction.
        lstm_layers: Number of LSTM layers.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        cnn_filters: int = 64,
        n_cnn_blocks: int = 3,
        kernel_size: int = 7,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

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

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input ``(batch, channels, seq_len)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        # CNN feature extraction
        x = self.cnn(x)  # (batch, cnn_out_ch, reduced_seq_len)

        # Reshape for LSTM: (batch, seq_len, features)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden)
        forward_h = h_n[-2]  # last layer forward
        backward_h = h_n[-1]  # last layer backward
        hidden = torch.cat([forward_h, backward_h], dim=1)

        hidden = self.dropout(hidden)
        return self.fc(hidden)
