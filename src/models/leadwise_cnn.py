"""
Lead-wise 1D CNN for ECG classification.

Processes each ECG lead independently through a shared CNN backbone,
then fuses lead-level features for classification.
"""

import torch
import torch.nn as nn


class LeadwiseCNN(nn.Module):
    """
    Lead-wise 1D CNN that processes each lead independently.

    Architecture:
        Per-lead: Conv1d blocks × n_blocks → AdaptiveAvgPool
        Fusion: Concatenate all lead features → FC layers → logits

    Args:
        input_channels: Number of leads (default 12).
        num_classes: Number of output classes.
        base_filters: Filters in the first conv layer per lead.
        n_blocks: Number of conv blocks per lead.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 32,
        n_blocks: int = 3,
        kernel_size: int = 7,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.n_leads = input_channels

        # Shared backbone applied to each lead
        layers = []
        in_ch = 1
        for i in range(n_blocks):
            out_ch = base_filters * (2 ** i)
            layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2, 2),
            ))
            in_ch = out_ch

        self.lead_backbone = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Total features = out_ch * n_leads
        feat_dim = in_ch * self.n_leads
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape ``(batch, n_leads, seq_len)``.

        Returns:
            Logits of shape ``(batch, num_classes)``.
        """
        batch_size = x.size(0)
        lead_features = []

        for i in range(self.n_leads):
            lead = x[:, i:i+1, :]  # (batch, 1, seq_len)
            feat = self.lead_backbone(lead)
            feat = self.pool(feat).squeeze(-1)  # (batch, out_ch)
            lead_features.append(feat)

        # Concatenate all lead features
        fused = torch.cat(lead_features, dim=1)  # (batch, out_ch * n_leads)
        return self.classifier(fused)
