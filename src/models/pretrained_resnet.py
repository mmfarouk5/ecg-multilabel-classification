"""
Pretrained 1D ResNet for ECG classification.

Provides a ResNet1D backbone with support for loading pretrained weights
from a file or URL. Useful for transfer learning — load backbone weights
pretrained on a large ECG dataset, then fine-tune the classifier head.

Supported pretrained sources:
- PTB-XL benchmark weights (xresnet1d)
- PhysioNet Challenge 2017 weights (hsd1503/resnet1d)
- Any custom .pt checkpoint (backbone-only or full model)
"""

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PretrainedResidualBlock(nn.Module):
    """
    Improved residual block (ResNet-D style).

    Uses BN-ReLU-Conv ordering (pre-activation) and optional
    average-pool downsampling for smoother gradient flow.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size.
        stride: Stride for downsampling.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.stride = stride

        self.bn1 = nn.BatchNorm1d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=padding
        )

        self.bn2 = nn.BatchNorm1d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding)

        # Skip with optional projection
        self.needs_proj = (stride != 1 or in_ch != out_ch)
        if self.needs_proj:
            self.skip_conv = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip_conv(x)
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.dropout(self.relu2(self.bn2(out))))
        # Handle potential size mismatch from conv rounding
        if out.shape[-1] != identity.shape[-1]:
            min_len = min(out.shape[-1], identity.shape[-1])
            out = out[..., :min_len]
            identity = identity[..., :min_len]
        return out + identity


class PretrainedResNet1D(nn.Module):
    """
    ResNet1D with pretrained backbone support for ECG classification.

    Architecture (xresnet1d style):
        Stem: 3 conv layers (7→3→3) with gradual channel expansion
        4 residual stages with increasing filters
        Global average pooling → Dropout → FC

    The model separates the backbone (stem + stages + pool) from the
    classifier head (FC). When loading pretrained weights, only the
    backbone is initialized, and the head is randomly initialized.

    Args:
        input_channels: Number of input channels (ECG leads).
        num_classes: Number of output classes.
        base_filters: Base filter count for the stem.
        layers: Number of residual blocks per stage.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
        pretrained_path: Path to pretrained backbone weights (.pt file).
        freeze_backbone: If True, freeze backbone weights during training.
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = 5,
        base_filters: int = 64,
        layers: Optional[list] = None,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()

        if layers is None:
            layers = [2, 2, 2, 2]  # ResNet-18 style

        bf = base_filters

        # ── Stem (xresnet-D style: 3 small convs instead of 1 large) ──
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, bf // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(bf // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bf // 2, bf // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(bf // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(bf // 2, bf, kernel_size=3, padding=1),
            nn.BatchNorm1d(bf),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        # ── Residual stages ──
        filter_sizes = [bf, bf * 2, bf * 4, bf * 8]
        self.stage1 = self._make_stage(
            bf, filter_sizes[0], layers[0], kernel_size, stride=1, dropout=dropout
        )
        self.stage2 = self._make_stage(
            filter_sizes[0], filter_sizes[1], layers[1], kernel_size, stride=2, dropout=dropout
        )
        self.stage3 = self._make_stage(
            filter_sizes[1], filter_sizes[2], layers[2], kernel_size, stride=2, dropout=dropout
        )
        self.stage4 = self._make_stage(
            filter_sizes[2], filter_sizes[3], layers[3], kernel_size, stride=2, dropout=dropout
        )

        # ── Backbone output ──
        self.backbone_dim = filter_sizes[3]
        self.pool = nn.AdaptiveAvgPool1d(1)

        # ── Classifier head ──
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.backbone_dim, num_classes),
        )

        # ── Load pretrained weights ──
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if freeze_backbone:
            self._freeze_backbone()

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        kernel_size: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        """Build a residual stage with n_blocks."""
        blocks = [
            PretrainedResidualBlock(in_ch, out_ch, kernel_size, stride, dropout)
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                PretrainedResidualBlock(out_ch, out_ch, kernel_size, 1, dropout)
            )
        return nn.Sequential(*blocks)

    def _load_pretrained(self, path: str) -> None:
        """
        Load pretrained weights from a checkpoint file.

        Handles three formats:
        1. Full model state_dict → loads matching keys
        2. Dict with 'model_state_dict' key → extracts and loads
        3. Dict with 'backbone' key → loads only backbone
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Pretrained weights not found: %s. Training from scratch.", path)
            return

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Extract state dict from various checkpoint formats
        if isinstance(checkpoint, dict):
            if "backbone" in checkpoint:
                state_dict = checkpoint["backbone"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint.state_dict() if hasattr(checkpoint, "state_dict") else checkpoint

        # Filter to only load backbone keys (exclude head/classifier)
        model_dict = self.state_dict()
        pretrained_dict = {}
        skipped = []

        for k, v in state_dict.items():
            # Remove common prefixes from other frameworks
            clean_key = k.replace("module.", "").replace("backbone.", "")

            if clean_key in model_dict and model_dict[clean_key].shape == v.shape:
                pretrained_dict[clean_key] = v
            else:
                skipped.append(k)

        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            logger.info(
                "Loaded %d/%d pretrained parameters from %s",
                len(pretrained_dict), len(model_dict), path,
            )
        else:
            logger.warning(
                "No matching pretrained parameters found in %s. Training from scratch.", path
            )

        if skipped:
            logger.info("Skipped %d keys (shape mismatch or head): %s",
                        len(skipped), skipped[:5])

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters (stem + stages)."""
        for module in [self.stem, self.stage1, self.stage2, self.stage3, self.stage4]:
            for param in module.parameters():
                param.requires_grad = False
        logger.info("Backbone frozen. Only classifier head will be trained.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen. All parameters will be trained.")

    def get_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract backbone features without classification."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.pool(x).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input ``(batch, channels, seq_len)``.

        Returns:
            Logits ``(batch, num_classes)``.
        """
        features = self.get_backbone_features(x)
        return self.head(features)

    def save_backbone(self, path: str) -> None:
        """
        Save only the backbone weights (useful for transfer learning).

        Args:
            path: Output file path.
        """
        backbone_state = {}
        for k, v in self.state_dict().items():
            if not k.startswith("head."):
                backbone_state[k] = v

        torch.save({"backbone": backbone_state}, path)
        logger.info("Backbone saved to %s (%d parameters)", path, len(backbone_state))
