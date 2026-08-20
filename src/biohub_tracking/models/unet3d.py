"""Lightweight 3D U-Net for cell detection (optional, used for training)."""

import logging
from typing import List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — UNet3D will not be available.")


if _TORCH_AVAILABLE:

    class DoubleConv3D(nn.Module):
        """Two consecutive (Conv3D → BN → ReLU) blocks."""

        def __init__(self, in_ch: int, out_ch: int, mid_ch: Optional[int] = None):
            super().__init__()
            mid_ch = mid_ch or out_ch
            self.block = nn.Sequential(
                nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(mid_ch),
                nn.ReLU(inplace=True),
                nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)


    class Down3D(nn.Module):
        """Downsampling block: max-pool then double conv."""

        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.block = nn.Sequential(
                nn.MaxPool3d(2),
                DoubleConv3D(in_ch, out_ch),
            )

        def forward(self, x):
            return self.block(x)


    class Up3D(nn.Module):
        """Upsampling block with skip connection."""

        def __init__(self, in_ch: int, out_ch: int, bilinear: bool = True):
            super().__init__()
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
                self.conv = DoubleConv3D(in_ch, out_ch, in_ch // 2)
            else:
                self.up = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv3D(in_ch, out_ch)

        def forward(self, x1, x2):
            x1 = self.up(x1)
            # Pad if needed
            diff = [x2.size(i) - x1.size(i) for i in range(2, 5)]
            x1 = F.pad(x1, [
                diff[2] // 2, diff[2] - diff[2] // 2,
                diff[1] // 2, diff[1] - diff[1] // 2,
                diff[0] // 2, diff[0] - diff[0] // 2,
            ])
            return self.conv(torch.cat([x2, x1], dim=1))


    class UNet3D(nn.Module):
        """3D U-Net for cell detection / semantic segmentation.

        Input:  (B, 1, Z, Y, X) float32 image
        Output: (B, n_classes, Z, Y, X) logits

        Architecture:
            Encoder: 4 down-sampling stages (16 → 32 → 64 → 128 → 256)
            Decoder: 4 up-sampling stages with skip connections
        """

        def __init__(
            self,
            in_channels: int = 1,
            n_classes: int = 2,         # background + foreground
            features: List[int] = None,
            bilinear: bool = True,
        ):
            super().__init__()
            features = features or [16, 32, 64, 128, 256]

            self.inc = DoubleConv3D(in_channels, features[0])
            self.downs = nn.ModuleList([
                Down3D(features[i], features[i + 1])
                for i in range(len(features) - 1)
            ])
            self.ups = nn.ModuleList([
                Up3D(features[i + 1] + features[i], features[i], bilinear)
                for i in range(len(features) - 2, -1, -1)
            ])
            # Actually build standard UNet ups
            self.ups = nn.ModuleList()
            for i in range(len(features) - 1, 0, -1):
                self.ups.append(Up3D(features[i], features[i - 1], bilinear))

            self.outc = nn.Conv3d(features[0], n_classes, kernel_size=1)

        def forward(self, x):
            # Encoder
            skips = [self.inc(x)]
            for down in self.downs:
                skips.append(down(skips[-1]))

            # Decoder
            out = skips[-1]
            for i, up in enumerate(self.ups):
                out = up(out, skips[-(i + 2)])

            return self.outc(out)

        @torch.no_grad()
        def predict(
            self,
            img: np.ndarray,
            device: str = "cpu",
            threshold: float = 0.5,
        ) -> np.ndarray:
            """Run inference on a single 3D image.

            Args:
                img:       (Z, Y, X) numpy array, normalised to [0, 1].
                device:    "cpu" or "cuda".
                threshold: Sigmoid threshold for foreground class.

            Returns:
                Binary mask (Z, Y, X).
            """
            self.eval()
            tensor = torch.from_numpy(img[None, None]).float().to(device)
            logits = self(tensor)            # (1, n_classes, Z, Y, X)
            probs = torch.softmax(logits, dim=1)
            fg = probs[0, 1].cpu().numpy()  # foreground probability
            return (fg > threshold).astype(np.uint8)

else:
    class UNet3D:  # type: ignore[no-redef]
        """Stub when PyTorch is not installed."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required for UNet3D. "
                "Install it with: pip install torch torchvision"
            )
