# -------------------------
# conv_adapter_module.py
# -------------------------
# Defines ConvAdapter block for ResNet from the paper Conv-Adapter: Exploring Parameter Efficient Transfer Learning for ConvNets
# -------------------------

import torch.nn as nn

class ConvAdapter(nn.Module):
    """
    ConvAdapter is a lightweight, residual adapter module inserted into each ResNet bottleneck block.
    It consists of a 1x1 → 3x3 depthwise → 1x1 convolutional bottleneck structure.
    """
    def __init__(self, in_channels, reduction=4, kernel_size=3):
        super().__init__()
        hidden_dim = in_channels // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1)
        )

    def forward(self, x):
        return x + self.adapter(x)

