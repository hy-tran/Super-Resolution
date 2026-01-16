import torch
import torch.nn as nn

# This model is inspired by the EDSR architecture,
# which uses residual blocks for enhanced feature extraction and reconstruction
class ResidualBlock(nn.Module):
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.res_scale = res_scale
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x) * self.res_scale
    

class ResidualSuperResolutionModel(nn.Module):
    def __init__(self, scale=4, num_blocks=8, channels=64):
        super().__init__()

        # Head (feature extraction)
        self.head = nn.Conv2d(3, channels, 3, padding=1)

        # Body (residual blocks)
        self.body = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # Body tail (global residual)
        self.body_tail = nn.Conv2d(channels, channels, 3, padding=1)

        # Upsampling + reconstruction
        self.tail = nn.Sequential(
            nn.Conv2d(channels, 3 * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        x_head = self.head(x)
        res = self.body(x_head)
        res = self.body_tail(res)

        x = x_head + res        # global residual
        x = self.tail(x)

        # Ensure pixel values are in the range [0, 1]
        return torch.clamp(x, 0.0, 1.0)
