import torch                                      # The main PyTorch library
import torch.nn as nn                             # Contains pytorch network building blocks (e.g., layers)
import torch.nn.functional as F                   # Contains functions for neural network operations (e.g., activation functions)

# This model is inspired by the SRCNN architecture, 
# using three layers: one for patch extraction, non-linear mapping,​ and reconstruction
class SuperResolutionModel(nn.Module):
    def __init__(self):
      # Image size is 3x32x32
      super().__init__()
      self.feature = nn.Sequential(
        # Patch extraction
        nn.Conv2d(3, 64, 5, padding=2),
        nn.ReLU(inplace=True),

        # Non-linear mapping
        nn.Conv2d(64, 64, 3, padding=1),
        nn.ReLU(inplace=True),

        # Reconstruction + ×4 upscaling
        nn.Conv2d(64, 3 * 16, 3, padding=1),
        nn.PixelShuffle(4)
    )

    def forward(self, x):
      output = self.feature(x)
      
      return output